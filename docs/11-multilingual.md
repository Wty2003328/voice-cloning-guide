# 11 — Multilingual: One Server, Many Languages

You need a TTS server that speaks Japanese, Chinese, and English (and
maybe more). There are two ways to do it: **one multilingual model** or
**a specialist per language**. Both ride the same vLLM-Omni Docker
deploy — what changes is the profile flag and which model loads.

## Strategy 1 — one multilingual model

A single container, one set of weights, switch language per request via
the `language` field. Lowest VRAM (one model loaded), simplest
operationally, but the per-language quality ceiling is the model's
weakest language.

### Candidate multilingual models

| Model | Profile | Languages | Notes |
|---|---|---|---|
| **VoxCPM2** (`openbmb/VoxCPM2`) | `vllm serve openbmb/VoxCPM2` | 30 languages including JA / ZH / EN / KO | Apache-2.0; ~8 GB VRAM. Widest reach. Hands-on reviews flag JA proper-noun / mixed-number cases as inconsistent. |
| **CosyVoice3** (`FunAudioLLM/Fun-CosyVoice3-0.5B-2512`) | `--profile cosy3` | 9 base langs + 18 Chinese dialects (incl. Cantonese) | Apache-2.0; CER 0.81% on native ZH. JA degrades on raw kanji (trained on kana-converted text). |
| **Qwen3-TTS** (`Qwen/Qwen3-TTS-12Hz-1.7B-Base`) | `--profile qwen` | JA / ZH / EN / KO + DE / FR / RU / PT / ES / IT | Apache-2.0. Multilingual baseline. Generic — beaten on JA by OmniVoice and on ZH by CosyVoice3. |
| **OmniVoice** (`k2-fsa/OmniVoice`) | _default_ | Multilingual but strongest on Japanese | Apache-2.0; ~7 GB system VRAM. EN / ZH are undertested in the public eval. |

### Switching language per call

Same model, just change the `language` field:

```bash
# Japanese
curl -X POST http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "input": "こんにちは、世界。",
    "language": "Japanese",
    "ref_audio": "data:audio/wav;base64,<...>",
    "ref_text": "<JA transcript of the ref>"
  }' \
  -o ja.wav

# Same ref clip, English target
curl -X POST http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "input": "Hello, world.",
    "language": "English",
    "ref_audio": "data:audio/wav;base64,<...>",
    "ref_text": "<JA transcript of the ref>"
  }' \
  -o en.wav
```

`language` is the English name (`"Japanese"`, `"English"`, `"Chinese"`,
`"Korean"`, `"German"`, ...), not BCP-47. Wrap a small code-to-name
table in your client if you prefer to configure with codes.

## Strategy 2 — per-language specialist services

A separate compose service per language, each loading the model that
performs best for it. Higher VRAM and more services to operate, but
each language gets its quality ceiling.

Recommended specialist per language (see the per-language pages for
detail):

| Language | Specialist | Profile |
|---|---|---|
| Japanese | OmniVoice (+ per-character SFT if needed) | _default_ |
| Chinese | CosyVoice3 | `--profile cosy3` |
| English | Open — see [per-language/english.md](per-language/english.md) | `--profile qwen` or default |
| Multilingual fallback | Qwen3-TTS or VoxCPM2 | `--profile qwen` |

Each specialist is its own compose service on its own port; a thin
language router in front dispatches requests by the `language` field.
Example compose fragment:

```yaml
services:
  omnivoice-ja:
    image: vllm/vllm-omni:v0.20.0
    ports: ["8001:8000"]
    command: vllm serve k2-fsa/OmniVoice --omni --host 0.0.0.0 --port 8000 ...

  cosyvoice3-zh:
    image: vllm-omni-cosy-ja:v0.20.0
    profiles: ["cosy3"]
    ports: ["8002:8000"]
    command: vllm serve FunAudioLLM/Fun-CosyVoice3-0.5B-2512 --omni --host 0.0.0.0 --port 8000 ...

  qwen3-tts-multi:
    image: vllm/vllm-omni:v0.20.0
    profiles: ["qwen"]
    ports: ["8003:8000"]
    command: vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-Base --omni --host 0.0.0.0 --port 8000 ...
```

The router code is application-specific (a 20-line FastAPI handler that
forwards by `language`). Keeping the upstream contract identical — same
OpenAI-compatible body, just different upstream URL — keeps client code
unchanged.

## Cross-lingual cloning: one ref clip, another language

A separate question from "which model": can a single Japanese reference
clip produce intelligible English / Chinese in the cloned voice?

Short answer: **content holds, accent leaks.** Modern multilingual
codebook architectures (Chatterbox-MTL v2, CosyVoice3, Qwen3-TTS)
preserve the target text faithfully when cross-lingual cloning. The
remaining cost is timbre / accent: a JA reference cloning EN produces
Japanese-accented English, especially on short utterances.

Full empirical measurements (six engine × direction combinations on a
12-prompt battery, char-jaccard + Whisper auto-language detection) live
in [ch. 14 — Cross-lingual limits](14-cross-lingual-limits.md).

Recommendation: if accent naturalness matters, use a **native
target-language reference** rather than a cross-lingual JA reference.
Cross-lingual cloning is only the right call when consistent timbre
across languages is more important than accent (one fictional character
speaking many languages, accent as a feature).

## Choosing between Strategies 1 and 2

| Priority | Pick |
|---|---|
| Lowest VRAM, lowest ops overhead | Strategy 1 — VoxCPM2 or Qwen3-TTS. |
| Highest per-language quality | Strategy 2 — OmniVoice JA + CosyVoice3 ZH + EN specialist. |
| One voice ID across languages (cross-lingual cloning) | Strategy 1 with a model that has a language-agnostic speaker encoder. |
| Mixed-language sentences ("APIを設定") | Strategy 1 — multilingual models handle in-sentence code-switching natively; specialist routers require sentence-level language detection. |

## Languages outside the supported sets

If your target is Cantonese, Vietnamese, Thai, Arabic, or Hindi:

- **Cantonese**: CosyVoice3 has first-class support via a `<|yue|>`
  dialect token (plus 17 other Chinese dialects).
- **VoxCPM2** covers 30 languages — start there for anything beyond
  the JA / ZH / EN core.
- For languages outside both: train a per-language specialist or
  accept English-mode fallback as a stopgap.

## See also

- [ch. 10 — Zero-shot cloning](10-zero-shot-cloning.md) — the base
  zero-shot request flow.
- [ch. 14 — Cross-lingual limits](14-cross-lingual-limits.md) —
  empirical accent-leak measurements.
- [ch. 15 — vLLM-Omni Docker](15-vllm-omni-docker.md) — the deploy
  walkthrough that all of the above ride on.
- [ch. 15 — Picking a model](15-vllm-omni-model-selection.md) — full
  per-model eval.
- [per-language/](per-language/) — language-specific picks and
  caveats.

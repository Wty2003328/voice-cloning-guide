# English TTS — model recommendations

**Status (2026-05): open.** English is the language where no single
vLLM-Omni-native model has clearly won the 2026 eval. Below is the
honest state and the most defensible interim picks.

## What we know

- **OmniVoice EN is undertested.** The 36k-hour pretrain is JA-dominated;
  English coverage and quality on the production eval rig haven't been
  measured at the same depth as Japanese.
- **Higgs Audio v2.5 is over budget.** It's the most-cited Apache-2.0
  English cross-lingual cloner in the 2025 leaderboards, but its
  ~10.75 GB weights exceed the 8 GB envelope this guide targets
  (and the 16 GB envelope after KV cache + Windows desktop overhead).
  **Not recommended for 16 GB consumer GPUs.**
- **Voxtral-TTS-4B (Mistral)** does support English (and other EU-centric
  multilingual coverage). It explicitly does *not* support Japanese.
- **CosyVoice3** has English in its supported set but is trained
  Chinese-first; EN is not its strongest language.

## Interim picks (2026-05)

| Approach | Model | When to pick |
|---|---|---|
| **Multilingual baseline** | Qwen3-TTS-12Hz-1.7B-Base via `--profile qwen` | Generic EN that works "out of the box" on the same vLLM-Omni Docker; ~1.7 B params; widely supported. |
| **Default OmniVoice** | `k2-fsa/OmniVoice` (default service) | EN reach is undertested but plausible — char-level Qwen3 tokenizer handles Latin natively. Try first if you're already on this model for JA. |
| **Cloud / low-latency** | OpenAI TTS via any OpenAI-compatible proxy | If local quality / latency don't meet bar and self-hosting isn't strict. |
| **Voxtral-TTS-4B** | `mistralai/Voxtral-TTS-4B` (if available) | If you specifically need EN-strong multilingual and don't need JA. |

## Candidate comparison

| Model | License | Voice clone | EN quality | VRAM | Notes |
|---|---|---|---|---|---|
| **Qwen3-TTS-12Hz-1.7B-Base** | Apache-2.0 | Zero-shot 3–30 s | Decent multilingual baseline | ~4 GB | Works in the same vLLM-Omni Docker; `--profile qwen`. |
| **OmniVoice (`k2-fsa/OmniVoice`)** | Apache-2.0 | Zero-shot + SFT | Undertested on EN; plausible | ~3 GB container delta | Default service. JA-strongest; EN coverage open. |
| **Voxtral-TTS-4B** (Mistral) | Apache-2.0 | Zero-shot | Multilingual EU-centric; supports EN | ~6–8 GB | Explicitly does not support JA. |
| **CosyVoice 3** | Apache-2.0 | Zero-shot 3–30 s | EN in supported set; ZH is the strength | ~3 GB | `--profile cosy3`. JA degrades on raw kanji. |
| **Higgs Audio v2.5** | Apache-2.0 | Zero-shot + explicit cross-lingual | Strong EN per vendor numbers | **~10.75 GB** | **Rejected for the 16 GB envelope.** |
| **Kokoro-82M** | Apache-2.0 | Limited (canonical voices) | Good EN; very fast | ~0.5 GB | Canonical voices only — not for cloning. |
| **XTTS-v2 (Coqui)** | CPML weights | Zero-shot | Solid older baseline | ~2 GB | Coqui shutdown Jan 2024; legal gray. |
| **F5-TTS / OpenF5** | F5: CC-BY-NC / OpenF5: Apache (alpha) | Zero-shot | High | ~3 GB | Mainline blocked by license; OpenF5 fork is alpha. |

## Recommendation

Start with **Qwen3-TTS via `--profile qwen`** on the same vLLM-Omni
Docker compose you'd use for JA / ZH. It's the multilingual baseline
documented in [ch. 15](../15-vllm-omni-model-selection.md); EN works
out of the box.

If quality isn't acceptable, evaluate base **OmniVoice** on your EN
prompts directly. Char-level Qwen3 tokenizer means it handles Latin
input without phonemizer-side surprises; the open question is whether
the JA-dominated pretrain leaves enough EN capacity.

If neither lands, the next options are out of the vLLM-Omni-Docker
envelope: cloud TTS (OpenAI / Cartesia / ElevenLabs via OpenAI-compatible
proxy), or a Voxtral-TTS-4B sidecar.

**Do not pick Higgs Audio v2.5 for a 16 GB consumer GPU.** Its weights
alone exceed the budget once Windows desktop or KV cache are
accounted for.

## Deploy via vLLM-Omni Docker

The fastest first cut — try Qwen3-TTS:

```bash
docker compose --profile qwen up -d
curl -s http://127.0.0.1:8000/v1/models
# → { "data": [ { "id": "Qwen/Qwen3-TTS-12Hz-1.7B-Base", ... } ] }
```

A minimal request:

```bash
curl -X POST http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d "$(cat <<'JSON'
{
  "model":     "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
  "input":     "Hello, world. This is a test of zero-shot English voice cloning.",
  "language":  "English",
  "ref_audio": "data:audio/wav;base64,<base64 of your reference clip>",
  "ref_text":  "<exact transcript of the reference clip>"
}
JSON
)" \
  -o hello.wav
```

Reference-clip tuning is the same as every other language — see
[ch. 10 — Zero-shot cloning](../10-zero-shot-cloning.md) for the recipe
(5–10 s clean clip, ~0.8 peak amplitude, exact transcript).

## See also

- [../15-vllm-omni-docker.md](../15-vllm-omni-docker.md) — production
  deploy walkthrough.
- [../15-vllm-omni-model-selection.md](../15-vllm-omni-model-selection.md)
  — per-model eval (English currently open).
- [../10-zero-shot-cloning.md](../10-zero-shot-cloning.md) —
  reference-clip recipe.
- [../models/higgs-audio.md](../models/higgs-audio.md) — Higgs Audio
  deep dive (rejected on VRAM for 16 GB envelope).
- [multilingual.md](multilingual.md) — when one model should serve EN
  + ZH + JA.

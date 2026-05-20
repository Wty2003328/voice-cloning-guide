# Multilingual TTS — when one model serves many languages

When you want **one TTS service speaking many languages** instead of N
language-specialized services. Lowest VRAM, simplest ops, but the
per-language quality ceiling is the model's weakest language.

For the broader picture (specialist-per-language vs one multilingual
model; cross-lingual cloning), see
[ch. 11 — Multilingual](../11-multilingual.md).

## When to want one multilingual model

- Tight VRAM budget — one model loaded vs N×.
- Voice-clone consistency across languages — same speaker embedding
  produces "the same voice" regardless of target language.
- Mixed-language sentences within one utterance ("APIを設定して
  ください") — multilingual models handle in-sentence code-switching
  natively.
- Deployment simplicity — one container, one set of weights, one
  update path.

If per-language quality matters more than these — read the
language-specific pages instead.

## Candidate multilingual models

| Model | Languages | Profile / command | VRAM |
|---|---|---|---|
| **VoxCPM2** (`openbmb/VoxCPM2`) | 30 languages including JA / ZH / EN / KO | `vllm serve openbmb/VoxCPM2 --omni --host 0.0.0.0 --port 8000` | ~8 GB |
| **Qwen3-TTS-12Hz-1.7B-Base** | JA / ZH / EN / KO + DE / FR / RU / PT / ES / IT (~10) | `--profile qwen` | ~4 GB |
| **OmniVoice** (`k2-fsa/OmniVoice`) | Multilingual; JA is the strongest by training-hours dominance | _default_ | ~7 GB system |
| **CosyVoice3** (`FunAudioLLM/Fun-CosyVoice3-0.5B-2512`) | 9 base + 18 Chinese dialects (incl. Cantonese) | `--profile cosy3` | ~3 GB container delta |
| **XTTS-v2** (Coqui) | 17 languages | Separate sidecar (not in vLLM-Omni-native set) | ~2 GB |

All Apache-2.0 (or equivalent permissive) except XTTS-v2 (CPML
weights; legal gray since the 2024 Coqui shutdown).

## Picking among them

| Priority | Pick | Why |
|---|---|---|
| Widest language reach (30) | **VoxCPM2** | Most languages of any current open model. |
| Balanced JA / ZH / EN baseline | **Qwen3-TTS** | Covers the big three with documented uneven-but-acceptable quality. |
| JA-strongest single model | **OmniVoice** | 36k+ hours JA pretrain; multilingual but JA wins. |
| ZH-strongest, plus Cantonese | **CosyVoice3** | CER 0.81% native ZH, `<|yue|>` dialect token. |

There is no model that wins per-language against every specialist. The
question is whether the multilingual ceiling is good enough for your
specific use case. For chat / companion / VTuber workloads it usually
is; for broadcast / audiobook in a non-strong language, switch to a
specialist.

## Cross-lingual cloning

A separate question from "which multilingual model": can a single ref
clip in language A produce intelligible speech in language B? Short
answer: **content holds, accent leaks** in modern (2025+) multilingual
codebook architectures. Empirical measurements at
[ch. 14 — Cross-lingual limits](../14-cross-lingual-limits.md).

## Deploy

Same vLLM-Omni Docker compose as every other model. To bring up Qwen3-TTS
as a multilingual generalist:

```bash
docker compose --profile qwen up -d
curl -s http://127.0.0.1:8000/v1/models
# → { "data": [ { "id": "Qwen/Qwen3-TTS-12Hz-1.7B-Base", ... } ] }
```

Switch language per request via the `language` field — see
[ch. 11 — Multilingual](../11-multilingual.md) for the same-model
multi-language request pattern.

## See also

- [../11-multilingual.md](../11-multilingual.md) — strategies and
  cross-lingual cloning.
- [../15-vllm-omni-docker.md](../15-vllm-omni-docker.md) — production
  deploy walkthrough.
- [../15-vllm-omni-model-selection.md](../15-vllm-omni-model-selection.md)
  — per-model eval.
- [japanese.md](japanese.md), [chinese.md](chinese.md), [english.md](english.md)
  — per-language specialist picks.

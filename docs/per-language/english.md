# English TTS — model recommendations

**Recommended pick:** [**Higgs Audio v2.5 (1B)**](../models/higgs-audio.md).
**Status:** Validated by 2026-05 research (task #141). Prototype planned.

## What makes English TTS comparatively easier

- **Training data abundance.** LibriTTS (585h), VCTK, plus internal
  Big Tech data — nearly every multilingual model treats English as
  primary.
- **Mature phonemization** — ARPABET/IPA, stress prediction libs,
  prosody tagging.

The hard parts:
- **OOV names + abbreviations** ("PostgreSQL", brand names)
- **Code-mixing** with foreign words
- **Per-voice consistency** under zero-shot

## Candidate comparison

| Model | License | Voice clone | EN quality | RTF (RTX 5080) | Notes |
|---|---|---|---|---|---|
| **Higgs Audio v2.5 (1B)** ✅ | Apache-2.0 | Zero-shot 3-30s **with explicit cross-lingual support** | 75.7% win vs gpt-4o-mini-tts on EmergentTTS-Eval Emotions | ~0.5 | Only pick with explicit cross-lingual voice clone (matters for using one shared reference clip across all 3 engines) |
| Kokoro-82M | Apache-2.0 | Limited (canonical voices) | Good | ~0.05 (extremely fast) | Best for low-latency EN-only; limited voice variety |
| CosyVoice 3 | Apache-2.0 | Zero-shot | Strong EN | ~0.5 (or 0.10 with TRT-LLM) | Same model serves ZH well too — share-the-engine option |
| XTTS-v2 (Coqui) | CPML weights | Zero-shot | Solid older baseline | ~0.8 | Coqui shutdown Jan 2024 — paid tier defunct, legal gray |
| F5-TTS / OpenF5 | F5 CC-BY-NC ❌ / OpenF5 Apache (alpha) | Zero-shot | High | ~0.3 | Mainline blocked; OpenF5 fork too alpha to recommend |
| Sesame CSM-1B | Apache-2.0 | Limited | Decent | ~0.4 | Newer, smaller community |
| Step-Audio 2-mini | Apache-2.0 | Zero-shot | Decent | TBD | Earlier-stage, less proven |

## Recommendation

**Higgs Audio v2.5 (1B condensed from 3B)** for our companion.

**Why it wins:**
- Apache-2.0 (commercial OK)
- **Only candidate with explicit cross-lingual voice clone** — same
  speaker_embedding maintains voice identity across JA / EN / ZH.
  This matters because the multi-engine architecture wants the character to
  sound consistent across languages; most other models' speaker
  encoders are language-locked.
- 75.7% emotion-aware quality win vs gpt-4o-mini-tts on EmergentTTS-Eval
  Emotions benchmark
- 1B-param condensed from 3B → fits comfortably alongside the JA+ZH
  engines in 16 GB VRAM

**Caveat to verify:** Higgs v2.5 RTF on Blackwell — only RTX 4090
numbers are published. Expect similar or slightly faster; need on-hardware
measurement.

**Alternative if EN latency matters more than quality:** Kokoro-82M.
At RTF ~0.05 (20× real-time), it's the fastest open-source EN TTS.
Voice variety limited but adequate for non-character chat.

## See also

- [../models/higgs-audio.md](../models/higgs-audio.md) — full model deep-dive (coming)
- [../models/kokoro.md](../models/kokoro.md) — fast alternative (coming)
- [../deployment/multi-engine.md](../deployment/multi-engine.md) — multi-engine router
- [../models/cosyvoice-3.md](../models/cosyvoice-3.md) — share-the-engine option (CosyVoice for ZH+EN)

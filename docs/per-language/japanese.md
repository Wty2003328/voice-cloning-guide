# Japanese TTS — model recommendations

**Recommended pick:** [**Style-Bert-VITS2 JP-Extra**](../models/style-bert-vits2.md) (fine-tuned on your character's voice).
**Status:** Validated by 2026-05 research (task #141). Prototype planned for Asuna voice next.

## What makes Japanese TTS hard

- **Pitch accent is lexically distinctive.** 橋(hashi-bridge) vs
  箸(hashi-chopsticks) vs 端(hashi-edge) — same phonemes, different
  pitch contours, different meanings. Most multilingual models get
  this wrong because their primary training language (English/Chinese)
  doesn't use pitch accent.
- **Kanji readings are context-dependent.** 行く=iku, 行う=okonau,
  行=gyō or kō — the model has to disambiguate from context.
- **Numbers and dates need normalization.** "2024年" should be read
  "nisen-niju-yo-nen", not "ni-zero-ni-yon-nen". Models without
  Japanese-aware number normalization (most multilingual ones)
  hallucinate or run away on numeric input.
- **Mora-timed prosody.** Different from English's stress timing —
  affects natural rhythm.

## Candidate comparison

| Model | License | Voice clone | JA quality | Native phonemization | Deploy difficulty |
|---|---|---|---|---|---|
| **Style-Bert-VITS2 JP-Extra** ✅ | AGPL-3.0 code; JP-Extra base trained on JSUT (Apache audio data) | Per-voice fine-tune (~1-2 days RTX 5080) | **Best** — MOS 4.37 vs human 4.38 on char-style JA; rule-based pitch accent via pyopenjtalk + Unidic | Yes (pyopenjtalk + Unidic) | 2/5 — pip-installable but AGPL-3.0 needs IPC sidecar for commercial use |
| Qwen3-TTS-12Hz-1.7B-Base | Apache-2.0 | Zero-shot | Mediocre — generic multi-lang, digit runaway, EOS misfires | No (model handles raw text) | 1/5 — pip-installable, native Windows |
| VOICEVOX | LGPL code / per-voice non-commercial | NO (canonical voices only) | Excellent for canonical voices | Yes (VOICEVOX-engine) | 3/5 — Docker container for engine |
| Aivisspeech | Style-BERT-VITS2 fork, AGPL | Per-voice fine-tune | Equivalent to SBV2 | Yes | 2/5 — similar to SBV2 |
| **Irodori-TTS-500M-v3** | MIT | Per-voice (?) | Promising but very fresh (2026-05) | TBD | TBD — re-evaluate Q3 2026 |

## Recommendation

For our commercial-OK companion: **Style-Bert-VITS2 JP-Extra**, deployed as an isolated IPC sidecar (to respect the AGPL-3.0 boundary — your closed-source app communicates with the sidecar over HTTP, the sidecar stays Apache-ish-isolated).

**Why it wins:**
- Only contender where pitch accent is a **rule-based input** (pyopenjtalk + Unidic), not "learn-and-hope"
- MOS 4.37 vs human reference 4.38 on character-style JA (essentially indistinguishable in blind tests)
- Per-voice fine-tune on the existing Asuna 168-clip pool (~1-2 days on RTX 5080) — uses the same diverse5 reference data you already have

**Caveat to verify:** SBV2 JP-Extra base-model commercial use needs explicit clarification from maintainer litagin02. The CODE license is AGPL-3.0 (commercial use OK if you keep the sidecar separate); the base model was trained on JSUT (Apache audio), so derivative models should be OK, but worth a maintainer check before shipping.

## Voice cloning recipe

1. Reference clip pool: existing `GPT-SoVITS/logs/asuna_combined/0_sliced/` (168 clips, ~7s avg)
2. Train SBV2 JP-Extra on those clips (~1-2 days)
3. Sidecar wraps the trained model behind the [TTS Provider Spec v1](../../zeroclaw-companion/docs/TTS-PROVIDER-SPEC.md) (same `/v1/audio/speech` contract as Qwen3-TTS)
4. Router (see [multi-engine.md](../deployment/multi-engine.md)) directs JA requests to this sidecar; falls back to Qwen3-TTS for other languages

## See also

- [../models/style-bert-vits2.md](../models/style-bert-vits2.md) — full model deep-dive
- [../models/qwen3-tts.md](../models/qwen3-tts.md) — multilingual fallback baseline
- [../deployment/multi-engine.md](../deployment/multi-engine.md) — multi-engine router architecture
- [../13-inference-optimization.md](../13-inference-optimization.md) — kernel-level perf (applies to Qwen3-TTS JA fallback)

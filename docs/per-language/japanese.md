# Japanese TTS — model recommendations

**Top pick (2026-05):** [**OmniVoice fine-tuned on your character**](../16-omnivoice-sft-recipe.md), served via [vLLM-Omni in Docker](../15-vllm-omni-docker.md). 36k hrs JA pretrain + 400-step SFT on ~10–20 min of target voice; 8-minute training run; ~7 GB system VRAM.

**Runner-up:** Style-Bert-VITS2 JP-Extra. Holds the human-parity MOS 4.37 baseline on anime-character benchmark, but non-AR flow architecture means it can't ride the vLLM-Omni Docker deploy flow — kept as a sidecar option in [`docs/models/style-bert-vits2.md`](../models/style-bert-vits2.md).

**Why the pivot:** the production goal in this guide shifted to "one Docker container per TTS server, OpenAI-compatible API." OmniVoice is the only vLLM-Omni-native model with documented multi-tens-of-kilohours JA training + a char-level tokenizer (no kanji-byte-fallback trap — see the empirical eval at [`docs/15-vllm-omni-model-selection.md`](../15-vllm-omni-model-selection.md) for why CosyVoice3 fails this gate).

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

## Candidate comparison (2026-05 update)

| Model | License | Voice clone | JA quality (measured) | Deploy path |
|---|---|---|---|---|
| **OmniVoice + per-character SFT** ✅ | Apache-2.0 base; SFT weights as-you-license-them | Per-voice SFT (~8 min train + 5 min deploy) | Mean jaccard 0.96-0.97 on 21-prompt battery; timbre clearly better than zero-shot per A/B listen | **vLLM-Omni Docker — production** |
| OmniVoice base (no SFT) | Apache-2.0 | Zero-shot from `ref_audio` | Mean jaccard 0.95-0.96; timbre "in the ballpark but not your character" | `docker compose up` (default) |
| Style-Bert-VITS2 JP-Extra (sidecar) | AGPL-3.0 code; JP-Extra base on JSUT | Per-voice fine-tune (~1-2 days train) | **Best published**: MOS 4.37 vs human 4.38 on char-style JA; rule-based pitch accent via pyopenjtalk + Unidic | Non-AR flow architecture — can't run on vLLM-Omni cleanly; needs separate sidecar |
| CosyVoice3 (Fun-0.5B-2512) | Apache-2.0 | Zero-shot | Mean jaccard 0.37 — **content drift on long-form / numbers / paragraphs**. Even with a fugashi kanji→kana adapter patch (see ch. 15 selection notes) only short cleanly-phrased input passes. | `docker compose --profile cosy3 up` — keep around for Chinese, not JA |
| Qwen3-TTS-12Hz-1.7B-Base | Apache-2.0 | Zero-shot | Mediocre — generic multi-lang, digit runaway, EOS misfires | `docker compose --profile qwen up` — multilingual baseline only |
| Fish-Speech S2 Pro | Open weights | Zero-shot | Best published JA paper numbers (CV3-Eval CER 3.96%) — but vLLM-Omni v0.20 has `ModuleNotFoundError: fish_speech` and v0.21.0rc1 is broken differently. **Blocked.** | Watch upstream issue #2404. |
| VoxCPM2 | Apache-2.0 | Zero-shot | 30 langs incl. JA. Not yet tested in this guide's eval; hands-on reviews flag JA proper nouns + mixed numbers as inconsistent. | `vllm serve openbmb/VoxCPM2` — fallback candidate |
| VOICEVOX | LGPL code / per-voice non-commercial | NO (canonical voices only) | Excellent for canonical voices | Separate Docker; non-cloning |
| Aivisspeech | SBV2 fork, AGPL | Per-voice fine-tune | Equivalent to SBV2 | Non-AR, same constraints as SBV2 |
| Irodori-TTS-500M-v3 | MIT | Per-voice (?) | Untested; very fresh | TBD — re-evaluate Q3 2026 |
| IndexTTS-2 | bilibili proprietary | Zero-shot | 42k hrs JA paper training; SS 0.833 / WER 9.95% on Common Voice JA. Port to vLLM-Omni is multi-week engineering (GPT-2 backbone, embedding-prefix inputs, global ODE solve). | Parked. Reactivate only if OmniVoice falls down on a future use case. |

## Recommendation

**Ship OmniVoice + 8-minute per-character SFT, deploy via vLLM-Omni Docker.** Full recipe at [`docs/16-omnivoice-sft-recipe.md`](../16-omnivoice-sft-recipe.md).

**Why this wins now:**
- One Docker service, one OpenAI-compatible URL, ~7 GB system VRAM, ~8 min training run — way less infrastructure than the SBV2 sidecar route.
- 36k hours of JA pre-training means the model already knows Japanese phonotactics; SFT just shifts the speaker prior toward your character.
- Verified content fidelity (mean jaccard 0.96-0.97 across 21 prompt categories) + verified timbre lift over base zero-shot.

**Why SBV2 is still in the table:** the published MOS 4.37 character-voice number is unbeaten on a head-to-head subjective listen. If your project specifically needs that ceiling and you can pay for a separate sidecar service outside the vLLM-Omni Docker, the SBV2 path in [`docs/models/style-bert-vits2.md`](../models/style-bert-vits2.md) still applies. For everyone else: OmniVoice SFT is the path that ships.

## Voice cloning recipe

See [`docs/16-omnivoice-sft-recipe.md`](../16-omnivoice-sft-recipe.md) for the full step-by-step. Quick version:

1. Slice + ASR + clean ~10–20 min of your target voice (any TTS dataset format works — the pipe-separated `asr.list` shown in ch. 16 step 1 is the easiest).
2. Build JSONL train + dev manifests.
3. Stage 0: audio tokenization (~30 s for 137 clips).
4. Stage 1: 400-step full FT, LR 5e-6, batch_tokens 1024, SDPA attn. ~8 min wallclock on RTX 5080.
5. Copy `audio_tokenizer/` from the base HF cache into your checkpoint dir.
6. Mount the checkpoint into the vLLM-Omni docker compose and `docker compose up -d`.

## See also

- [`../15-vllm-omni-docker.md`](../15-vllm-omni-docker.md) — production deploy walkthrough.
- [`../15-vllm-omni-model-selection.md`](../15-vllm-omni-model-selection.md) — the full eval that drove the OmniVoice pick.
- [`../16-omnivoice-sft-recipe.md`](../16-omnivoice-sft-recipe.md) — fine-tune walkthrough (this is where the production setup lives).
- [`../14-cross-lingual-limits.md`](../14-cross-lingual-limits.md) — cross-lingual cloning empirical limits.
- [`../models/style-bert-vits2.md`](../models/style-bert-vits2.md) — SBV2 deep dive (now legacy / fallback).
- [`../models/qwen3-tts.md`](../models/qwen3-tts.md) — multilingual baseline (now `--profile qwen`).

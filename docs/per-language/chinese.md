# Mandarin Chinese TTS — model recommendations

**Recommended pick:** [**CosyVoice 3 (Fun-CosyVoice3-0.5B-2512)**](../models/cosyvoice-3.md).
**Status:** Validated by 2026-05 research (task #141). Prototype planned via WSL2 + TensorRT-LLM.

## What makes Chinese TTS hard

- **Tones are lexically distinctive.** 妈/麻/马/骂 (mā/má/mǎ/mà) —
  same syllable, four tones, four meanings.
- **Polyphonic characters.** 行 = xíng (walk) or háng (row).
  Disambiguation requires context understanding.
- **No spaces.** Sentence segmentation depends on the model's
  understanding of word boundaries.
- **Code-mixing.** Modern Chinese text often mixes English loanwords
  (APP, OK, WiFi) — model has to switch phoneme systems mid-sentence.

## Candidate comparison

| Model | License | Voice clone | ZH quality | Deploy difficulty |
|---|---|---|---|---|
| **CosyVoice 3 (Fun-0.5B-2512)** ✅ | Apache-2.0 (clean) | Zero-shot 3-30s | **Best** — CER 0.81%, SIM 78% (beats human ref 75.5%) | 3/5 PyTorch; 5/5 with TRT-LLM (WSL2 + custom plugin) |
| GPT-SoVITS v4 | MIT | Per-voice fine-tune | Strong ZH (Chinese community origin) | 3/5 — multi-step training pipeline |
| Qwen3-TTS-12Hz-1.7B-Base | Apache-2.0 | Zero-shot | Decent multilingual baseline | 1/5 — pip + native Windows |
| Spark-TTS | CC-BY-NC-SA ❌ | Zero-shot | Strong | Blocked: license flipped from Apache to non-commercial |
| ChatTTS (weights) | CC-BY-NC ❌ | Limited | Conversational style | Blocked: non-commercial weights |

## Recommendation

**CosyVoice 3 Fun-0.5B-2512** for commercial-OK ZH at SOTA quality.

**Why it wins:**
- Apache-2.0 weights (clean commercial use)
- Alibaba-native training data → real Chinese language depth (vs
  multilingual generalists)
- CER 0.81% on Chinese test sets (industry-leading)
- Speaker SIM 78% — *better than the human reference at 75.5%* (i.e.
  the model's clones are more consistent with the speaker's identity
  than a different recording of the same speaker would be)
- Optional TensorRT-LLM gives **4× speedup → RTF ~0.10 on RTX 5080**
  (5× faster than realtime)

**Caveat to verify:** The `ttsfrd` text-frontend dependency caused a
prior trial to fail (see project history). Spike: 1 day to re-verify
deployment on current versions before committing to the full
integration.

## Deployment paths

| Path | Effort | RTF on RTX 5080 | When to pick |
|---|---|---|---|
| PyTorch native | 1-2 days | ~0.5 | Windows, prefer simplicity |
| TensorRT-LLM via WSL2 | 5-7 days | ~0.10 | Production, need lowest latency, OK with WSL2 |
| Docker (CosyVoice's official container) | 2-3 days | ~0.5 | Want isolation, OK with Docker |

For our companion: PyTorch native first to validate quality, then
optionally migrate to TRT-LLM for production speed.

## Voice cloning

Zero-shot — the same reference clip clip (`target_concat_diverse5.wav`)
can be reused. CosyVoice 3's speaker encoder produces a different
embedding than Qwen3-TTS / SBV2 / Higgs, so the voice timbre will
differ slightly across engines. Expected ~10-15% drift; blind A/B
test required before shipping multi-engine.

## See also

- [../models/cosyvoice-3.md](../models/cosyvoice-3.md) — full model deep-dive
- [../deployment/multi-engine.md](../deployment/multi-engine.md) — multi-engine router (CosyVoice 3 as the ZH engine)
- [../models/qwen3-tts.md](../models/qwen3-tts.md) — multilingual fallback

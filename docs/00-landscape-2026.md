# 00 — The TTS Landscape (2026)

A taxonomy of open-source text-to-speech as of 2026, why so many
models exist, and how to navigate the trade-offs. Read this before
picking a model.

## The core split: AR vs non-AR

Modern neural TTS divides cleanly into two architectural families:

### Autoregressive (AR) — generate audio one token at a time

The model predicts the next audio codec token given everything before
it. Examples: VALL-E, Qwen3-TTS, GPT-SoVITS, CosyVoice, XTTS-v2,
IndexTTS, Spark-TTS, Higgs Audio.

**Pros**
- Generally higher acoustic quality
- Natural at modeling speaker style + prosody
- Zero-shot voice cloning works well (model conditions on reference audio)
- Continuous improvement: scales like LLMs

**Cons**
- Slow without optimization (~RTF 2-3 baseline; sub-RTF requires kernel work)
- Stochastic failures: **runaway loops** (model gets stuck repeating tokens),
  **stochastic EOS** (model stops too early or too late)
- Quality drops on out-of-distribution input (rare characters, numbers,
  abbreviations)

### Non-autoregressive (non-AR) — generate all audio at once

The model produces the entire utterance in a single forward pass.
Subdivides into:
- **Diffusion** — denoise from noise to mel/audio (NaturalSpeech 3,
  StyleTTS2)
- **Flow matching** — learn the continuous flow from noise to data
  (F5-TTS, OpenF5, MaskGCT)
- **VAE-based / DDM** — earlier generation (FastSpeech 2, VITS,
  YourTTS)

**Pros**
- Steady output: no AR loops, no stochastic EOS
- Often faster than naive AR (single forward pass)
- More predictable failure modes

**Cons**
- Lower ceiling on speaker fidelity (harder to fine-tune for new voices)
- Diffusion variants need many denoising steps to reach quality (10-50)
- Smaller community vs AR (less battle-tested code)

**Implication:** for an interactive companion where occasional weird
outputs are jarring, non-AR is appealing. For best per-voice quality,
AR with safety nets (repetition guard, max_new_tokens cap, ASR-validate)
is still ahead.

## The license fragmentation problem

As of 2026, the SOTA quality leaders are NOT commercially usable
without paying:

| Model | Code license | Weights license | Commercial use |
|---|---|---|---|
| IndexTTS 2.5 | Apache-2.0 | **Non-commercial only** | ❌ |
| F5-TTS | MIT | **CC-BY-NC** | ❌ |
| MaskGCT | Apache-2.0 | **CC-BY-NC-4.0** | ❌ |
| Spark-TTS | (was Apache, flipped to **CC-BY-NC-SA**) | CC-BY-NC-SA | ❌ |
| Coqui XTTS-v2 | MPL-2.0 | CPML (commercial OK with attribution) | ⚠️ with attribution |
| Qwen3-TTS-1.7B | Apache-2.0 | Apache-2.0 | ✅ |
| CosyVoice 3 | Apache-2.0 | Apache-2.0 | ✅ |
| GPT-SoVITS v4 | MIT | MIT | ✅ |
| Style-Bert-VITS2 | MIT | MIT (community-trained voices vary) | ✅ |
| Kokoro-82M | Apache-2.0 | Apache-2.0 | ✅ |
| Higgs Audio v2 / v2.5 | Apache-2.0 | Apache-2.0 | ✅ |
| Sesame CSM-1B | Apache-2.0 | Apache-2.0 | ✅ |
| OpenF5 (F5-TTS Apache fork) | Apache-2.0 | Apache-2.0 (alpha) | ✅ (status: alpha) |
| Step-Audio 2-mini | Apache-2.0 | Apache-2.0 | ✅ |
| VOICEVOX | LGPL | Per-voice (most **non-commercial**) | ⚠️ canonical voices only |

**Practical consequence:** if you're shipping a commercial product, your
shortlist is much smaller than the Hugging Face leaderboard suggests.
A model topping benchmarks is irrelevant if it can't be shipped.

## The multilingual-vs-specialized trade-off

### Multilingual generalist (one model, many languages)

Examples: Qwen3-TTS, XTTS-v2, CosyVoice 3, MeloTTS.

- **One model** loaded, one weight set, one tokenizer.
- VRAM-efficient: 1-3 GB total
- Voice consistency across languages: same speaker embedding works for
  all langs in the model's training set
- Quality: **language-by-language is uneven.** The model's primary
  training language (usually English or Chinese) sounds best;
  secondary languages (Japanese, Korean) often have prosody/accent
  errors

### Language-specialized (one model per language)

Examples: Style-Bert-VITS2 (JA), GPT-SoVITS (multi but ZH-strongest),
Kokoro (EN), VOICEVOX (JA-only).

- **N models loaded**, one per language
- VRAM scales with language count: ~1 GB per model
- Voice consistency across languages: lost — each engine's speaker
  encoder differs, so the same reference produces different "interpretation"
  of the voice in each language
- Quality: **better in the target language.** Native phonemization,
  language-specific training data, community focused on the language

The user's companion-app pattern (chat replies arriving in JA, ZH, or
EN depending on user input) is exactly the case where specialized
wins. A multi-engine sidecar router (see
[deployment/multi-engine.md](deployment/multi-engine.md)) lets you keep
the per-language quality without dragging the rest of the stack into
per-engine complexity.

## The phonemization layer matters

What the TTS model sees as input is NOT raw text. There's a
preprocessing pipeline:

```
raw text → normalize (digits, dates, abbreviations)
        → phonemize (text → phoneme sequence with prosody markers)
        → tokenize (phonemes → integer IDs the model embeds)
```

**Models that do their own phonemization** (Qwen3-TTS, CosyVoice,
XTTS-v2 to some extent) accept raw text and figure out readings via
the trained model. Convenient, but fails on out-of-distribution:
"2024年" might be read as "ni-zero-ni-yon-nen" instead of
"nisen-niju-yo-nen" because the model never learned the year convention.

**Models that require external phonemization** (Style-Bert-VITS2 via
pyopenjtalk, GPT-SoVITS via cnHuBert+BERT, VOICEVOX via VOICEVOX-engine)
take pre-processed input. More robust on edge cases, less convenient
because you need to deploy the phonemizer.

For Japanese specifically, pyopenjtalk / MeCab + pitch-accent
dictionaries are essentially mandatory for production quality.
Models that skip this step (Qwen3-TTS) handle the common case but
break on numerics — exactly the failure class we hit in production
([13-inference-optimization.md](13-inference-optimization.md) covers
the fix: pre-normalize digit runs to kana before feeding the model).

## Deployment difficulty axes

When evaluating a model, the four practical questions:

1. **Install dependencies** — does it pip-install cleanly? Or need
   custom CUDA builds, conda envs, system libraries?
2. **Model weights** — Hugging Face / GitHub release / Modelscope?
   How big? Mirror availability?
3. **Inference framework** — pure PyTorch? Needs ONNX runtime,
   TensorRT-LLM, ctranslate2, vLLM, custom Triton kernels?
4. **Voice cloning workflow** — pass reference WAV at inference time
   (zero-shot)? Train per-voice (LoRA, full fine-tune)? Use canonical
   speakers only?

A pip-installable Apache-2.0 zero-shot model that runs on PyTorch
out-of-the-box (Qwen3-TTS) is at one extreme. A model requiring
TensorRT-LLM custom plugin + Linux + few-shot voice training
(CosyVoice 3 at peak performance) is at the other.

**Rough difficulty scale** (1 = easiest, 5 = hardest to deploy on
Windows):

| Model | Difficulty | Notes |
|---|---|---|
| Qwen3-TTS-12Hz-1.7B-Base | 1 | `pip install qwen-tts`, model from HF, runs on Windows+RTX 5080 native |
| Kokoro-82M | 1 | Small, pip-installable, pure PyTorch |
| Style-Bert-VITS2 | 2 | Pip-installable but needs pyopenjtalk + JA-specific assets |
| GPT-SoVITS v4 | 3 | Multi-step training pipeline, complex feature extraction |
| CosyVoice 3 (PyTorch) | 3 | Pip-installable but heavy deps; native PyTorch slow |
| CosyVoice 3 (TensorRT-LLM) | 5 | Linux/WSL2, custom plugin compile, hardest path but 4-10× faster |
| F5-TTS / MaskGCT | 4 | License issues + custom training pipelines |
| VOICEVOX | 4 | Need separate VOICEVOX-engine (Docker) for phonemization |

## How to use this guide

- **You have a target language and need to ship**: go to
  [docs/per-language/](per-language/) and pick the recommended model
- **You have a specific model in mind and want details**: go to
  [docs/models/](models/) for deep dives
- **You're shipping a multi-language product**: read
  [docs/deployment/multi-engine.md](deployment/multi-engine.md) for
  the multi-engine router architecture
- **You want kernel-level perf**: read
  [docs/13-inference-optimization.md](13-inference-optimization.md)
  for general autoregressive-TTS optimization tactics (~6× speedup
  recipe on Qwen3-TTS, generalizable)
- **You want zero-shot voice cloning specifically**: read
  [docs/10-zero-shot-cloning.md](10-zero-shot-cloning.md)
- **You want to fine-tune your own character voice**: see
  [models/gpt-sovits-v4.md](models/gpt-sovits-v4.md) — full recipe + architecture + extending

## Where this guide is going

The per-model and per-language pages are filling in as research lands.
Status:

- ✅ **Qwen3-TTS** — covered (zero-shot recipe + optimization)
- ✅ **GPT-SoVITS v4** — covered (fine-tune pipeline)
- 🚧 **Style-Bert-VITS2** — research in progress, prototype planned
- 🚧 **CosyVoice 3** — research done (multi-engine.md), prototype planned
- 🚧 **Kokoro-82M** — research in progress
- 📋 **F5-TTS / OpenF5, XTTS-v2, Higgs Audio v2, Sesame CSM-1B,
  IndexTTS, MaskGCT, Step-Audio** — research-only (license-blocked or
  awaiting validation)

The expanded guide is a snapshot of a fast-moving field. Dates on
sections tell you when each was last validated.

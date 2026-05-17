# Open-Source TTS Guide (2026)

A vendor-neutral, hands-on comparison of open-source text-to-speech
models for production use. Picks the right model per language, per
license, per deployment constraint — and shows you how to actually run
it.

This started as a GPT-SoVITS fine-tuning tutorial, expanded to cover
Qwen3-TTS zero-shot cloning, and is now reorganized into a **broader
TTS atlas** because no single model wins across all languages and use
cases. The original recipes still live here (see [Path A](#path-a--zero-shot-cloning)
and [Path B](#path-b--fine-tune-training)) — what's new is the
landscape view and per-language model selection guidance.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## What this guide answers

| Question | Where to look |
|---|---|
| "Which model should I use for Japanese / Chinese / English?" | [docs/per-language/](docs/per-language/) |
| "What are the trade-offs of model X?" | [docs/models/](docs/models/) |
| "Why are there so many TTS models?" (AR vs non-AR, multilingual vs specialized, licenses) | [docs/00-landscape-2026.md](docs/00-landscape-2026.md) |
| "How do I deploy multiple models behind one API?" | [docs/deployment/multi-engine.md](docs/deployment/multi-engine.md) |
| "How do I clone a voice from a short reference?" | [docs/10-zero-shot-cloning.md](docs/10-zero-shot-cloning.md) (Qwen3-TTS recipe) |
| "How do I fine-tune a custom voice model?" | [docs/models/gpt-sovits-v4.md](docs/models/gpt-sovits-v4.md) (GPT-SoVITS recipe) |
| "How do I get sub-real-time inference?" | [docs/13-inference-optimization.md](docs/13-inference-optimization.md) |

## TL;DR — recommendations as of 2026-05

| Language | Top pick | Why | License |
|---|---|---|---|
| **Japanese** | [**Style-Bert-VITS2 JP-Extra**](docs/models/style-bert-vits2.md) (fine-tuned per voice) | Rule-based pitch accent via pyopenjtalk+Unidic (not "learn-and-hope"); MOS 4.37 vs human 4.38 | AGPL-3.0 code (commercial OK via IPC sidecar) |
| **Chinese** | [**CosyVoice 3 (Fun-0.5B-2512)**](docs/models/cosyvoice-3.md) | Apache, CER 0.81%, SIM 78% (beats human ref); 4× speedup with TRT-LLM (RTF 0.10 on 5080) | Apache-2.0 |
| **English** | [**Higgs Audio v2.5 (1B)**](docs/models/higgs-audio.md) | Apache; only candidate with explicit cross-lingual voice clone (keeps voice ID across all 3 engines); 75.7% EmergentTTS-Eval emotions win vs gpt-4o-mini-tts | Apache-2.0 |
| **Multilingual single-model** (all three in one) | [**Qwen3-TTS-12Hz-1.7B-Base**](docs/models/qwen3-tts.md) | Apache, true zero-shot, sub-real-time after kernel-opt (~6× over baseline), but JA pitch-accent imperfect and digit-runaway needs pre-normalization | Apache-2.0 |
| **Best zero-shot voice clone** (3-30s reference → any voice) | Qwen3-TTS or Higgs Audio v2.5 | Both Apache, no training, multilingual | Apache-2.0 |
| **Best per-character voice quality** (10+ min training audio) | GPT-SoVITS v4 with LoRA | MIT, learns speaker prosody not just timbre | MIT |

Architecture for multi-language deployments:
**3 sidecars, lazy-load + LRU** (see [multi-engine.md](docs/deployment/multi-engine.md)).
Full research report at [tts_lab/research_per_language_tts_2026.md](https://github.com/Wty2003328/gpt-sovits-voice-cloning-guide)
(60+ cited sources, 9 candidates analyzed, 6 honorable mentions evaluated and excluded).

## Why no single model wins

- **Training data is language-skewed.** English-centric models treat
  JA/ZH as second-class even when nominally "multilingual." Native-JA
  models (VOICEVOX, Style-Bert-VITS2) get prosody right; multilingual
  models trip on pitch accent.
- **Phonemization is language-specific.** Japanese needs MeCab +
  pitch-accent dictionaries. Chinese needs tone marking. English needs
  stress prediction. Generic char-level tokenization works for the
  common case but fails on edge inputs (dates, numbers, abbreviations).
- **License fragmentation.** SOTA quality often comes with CC-BY-NC
  (IndexTTS 2.5, F5-TTS, MaskGCT, Spark-TTS). Commercial-OK weights are
  rarer and usually 1-2 generations behind SOTA.
- **Different architectures have different failure modes.**
  Autoregressive models can loop / run away on out-of-distribution
  input. Non-autoregressive (diffusion / flow-matching) models are
  steadier but harder to fine-tune for new voices.

See [docs/00-landscape-2026.md](docs/00-landscape-2026.md) for the full
architectural taxonomy and the trade-off matrices.

## Path A — Zero-shot cloning (give me a voice in 3 seconds)

The Qwen3-TTS quickstart. Single Apache-2.0 model handles JA/ZH/EN with
acceptable quality. Best when you want to copy an existing voice with
no training pipeline.

```bash
pip install -U qwen-tts huggingface_hub soundfile
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-Base --local-dir ./qwen3-tts-1.7b-base
python scripts/zero_shot_clone.py \
    --model-dir ./qwen3-tts-1.7b-base \
    --reference my_voice.wav \
    --reference-text "Hello, this is my reference recording." \
    --text "I can now speak any text you want." \
    --language English \
    --out cloned.wav
```

Full tutorial: [docs/10-zero-shot-cloning.md](docs/10-zero-shot-cloning.md).

## Path B — Fine-tune training (best per-character quality)

The GPT-SoVITS v4 LoRA pipeline. Needs ~10-20 minutes of training audio
+ ~30-60 min training on RTX 5080. Highest quality for a specific
character voice.

```bash
python scripts/demucs_isolate.py --input video_audio.wav --output speaker_vocals.wav
cd scripts
python 01_slice_audio.py      --vocals ../speaker_vocals.wav --exp my_speaker
python 02_asr_transcribe.py   --exp my_speaker --lang ja
python 03_extract_features.py --exp my_speaker
python 04_extract_semantic.py --exp my_speaker
python 05_train_sovits_v4.py  --exp my_speaker --epochs 20 --lora-rank 32
python 06_train_gpt.py        --exp my_speaker --epochs 15 --pretrained-version v4
python 07_inference_v4.py     --exp my_speaker --lang ja --text "こんにちは！" \
    --ref-wav ../GPT-SoVITS/logs/my_speaker/0_sliced/0003.wav \
    --ref-text "ここは私に任せて私を選んでくれる" --ref-lang ja \
    --out hello.wav
```

Full tutorial: [docs/models/gpt-sovits-v4.md](docs/models/gpt-sovits-v4.md)
(consolidated deep dive — recipe + architecture + extending) plus the
training scripts in [`scripts/`](scripts/).

## Documentation map

### Landscape + decision-making
- [docs/00-landscape-2026.md](docs/00-landscape-2026.md) — TTS architectures, license decision matrix, why so many models
- [docs/per-language/](docs/per-language/) — model recommendations per language
- [docs/models/](docs/models/) — deep dive per model (pros/cons, deployment, quality)

### Practical recipes
- [docs/10-zero-shot-cloning.md](docs/10-zero-shot-cloning.md) — Qwen3-TTS quickstart and tuning
- [docs/11-multilingual.md](docs/11-multilingual.md) — cross-lingual cloning (one model, many languages)
- [docs/models/gpt-sovits-v4.md](docs/models/gpt-sovits-v4.md) — GPT-SoVITS v4 LoRA fine-tune (consolidated)
- [docs/12-integration.md](docs/12-integration.md) — wrapping any TTS as a Provider-Spec sidecar
- [docs/07-windows-guide.md](docs/07-windows-guide.md) — Windows-specific quirks (CUDA, build flags, audio I/O)

### Theory + comparison
- [docs/01-theory.md](docs/01-theory.md) — TTS theory (transfer learning, two-stage design, info bottleneck) — universal, framed via GPT-SoVITS
- [docs/02-comparison.md](docs/02-comparison.md) — cross-model decision tree ("when to use which") for 14+ models

### Performance + optimization
- [docs/13-inference-optimization.md](docs/13-inference-optimization.md) — kernel-level optimization (T1+T2+T3+T4-prealloc on Qwen3-TTS, 5.98× over baseline; tactics generalize to most AR TTS)
- [docs/deployment/multi-engine.md](docs/deployment/multi-engine.md) — multi-model sidecar router architecture

## Validated on

| Track | Hardware | OS | GPU mem | Model versions tested |
|---|---|---|---|---|
| Zero-shot (Qwen3-TTS) | RTX 5080 (Blackwell) | Windows 11 | 16 GB | Qwen3-TTS-12Hz-1.7B-Base, qwen-tts 0.1.x, torch 2.11+cu128 |
| Fine-tune (GPT-SoVITS) | RTX 5080 | Windows 11 | 16 GB | GPT-SoVITS v4 (LoRA), pretrained s2Gv4 |

## How to contribute a model page

If you've deployed a TTS model not yet covered:
1. Copy [docs/models/_template.md](docs/models/_template.md) (coming)
2. Fill in: license, quality benchmarks, voice-clone capability,
   phonemization, VRAM, RTF, Windows-deploy difficulty, known failure
   modes
3. Open a PR

## Project history

This repo started as `gpt-sovits-voice-cloning-guide` — a hands-on
tutorial for fine-tuning GPT-SoVITS v4. As the open-source TTS
landscape diversified (zero-shot via Qwen3-TTS, native-JA via
Style-Bert-VITS2, fast-EN via Kokoro, etc.), the original mono-model
framing stopped serving readers who needed to pick AMONG models. The
2026 reorg keeps every original chapter as a hands-on recipe but
reframes the entry point as a comparison-first resource.

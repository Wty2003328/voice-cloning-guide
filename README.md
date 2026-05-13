# GPT-SoVITS Voice Cloning Guide

An end-to-end, **English-language** tutorial for fine-tuning [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) on a single GPU — with the ML theory you need to understand *why* it works, not just click-through instructions. Covers both **v2** (default, robust) and **v4** (LoRA-based, 48 kHz output).

Cloned a character voice from ~16 minutes of audio on a single RTX 5080 (16 GB VRAM, Windows 11). Outputs Japanese, English, and Chinese speech from arbitrary text. The official repo is excellent but assumes you know what you're doing and that you read Chinese; this guide is for the rest of us.

## Two tracks

This guide serves two audiences:

- **Quick-start track** — [**quickstart.md**](quickstart.md): clone a voice and start a TTS server in ~45 minutes. No theory, just commands. Skip straight there if you want results fast.
- **Learning track** — the docs below teach the ML behind every step: architecture, loss functions, sampling parameters, and why the pipeline works. Start with [01 — Theory](docs/01-theory.md).

Both tracks share the same scripts and upstream repo.

## Who this is for

- **Developers who want to understand voice cloning**, not just run a WebUI button. The docs explain transfer learning, semantic-token quantization, and the two-stage GPT/SoVITS design philosophy.
- **Single-GPU users on Windows or Linux**. Standalone Python scripts replace the official DDP+Lightning training loop, which is fragile on Windows and unnecessary at this data scale.
- **People who want a clean script base to extend**. Each script is ~100-200 lines, documented, and decoupled from the WebUI.

## Setup

For first-time setup (cloning both repos, installing PyTorch + CUDA, downloading pretrained models), follow **[docs/00-setup.md](docs/00-setup.md)** — it walks through everything end-to-end and takes ~30-60 minutes.

In short, you'll end up with:

```
your_workspace/
├── GPT-SoVITS/                          # upstream repo (RVC-Boss/GPT-SoVITS)
│   └── GPT_SoVITS/pretrained_models/    # downloaded model weights
└── gpt-sovits-voice-cloning-guide/      # this repo
    └── scripts/                          # standalone training scripts
```

## Quick start

After setup completes:

```bash
# Optional: if your source has BGM/SFX, isolate vocals first
python scripts/demucs_isolate.py --input my_video_audio.wav --output my_speaker_vocals.wav

# Data pipeline (shared between v2 and v4 — each step writes outputs the next consumes)
cd scripts
python 01_slice_audio.py      --vocals ../my_speaker_vocals.wav --exp my_speaker
python 02_asr_transcribe.py   --exp my_speaker --lang ja
python 03_extract_features.py --exp my_speaker
python 04_extract_semantic.py --exp my_speaker
```

Then pick **v2** (default, robust on noisy data) or **v4** (LoRA, 48 kHz output, recommended on clean data). See [09-v4.md](docs/09-v4.md) for the comparison.

### v2 path — full fine-tune, 32 kHz

```bash
python 05_train_sovits.py --exp my_speaker --epochs 20                     # ~25 min on RTX 5080
python 06_train_gpt.py    --exp my_speaker --epochs 15                     # ~30 sec
python 07_inference.py --exp my_speaker --lang ja \
    --text "こんにちは、はじめまして！" \
    --ref-wav ../GPT-SoVITS/logs/my_speaker/0_sliced/0003.wav \
    --ref-text "ここは私に任せて私を選んでくれる" --ref-lang ja \
    --out hello_v2.wav
```

### v4 path — LoRA fine-tune, 48 kHz

```bash
python 05_train_sovits_v4.py --exp my_speaker --epochs 20 --lora-rank 32   # ~25 min on RTX 5080
python 06_train_gpt.py       --exp my_speaker --epochs 15 --pretrained-version v4
python 07_inference_v4.py --exp my_speaker --lang ja \
    --text "こんにちは、はじめまして！" \
    --ref-wav ../GPT-SoVITS/logs/my_speaker/0_sliced/0003.wav \
    --ref-text "ここは私に任せて私を選んでくれる" --ref-lang ja \
    --out hello_v4.wav
```

If you're not in `scripts/`, set `GS_DIR=/abs/path/to/GPT-SoVITS` so the scripts can locate the upstream repo.

## Documentation (learning track)

The docs are organized so you can read top-to-bottom for understanding, or jump to a specific section when you hit a problem. For the streamlined version, see [quickstart.md](quickstart.md).

| Doc | What it covers |
|---|---|
| [00 — Setup](docs/00-setup.md) | Full installation walkthrough: clone both repos, conda env, CUDA/PyTorch, model downloads |
| [01 — Theory](docs/01-theory.md) | Why fine-tuning works with so little data; transfer learning, few-shot voice cloning, the two-stage design |
| [02 — Comparison](docs/02-comparison.md) | GPT-SoVITS vs RVC, CosyVoice, XTTS, Bark, Fish Speech — when to use each |
| [03 — Architecture](docs/03-architecture.md) | GPT (Text2SemanticDecoder) and SoVITS (VITS-based) deep dive, with loss formulations |
| [04 — Data pipeline](docs/04-data-pipeline.md) | Slicing, ASR, phonemization, HuBERT/BERT/semantic feature extraction |
| [05 — Training](docs/05-training.md) | Hyperparameters, when to stop, reading the loss curves |
| [06 — Inference](docs/06-inference.md) | How reference audio works, sampling parameters, choosing a good ref clip |
| [07 — Windows guide](docs/07-windows-guide.md) | CUDA 13 issues, torchcodec failures, DDP bypass, the gotchas you'll hit |
| [08 — Extending](docs/08-extending.md) | More data, more epochs, multi-speaker, integrating with VTuber pipelines |
| [09 — v4](docs/09-v4.md) | The v4 path: LoRA fine-tuning, 48 kHz vocoder, when v4 beats v2 |

## Requirements

- **GPU**: NVIDIA with ≥6 GB VRAM (tested on RTX 5080, 16 GB). CPU works for inference but training needs CUDA.
- **OS**: Windows 11 (validated) or Linux. macOS unsupported by the underlying GPT-SoVITS.
- **Audio**: ≥4 minutes of clean speech for usable results, ≥10 minutes for production quality. See [docs/04-data-pipeline.md](docs/04-data-pipeline.md) for what counts as "clean."
- **Disk**: ~10 GB total — pretrained models ~5 GB, Python deps ~3 GB, training artifacts 1+ GB.

Detailed install instructions in [docs/00-setup.md](docs/00-setup.md).

## Repo layout

```
gpt-sovits-voice-cloning-guide/
├── README.md                # this file
├── LICENSE                  # MIT
├── docs/                    # tutorial documentation
├── scripts/                  # standalone training & inference scripts
│   ├── _common.py            #   shared setup helpers
│   ├── 01_slice_audio.py     #   data pipeline (shared between v2 and v4)
│   ├── 02_asr_transcribe.py
│   ├── 03_extract_features.py
│   ├── 04_extract_semantic.py
│   ├── 05_train_sovits.py    #   v2 SoVITS — full fine-tune
│   ├── 05_train_sovits_v4.py #   v4 SoVITS — LoRA fine-tune
│   ├── 06_train_gpt.py       #   GPT — works for v2 or v4 via --pretrained-version
│   ├── 07_inference.py       #   v2 inference — 32 kHz output
│   ├── 07_inference_v4.py    #   v4 inference — 48 kHz output via vocoder
│   ├── demucs_isolate.py     #   optional vocal isolation
│   └── requirements.txt
├── configs/
│   └── example_s2.json      # SoVITS v2 hyperparameter reference
└── examples/                # sample outputs (added as the project matures)
```

## Acknowledgements

- [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) — the underlying model and codebase. Read their docs for the WebUI workflow.
- [CorentinJ/Real-Time-Voice-Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning) — early pioneer of few-shot voice cloning.
- [VITS](https://arxiv.org/abs/2106.06103) and [HuBERT](https://arxiv.org/abs/2106.07447) — the architectural primitives.

## License

MIT — see [LICENSE](LICENSE).

# 00 — Setup: Full Workspace Installation

A complete walkthrough for setting up the GPT-SoVITS workspace from scratch. After this, you'll be ready to follow the rest of the guide.

This page is intentionally explicit — copy each command in order, verify each step. If you've never set up a CUDA + PyTorch environment before, plan ~30-60 min for the full install (mostly model downloads).

## Prerequisites

| Component | Required version | Notes |
|---|---|---|
| OS | Windows 10/11 or Linux | macOS not supported by upstream GPT-SoVITS |
| Python | 3.10 | Newer versions may work but 3.10 is what upstream tests against |
| GPU | NVIDIA, ≥6 GB VRAM | Tested up to RTX 5080 (16 GB) |
| CUDA | 12.6 or 12.8 (Linux), 12.6 or 12.8 (Windows) | Newer (CUDA 13) works with the right PyTorch index |
| Disk | ~10 GB free | Models: ~5 GB; Python deps: ~3 GB; training artifacts: 1+ GB |
| conda or venv | Either works | Examples below use conda |

## Workspace layout

We'll create this structure:

```
your_workspace/
├── GPT-SoVITS/                          # upstream repo (heavy: code + models)
│   └── GPT_SoVITS/pretrained_models/    # all the model weights
└── gpt-sovits-voice-cloning-guide/      # this guide (lightweight)
    └── scripts/                          # standalone training scripts
```

The two repos sit side-by-side. The guide's scripts assume `GPT-SoVITS/` lives at `../GPT-SoVITS` relative to where you run them (override with `--gs-dir` or the `GS_DIR` env var if you put it elsewhere).

## Step 1: Clone both repositories

```bash
# Pick a workspace dir without spaces or unicode in the path
mkdir voice-cloning && cd voice-cloning

# Upstream — provides the model code and pretrained weights infrastructure
git clone https://github.com/RVC-Boss/GPT-SoVITS

# This guide — standalone scripts + tutorial docs
git clone https://github.com/Wty2003328/gpt-sovits-voice-cloning-guide
```

## Step 2: Create the conda environment

```bash
conda create -n gpt-sovits python=3.10 -y
conda activate gpt-sovits
```

If you prefer venv:
```bash
python3.10 -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
```

## Step 3: Install GPT-SoVITS dependencies

The upstream provides install scripts that pick the right PyTorch + CUDA combo for you.

### Linux

```bash
cd GPT-SoVITS
bash install.sh --device CU126 --source HF
# Replace CU126 with CU128 if your CUDA driver supports 12.8
# Replace HF with HF-Mirror if you're in mainland China
pip install -r requirements.txt
conda install ffmpeg -y
sudo apt install libsox-dev   # for some audio ops
cd ..
```

### Windows (PowerShell)

```powershell
cd GPT-SoVITS
pwsh -F install.ps1 -Device CU126 -Source HF
pip install -r extra-req.txt --no-deps
pip install -r requirements.txt
cd ..
```

You'll also need `ffmpeg.exe` and `ffprobe.exe` on PATH. The simplest route on Windows:

```powershell
# Install via conda (recommended — keeps it scoped to this env)
conda install ffmpeg -y
```

If `conda install ffmpeg` fails or you prefer a system install, download the official Windows builds from [ffmpeg.org](https://ffmpeg.org/download.html) and add the `bin/` dir to PATH.

### Verify CUDA

```bash
python -c "import torch; print('cuda available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
```

If `cuda available: False`, you have a PyTorch / CUDA driver mismatch. Reinstall PyTorch from the right index URL:

```bash
# CUDA 12.6
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126

# CUDA 12.8
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# CUDA 13.0
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu130
```

## Step 4: Install this guide's dependencies

```bash
cd gpt-sovits-voice-cloning-guide
pip install -r scripts/requirements.txt
cd ..
```

This adds anything our standalone scripts need on top of GPT-SoVITS's requirements (faster-whisper, demucs, soundfile, etc.).

## Step 5: Download pretrained models

GPT-SoVITS doesn't bundle weights; you download them after install. The upstream models live on HuggingFace at [lj1995/GPT-SoVITS](https://huggingface.co/lj1995/GPT-SoVITS). For v2, you need:

### Required files

| File | Path | Purpose |
|---|---|---|
| `s2G2333k.pth` (~95 MB) | `GPT-SoVITS/GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/` | SoVITS generator |
| `s2D2333k.pth` (~95 MB) | same | SoVITS discriminator |
| `s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt` (~155 MB) | same | GPT (Text2SemanticDecoder) |
| `chinese-hubert-base/` | `GPT-SoVITS/GPT_SoVITS/pretrained_models/` | HuBERT encoder (full HF model dir) |
| `chinese-roberta-wwm-ext-large/` | same | BERT for Chinese text |

Easiest install: download the entire bundled `gsv-v2final-pretrained` folder + the two HuggingFace model directories.

```bash
# Inside the workspace root (parent of GPT-SoVITS/)
cd GPT-SoVITS/GPT_SoVITS/pretrained_models

# v2 generator + discriminator + GPT
huggingface-cli download lj1995/GPT-SoVITS --include "gsv-v2final-pretrained/*" --local-dir . --local-dir-use-symlinks False

# CN-HuBERT (audio encoder)
huggingface-cli download lj1995/GPT-SoVITS --include "chinese-hubert-base/*" --local-dir . --local-dir-use-symlinks False

# Chinese-RoBERTa (BERT for zh text)
huggingface-cli download lj1995/GPT-SoVITS --include "chinese-roberta-wwm-ext-large/*" --local-dir . --local-dir-use-symlinks False

cd ../../..  # back to workspace root
```

If you don't have `huggingface-cli`, `pip install huggingface_hub[cli]`.

### Optional: G2PWModel (only for Chinese phonemization)

If you'll work with Chinese text (training or inference), download the G2PW model:

```bash
cd GPT-SoVITS/GPT_SoVITS/text
# Download from HuggingFace
wget https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/G2PWModel.zip
unzip G2PWModel.zip
# Renames to G2PWModel/ — verify
ls G2PWModel
cd ../../..
```

If you only ever train and inference on Japanese / English, you can skip G2PW.

### Optional: faster-whisper for ASR

The pipeline auto-downloads `Systran/faster-whisper-large-v3` (~3 GB) on first ASR run, into `GPT-SoVITS/tools/asr/models/`. You can pre-download it:

```bash
huggingface-cli download Systran/faster-whisper-large-v3 \
    --local-dir GPT-SoVITS/tools/asr/models/faster-whisper-large-v3 \
    --local-dir-use-symlinks False
```

If you're transcribing Japanese / English / Chinese and have a slow internet connection, pre-downloading saves time during training.

## Step 6: Verify the installation

A minimal smoke test — runs nothing real, just imports the modules and confirms paths:

```bash
cd gpt-sovits-voice-cloning-guide/scripts
python -c "
from _common import setup, pretrained_paths
gs = setup()
p = pretrained_paths(gs)
import os
for k, v in p.items():
    ok = os.path.exists(v)
    print(f'{k:10s}  {(\"OK \" if ok else \"MISS\")}  {v}')
"
```

You should see `OK` next to all six entries (`s2g`, `s2d`, `s1`, `cnhubert`, `bert`, `s2_config`). Any `MISS` means a model file is in the wrong place.

## Step 7: (Optional) Set up vocal isolation

If your training audio is mixed (anime / game / streamer source with BGM), install Demucs for vocal isolation:

```bash
pip install demucs
```

Demucs auto-downloads its model on first run.

**Windows note**: Demucs ships with `torchcodec` for audio I/O, which often fails on Windows. The [`demucs_isolate.py`](../scripts/demucs_isolate.py) script in this repo bypasses torchcodec by feeding pre-loaded tensors. If you see `OSError: Could not load this library: ...torchcodec...`, use our script instead of `python -m demucs`.

## You're done!

Move on to the [main quickstart](../README.md#quick-start) or any of the docs:

- [01 — Theory](01-theory.md): why this works
- [04 — Data pipeline](04-data-pipeline.md): preparing your audio
- [05 — Training](05-training.md): the actual fine-tuning step

If you got stuck, [07 — Windows guide](07-windows-guide.md) has fixes for the common Windows-specific failure modes.

## Troubleshooting cheatsheet

| Error | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'GPT_SoVITS'` | Wrong working directory; the scripts `chdir` to `GS_DIR`. Run them from this guide's `scripts/` dir, or set `GS_DIR=/abs/path/to/GPT-SoVITS`. |
| `torch.cuda.is_available() == False` | PyTorch installed without CUDA support. Reinstall from the right `--index-url`. |
| `OSError: ...libtorchcodec_core6.dll` | Demucs's torchcodec dep failing on Windows. Use [`demucs_isolate.py`](../scripts/demucs_isolate.py) instead. |
| `OSError: model.safetensors not found` from HuggingFace | Cache path has unicode/spaces. Set `HF_HOME=C:\hf_cache`. |
| Git warnings about `LF will be replaced by CRLF` | Harmless Git autocrlf message on Windows. Add `* text=auto eol=lf` to `.gitattributes` if it bothers you. |
| `CUDA out of memory` during SoVITS training | Drop `--batch-size 2` and retry. See [05 — Training](05-training.md). |

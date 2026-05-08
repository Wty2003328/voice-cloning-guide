# 07 — Windows-Specific Guide

Most ML tutorials assume Linux. This one was developed on Windows 11 + RTX 5080 + CUDA 13, and the entire training pipeline works there — but you have to know the gotchas. This document collects every Windows-specific issue we hit, with fixes.

## Why Windows is harder than Linux for this

- **NCCL doesn't exist on Windows**. PyTorch's standard distributed training (`DistributedDataParallel`) requires NCCL → won't work. The original GPT-SoVITS trainers assume DDP, so we can't use them as-is. **This repo's standalone scripts bypass DDP entirely**, which is the main reason they exist.

- **CTranslate2 (faster-whisper backend) often mismatches CUDA version**. CTranslate2 ships with bundled CUDA libraries that may not match your driver. We force CPU mode (`compute_type=int8`) by default — slightly slower but reliable.

- **torchcodec depends on FFmpeg libraries that may not be installed**. Demucs uses torchcodec internally for audio I/O, which fails on most Windows setups. Our `demucs_isolate.py` bypasses it by feeding pre-loaded tensors directly to `apply_model`.

- **torchaudio's audio I/O backend selection is unreliable**. We avoid `torchaudio.load` entirely, using `soundfile` and `librosa` instead.

- **Path separators**. Mixing forward and backward slashes in scripts can break on edge cases. We use `pathlib.Path` everywhere.

- **`os.fork()` doesn't exist**. Some libraries assume it. The standalone scripts avoid multiprocessing data loaders (`num_workers=0`).

## CUDA setup

Tested combinations:

| OS | GPU | CUDA | PyTorch | Status |
|---|---|---|---|---|
| Windows 11 | RTX 5080 (16 GB) | 13.0 | 2.10.0+cu130 | ✅ Works (this guide's reference) |
| Windows 11 | RTX 4090 | 12.1 | 2.4.0+cu121 | ✅ Works |
| Windows 11 | RTX 3060 (12 GB) | 12.1 | 2.4.0+cu121 | ✅ Works (slower training) |

For a fresh setup:
1. Install CUDA from NVIDIA. Match your GPU's compute capability.
2. Install PyTorch with the matching CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu121` (or cu130).
3. Verify: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())"`.

If `torch.cuda.is_available()` is `False`, no amount of pip-installing fixes it — you have a CUDA driver / PyTorch version mismatch. Reinstall PyTorch with the right `--index-url`.

## ffmpeg on PATH

The slicer and ASR scripts call ffmpeg via subprocess. Conda environments typically include ffmpeg, but the binary lives at:

```
<env>/Scripts/ffmpeg.exe   (Windows)
<env>/bin/ffmpeg           (Linux/macOS)
```

Our scripts add `<env>/Scripts` to `PATH` automatically. If you're running outside conda, install ffmpeg system-wide and confirm `where ffmpeg` resolves.

## CTranslate2 / faster-whisper

The `WhisperModel(device="cuda", compute_type="float16")` path breaks on most Windows setups with errors like:
```
CUDA failed with error CUDA driver version is insufficient for CUDA runtime version
```

Workaround: keep `device="cpu", compute_type="int8"` (the default in [02_asr_transcribe.py](../scripts/02_asr_transcribe.py)). For 200 slices on a modern CPU expect 15-20 minutes. This is acceptable as a one-time data-prep cost.

If you really need GPU ASR, install CTranslate2 from source against your specific CUDA. Most users shouldn't bother.

## Demucs / torchcodec

Running `python -m demucs ...` on Windows often errors at startup with:
```
OSError: Could not load this library: ...torchcodec\libtorchcodec_core6.dll
FileNotFoundError: Could not find module 'libtorchcodec_core4.dll' (or one of its dependencies)
```

This is torchcodec failing to find FFmpeg shared libraries. Two fixes:

**Recommended**: use [`demucs_isolate.py`](../scripts/demucs_isolate.py). It loads audio with soundfile (no torchcodec), constructs a tensor manually, and calls `demucs.apply.apply_model` directly. We've verified this works on Windows 11.

**Alternative**: install FFmpeg shared libraries system-wide via vcpkg or chocolatey, then reinstall torchcodec. More setup, but lets you use the official Demucs CLI.

## DDP-related code paths in upstream GPT-SoVITS

The official `s2_train.py` and `s1_train.py` scripts use `torch.distributed`, `DistributedBucketSampler`, and Lightning's `Trainer` with multi-GPU defaults. Even on a single GPU these:

- Initialize a process group (silly but harmless).
- Use the bucket sampler in a way that requires `set_epoch` calls.
- Wrap the model in DDP, which requires NCCL → fails on Windows.

Our [`05_train_sovits.py`](../scripts/05_train_sovits.py) and [`06_train_gpt.py`](../scripts/06_train_gpt.py) keep the bucket sampler (because it batches by length, which matters for memory) but skip DDP and Lightning. Look at the imports vs the upstream `s2_train.py` to see the exact differences.

## File path quirks

Long paths with spaces or unicode characters can break some HuggingFace cache logic. Symptom: `OSError: model.safetensors not found` even though the file exists. Workaround: clone GPT-SoVITS to a path without spaces (e.g., `C:\dev\GPT-SoVITS`, not `C:\Users\My Name\Desktop\projects\GPT-SoVITS`).

If your username has unicode (e.g., a Chinese or Japanese name), set `HF_HOME=C:\hf_cache` or similar to redirect the HuggingFace cache outside your home directory.

## Memory tuning

RTX 4060 / 4070 (8-12 GB) users may OOM during SoVITS training with the default batch size 4. Workarounds:

- Drop `--batch-size` to 2.
- Edit `s2.json` to lower `segment_size` from 20480 to 10240 (halves activation memory in the discriminator).
- Skip discriminator entirely by removing the `loss_disc + ` term — produces lower-quality output but trains in 4-6 GB.

Recovery from OOM mid-training: just re-run the trainer; it will load the latest saved checkpoint and continue.

## Numbered things to check when something breaks

In rough order of likelihood:

1. **Did you set the conda/venv environment correctly?** Most "module not found" errors are wrong-environment errors.
2. **Does `torch.cuda.is_available()` return True?** If not, reinstall PyTorch with the right CUDA index URL.
3. **Are pretrained models in the expected paths?** `GPT-SoVITS/GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth` and friends. If missing, follow the upstream README to download them.
4. **Is `ffmpeg` on PATH?** `where ffmpeg` should print a path.
5. **Did the data pipeline produce the expected files?** Check `2-name2text.txt`, `4-cnhubert/`, `5-wav32k/`, `6-name2semantic.tsv` all exist with comparable counts.
6. **Are you running from the repo root?** The scripts `chdir` to the GPT-SoVITS directory and assume specific relative paths.

If something still doesn't work, open a GitHub issue with the error message and the output of:
```bash
python -c "import torch, sys; print(sys.version, torch.__version__, torch.cuda.is_available())"
```

## Linux note

Everything works on Linux too, usually with fewer issues. NCCL is available, so if you really want DDP you can use the upstream trainers. But the standalone scripts in this repo work fine on Linux as a single-GPU path, and they're easier to debug.

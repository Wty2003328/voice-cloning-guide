# Quick Start: Clone a Voice and Deploy to zeroclaw-companion

Train a GPT-SoVITS v4 voice model from ~10 minutes of audio, then start a TTS server that speaks in the cloned voice. Total time: ~45 minutes (setup 15 min, training 30 min).

**What you'll end up with**: a running TTS server at `http://127.0.0.1:9880` that generates speech in your character's voice on every request — Japanese, English, or Chinese. Ready to plug into [zeroclaw-companion](https://github.com/Wty2003328/zeroclaw-companion) or any client that speaks `POST /tts`.

Want the theory behind every step? Each section links to the corresponding deep-dive in the [learning track](docs/).

> **Note on paths**: the training scripts change directory (`os.chdir`) into the GPT-SoVITS repo at startup. All file paths passed to the numbered scripts (`01_slice_audio.py`, `07_inference_v4.py`, etc.) are resolved **relative to the GPT-SoVITS root**, not the directory you run the command from. The examples below account for this.

---

## Prerequisites

| Requirement | Details |
|---|---|
| GPU | NVIDIA with ≥6 GB VRAM. Tested on RTX 5080 (16 GB). CPU inference works but training needs CUDA. |
| OS | Windows 11 or Linux. |
| Disk | ~10 GB free (pretrained models ~5 GB, deps ~3 GB, training artifacts 1+ GB). |
| Python env | Conda with Python 3.10+. |
| Audio | ≥4 minutes of clean speech (≥10 min recommended). Single speaker, no BGM, no reverb. |

---

## 1. Setup (15 min)

Create a workspace and clone the repos side by side:

```bash
mkdir voice-cloning && cd voice-cloning
git clone https://github.com/RVC-Boss/GPT-SoVITS.git
git clone https://github.com/Wty2003328/gpt-sovits-voice-cloning-guide.git
```

Create a conda environment and install dependencies:

```bash
conda create -n gpt-sovits python=3.10 -y
conda activate gpt-sovits

# PyTorch with CUDA — adjust the CUDA version for your driver.
# See https://pytorch.org/get-started/locally/ for the right command.
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# Guide scripts + faster-whisper + PEFT
cd gpt-sovits-voice-cloning-guide/scripts
pip install -r requirements.txt
```

Download pretrained model weights (~5 GB):

```bash
cd ../../GPT-SoVITS/GPT_SoVITS/pretrained_models

# HuBERT (audio encoder)
huggingface-cli download lj1995/GPT-SoVITS --include "chinese-hubert-base/*" --local-dir . --local-dir-use-symlinks False

# BERT (only used for Chinese text — small and useful to have)
huggingface-cli download lj1995/GPT-SoVITS --include "chinese-roberta-wwm-ext-large/*" --local-dir . --local-dir-use-symlinks False

# v4 SoVITS base + vocoder
huggingface-cli download lj1995/GPT-SoVITS --include "gsv-v4-pretrained/*" --local-dir . --local-dir-use-symlinks False

# GPT v3 base (shared by v4)
huggingface-cli download lj1995/GPT-SoVITS --include "s1v3.ckpt" --local-dir . --local-dir-use-symlinks False
```

If you don't have `huggingface-cli`, install it: `pip install huggingface_hub[cli]`.

**Trouble?** See [docs/00-setup.md](docs/00-setup.md) for the full walkthrough with troubleshooting for CUDA, ffmpeg, and Windows-specific issues.

Verify the layout:

```
voice-cloning/
├── GPT-SoVITS/                          # upstream repo
│   └── GPT_SoVITS/pretrained_models/    # downloaded weights
├── gpt-sovits-voice-cloning-guide/      # this guide
│   └── scripts/                         # training scripts
└── my_speaker_vocals.wav                # ← put your audio here (workspace root)
```

---

## 2. Prepare your audio

Place your source audio file in the workspace root (`voice-cloning/`). It should be a single `.wav` file of your target speaker's clean speech.

Quick checklist:
- **No background music or sound effects.** If your source has BGM, isolate vocals first (see below).
- **No reverb or echo.** Record in a quiet room or use a dry studio recording.
- **Single speaker only.** No interviews or multi-person dialogue.
- **At least 4 minutes.** 10+ minutes gives noticeably better quality.

### Optional: isolate vocals from BGM

`demucs_isolate.py` does NOT change directory, so paths are relative to your current working directory. From the workspace root:

```bash
# Adjust the cd to wherever your workspace is
cd voice-cloning/gpt-sovits-voice-cloning-guide/scripts
python demucs_isolate.py --input ../../my_video_audio.wav --output ../../my_speaker_vocals.wav
```

### Run the data pipeline

All numbered scripts (`01` through `07`) change directory into the GPT-SoVITS root at startup. All `--vocals`, `--ref-wav`, etc. paths are resolved from there. From the `scripts/` directory, that means one `../` reaches the workspace root.

```bash
cd voice-cloning/gpt-sovits-voice-cloning-guide/scripts

# So the scripts can find the upstream repo. PowerShell: $env:GS_DIR = "../../GPT-SoVITS"
export GS_DIR=../../GPT-SoVITS

# Step 1: Slice into 3-15s segments → GPT-SoVITS/logs/my_speaker/0_sliced/
python 01_slice_audio.py --vocals ../my_speaker_vocals.wav --exp my_speaker

# Step 2: Transcribe each slice with Whisper → GPT-SoVITS/logs/my_speaker/2-name2text.txt
python 02_asr_transcribe.py --exp my_speaker --lang ja

# Step 3: Extract HuBERT features → GPT-SoVITS/logs/my_speaker/4-cnhubert/
python 03_extract_features.py --exp my_speaker

# Step 4: Extract semantic tokens → GPT-SoVITS/logs/my_speaker/6-name2semantic.tsv
python 04_extract_semantic.py --exp my_speaker
```

Replace `--lang ja` with `en` or `zh` for English or Chinese source audio. On Windows PowerShell, use `$env:GS_DIR = "../../GPT-SoVITS"` instead of `export`.

**What to check**: after step 2, open `GPT-SoVITS/logs/my_speaker/asr.list` (inside the GPT-SoVITS repo, not `scripts/logs/`) and spot-check a few lines. Each line looks like:

```
0003.wav|speaker|ja|ここは私に任せて私を選んでくれる
```

The Japanese/English/Chinese text should match what's actually said in the audio. If it's wildly wrong, your audio quality may be too poor.

Deep dive: [docs/04-data-pipeline.md](docs/04-data-pipeline.md).

---

## 3. Train (30 min)

We use the v4 path (LoRA fine-tune, 48 kHz output). Two training commands, still from `scripts/`:

```bash
# SoVITS LoRA fine-tune (~25 min on RTX 5080)
python 05_train_sovits_v4.py --exp my_speaker --epochs 20 --lora-rank 32

# GPT fine-tune (~30 sec on RTX 5080)
python 06_train_gpt.py --exp my_speaker --epochs 15 --pretrained-version v4
```

**How to know it worked:**

SoVITS: watch `avg_cfm_loss` — it should drop from ~0.4 to below 0.1 by epoch 15-20. If it plateaus above 0.3, your data may be too noisy for v4; try the v2 path instead ([docs/09-v4.md](docs/09-v4.md#when-v4-underperforms-v2)).

GPT: watch `accuracy` — target 95-97%. Much lower = under-trained. Much higher = overfit.

**Checkpoints land in:**
- SoVITS: `GPT-SoVITS/SoVITS_weights_v4/my_speaker_e<N>_s<step>_l32.pth`
- GPT: `GPT-SoVITS/GPT_weights_v3/my_speaker-e<N>.ckpt`

Deep dive: [docs/05-training.md](docs/05-training.md), [docs/09-v4.md](docs/09-v4.md).

---

## 4. Verify the voice

Before wiring it into the companion, test that the model sounds right. Because the script changes directory into the GPT-SoVITS root, reference audio paths are relative to that root:

```bash
cd voice-cloning/gpt-sovits-voice-cloning-guide/scripts

python 07_inference_v4.py \
    --exp my_speaker \
    --text "こんにちは、はじめまして！" \
    --lang ja \
    --ref-wav logs/my_speaker/0_sliced/0003.wav \
    --ref-text "ここは私に任せて私を選んでくれる" \
    --ref-lang ja \
    --out logs/my_speaker/test_output.wav
```

> **Note**: `--ref-wav` and `--out` paths are relative to the GPT-SoVITS root (the script `chdir`s there at startup). `logs/my_speaker/0_sliced/0003.wav` refers to `GPT-SoVITS/logs/my_speaker/0_sliced/0003.wav`.

Play the output (at `GPT-SoVITS/logs/my_speaker/test_output.wav`). It should sound like your target speaker reading the `--text`. If it doesn't:
- Wrong speaker = wrong reference clip or SoVITS not loaded. Check the "Loading SoVITS" log line shows your fine-tuned `.pth`.
- Silence / truncation = GPT early-stopped. Try a different `--ref-wav` (pick from slices 0003-0010).
- Noisy output = training data wasn't clean enough. Re-run Demucs or switch to v2.

The `--ref-text` must exactly match what's said in `--ref-wav`. Pick a slice whose ASR transcript you trust — check `GPT-SoVITS/logs/my_speaker/asr.list` for the text transcript of each slice.

Deep dive: [docs/06-inference.md](docs/06-inference.md).

---

## 5. Start the TTS server

Now the trained model becomes a live TTS service. Clone the companion repo to get the server script:

```bash
cd voice-cloning       # or wherever your workspace is
git clone https://github.com/Wty2003328/zeroclaw-companion.git
```

The server script is at `zeroclaw-companion/tools/avatar/gptsovits_tts_server.py`. It auto-discovers your fine-tuned checkpoints — no modification needed.

### 5.1 Pick a reference clip

The server caches one reference clip at startup and uses it for every synthesis request. Requirements:

- **3-10 seconds** of clean speech from your target speaker.
- **Neutral emotion**, single sentence, clear delivery.
- **Transcript must be exact** — the server phonemizes this text to condition the model.

A good choice is one of your training slices (e.g., slice 0005). Get the transcript from `GPT-SoVITS/logs/my_speaker/asr.list` — the text is the last pipe-separated field on each line.

### 5.2 Start the server

Make sure the `gpt-sovits` conda env is active, then set the env vars and launch:

```bash
conda activate gpt-sovits

# --- Windows (PowerShell) ---
$env:TTS_MODEL_PATH     = "C:\Users\you\voice-cloning\GPT-SoVITS"
$env:TTS_REFERENCE_AUDIO = "C:\Users\you\voice-cloning\GPT-SoVITS\logs\my_speaker\0_sliced\0005.wav"
$env:TTS_REFERENCE_TEXT  = "exact transcript of that clip"
$env:TTS_REFERENCE_LANG  = "ja"
$env:TTS_VOICE           = "my_speaker"

python zeroclaw-companion/tools/avatar/gptsovits_tts_server.py

# --- Linux/macOS ---
export TTS_MODEL_PATH=/home/you/voice-cloning/GPT-SoVITS
export TTS_REFERENCE_AUDIO=/home/you/voice-cloning/GPT-SoVITS/logs/my_speaker/0_sliced/0005.wav
export TTS_REFERENCE_TEXT="exact transcript of that clip"
export TTS_REFERENCE_LANG=ja
export TTS_VOICE=my_speaker

python zeroclaw-companion/tools/avatar/gptsovits_tts_server.py
```

Wait for `serving on http://127.0.0.1:9880`. The server loads four models at startup (~5-15s): HuBERT, SoVITS v4 (DiT + LoRA-merged weights), 48kHz vocoder, and GPT. It then caches the reference clip's features.

### 5.3 Test the server

In a separate terminal:

```bash
# Health check — should return {"status": "ok", ...}
curl http://127.0.0.1:9880/health

# Synthesize a test sentence
curl -X POST http://127.0.0.1:9880/tts \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"こんにちは！\", \"language\": \"ja\"}" \
  -o test.wav
```

Play `test.wav` — it should be your cloned voice saying "こんにちは！".

### 5.4 Environment variables (reference)

| Variable | Purpose |
|---|---|
| `TTS_MODEL_PATH` | GPT-SoVITS repo root (**required**) |
| `TTS_REFERENCE_AUDIO` | Path to a 3-10s reference clip (**required**) |
| `TTS_REFERENCE_TEXT` | Exact transcript of the reference clip (**required**) |
| `TTS_REFERENCE_LANG` | Language of the reference clip (default `ja`) |
| `TTS_VOICE` | Checkpoint name prefix — matches your `--exp` name |
| `TTS_PORT` | Server bind port (default `9880`) |
| `TTS_LORA_NAME` | Override checkpoint prefix (defaults to `TTS_VOICE`) |
| `CUDA_VISIBLE_DEVICES` | GPU index; set to `-1` for CPU |

### 5.5 Wiring into zeroclaw-companion

To have the companion manage the TTS server lifecycle (auto-start, health check, graceful shutdown), add this to your `companion.toml`:

```toml
[avatar]
enabled = true

[avatar.tts]
engine             = "gpt-sovits-v4"
port               = 9880
language           = "ja"
voice              = "my_speaker"
speed              = 1.0
auto_start         = true
launch_command     = "python tools/avatar/gptsovits_tts_server.py"
model_path         = "C:/Users/you/voice-cloning/GPT-SoVITS"
reference_audio    = "C:/Users/you/voice-cloning/GPT-SoVITS/logs/my_speaker/0_sliced/0005.wav"
reference_text     = "exact transcript of the reference clip"
reference_language = "ja"
gpu_device         = 0
```

The companion forwards the TOML fields as env vars and spawns the server as a subprocess. See the [companion README](https://github.com/Wty2003328/zeroclaw-companion) for build and agent setup instructions.

> **Tip**: `launch_command` uses bare `python`. If your conda env isn't on PATH when the companion starts, use the full path: `"python tools/avatar/gptsovits_tts_server.py"`.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `GPT-SoVITS not found at ...` | GS_DIR not set | Set `GS_DIR` to your GPT-SoVITS path. PowerShell: `$env:GS_DIR = "C:\Users\you\voice-cloning\GPT-SoVITS"` |
| Server exits with "No SoVITS LoRA checkpoints" | `TTS_VOICE` doesn't match `--exp` name, or training didn't finish | Check `GPT-SoVITS/SoVITS_weights_v4/` for `.pth` files. |
| Server exits with "TTS_MODEL_PATH not set" | Missing env var | Set `TTS_MODEL_PATH` to the GPT-SoVITS repo root (absolute path). |
| Server exits with "TTS_REFERENCE_AUDIO" | Missing reference clip | Set `TTS_REFERENCE_AUDIO` and `TTS_REFERENCE_TEXT`. |
| `WinError 2: cannot find the file specified` | ffmpeg not found | `conda install ffmpeg`, or add ffmpeg to PATH. |
| Audio is silence / very short | GPT early-stopped (stochastic) | Server has built-in re-roll. If persistent, try a different reference clip. |
| Audio sounds like a different speaker | Loaded pretrained weights instead of fine-tuned | Check logs for "SoVITS ckpt:" — should show your `.pth`, not `s2Gv4.pth`. |
| CUDA out of memory | Too many models for VRAM | Set `CUDA_VISIBLE_DEVICES=-1` for CPU (slower). |
| Port 9880 already in use | Another server running | Kill it, or set `TTS_PORT` to a different port. |

---

## Next steps

- **Improve quality**: add more diverse training data (emotional range, sentence variety). See [docs/08-extending.md](docs/08-extending.md).
- **Switch to v2**: if your training data has background noise, v2 is more forgiving. Follow the v2 path in the [README](README.md).
- **Understand the architecture**: read the [learning track](docs/) from start to finish.
- **Fine-tune further**: adjust training epochs, LoRA rank, or sampling parameters. See [docs/05-training.md](docs/05-training.md) and [docs/09-v4.md](docs/09-v4.md).

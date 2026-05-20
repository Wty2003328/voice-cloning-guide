# GPT-SoVITS v4 (LoRA fine-tune)

| Field | Value |
|---|---|
| **Family** | Two-network design: SoVITS (VITS-based mel synthesizer) + GPT (autoregressive semantic-token predictor) |
| **License (code)** | MIT |
| **License (weights)** | MIT |
| **Best languages** | Japanese, Chinese, English, Korean, Cantonese (yue) |
| **Voice cloning** | **Per-voice fine-tune** (1-30 min training audio; ~30-60 min training on RTX 5080) |
| **Phonemization** | External: per-language (cnHuBERT for ZH semantic, pyopenjtalk for JA, g2p_en for EN) |
| **Params** | Pretrained ~150M + per-voice LoRA (rank 32, ~5 MB per voice) |
| **VRAM (training)** | ~12 GB |
| **VRAM (inference)** | ~3 GB |
| **RTF (RTX 5080)** | ~0.5 (faster with batching) |
| **Output sample rate** | 48 kHz (v4) — vs v2's 32 kHz |
| **Best-in-class for** | Per-character voice quality (anime, VTuber, OC) where you have ≥10 min of training data |
| **Status in this repo** | ✅ Validated end-to-end on Windows 11 + RTX 5080 |

## When to pick GPT-SoVITS v4

- You're building a **custom character voice** (anime, VTuber, OC,
  game character) that needs to learn the speaker's specific prosody
  patterns — not just timbre
- You have **10-30 minutes** of clean training audio for that voice
- You can spend ~30-60 min on a one-time training run per voice (then
  unlimited inference)
- You're on Windows or Linux with an NVIDIA GPU (≥6 GB VRAM for
  inference; ≥12 GB for training)
- MIT license commercial use is required

## When NOT to pick it

- You only have a 3-30s reference clip → use Qwen3-TTS or Higgs Audio
  v2.5 (zero-shot, no training)
- You only need Japanese at SOTA prosody quality → Style-Bert-VITS2
  has better JA pitch-accent handling
- You need real-time voice conversion of live input → RVC v2
- Your target language isn't in zh/ja/en/ko/yue

## Architecture in brief

GPT-SoVITS is two networks plus peripheral models, wired through a
discrete intermediate representation (semantic tokens):

```
                       ┌── BERT (zh only) ───┐
                       ▼                     │
text (zh/ja/en/ko/yue) ─┬─ phonemizer ──► phoneme IDs ─┐
                       │                              │
                       └────────────────────────────► GPT
                                                       │
                              ┌──────── semantic tokens ┘
                              │
ref audio (3-15s) ─► cnHuBERT ─► reference semantic tokens
                              │            │
                              ▼            ▼
                  SoVITS (VITS-based decoder) ──► 48 kHz waveform
```

**Two-stage rationale.** The GPT predicts the **what** — discrete
semantic tokens that capture phoneme + prosody + duration. The SoVITS
synthesizes the **how** — turning semantic tokens + speaker
conditioning into actual mel-spectrogram → waveform. Separating these
makes few-shot voice cloning tractable: SoVITS handles speaker
identity (mostly fixed during fine-tune), GPT handles speaker prosody
(updated during fine-tune to learn the speaker's rhythm patterns).

**LoRA fine-tune.** Instead of updating all weights, v4 trains a
low-rank adapter (rank 32 default) on top of the pretrained model. The
adapter weights are tiny (~5 MB) and the pretrained weights stay
frozen, so:
- Training is fast (~30-60 min on RTX 5080 for ~10 min of data)
- The pretrained model's broad voice knowledge isn't forgotten
- You can ship many character voices as small LoRAs

## End-to-end recipe

Short form below. The training scripts in [`scripts/`](../../scripts/)
are the canonical implementation — Windows-friendly (bypass GPT-SoVITS's
default DDP-based trainer which doesn't work on Windows).

### 1. Workspace setup (one-time, ~30-60 min)

Prereqs: Windows 11 (or Linux) + NVIDIA GPU (≥12 GB VRAM for training,
≥6 GB for inference) + CUDA 12.x + Python 3.10. Essential steps:

```bash
# Clone + install
git clone https://github.com/RVC-Boss/GPT-SoVITS.git
cd GPT-SoVITS
pip install -r requirements.txt
pip install -e .

# Download pretrained weights (~4 GB)
python download_pretrained.py
```

### 2. Optional: vocal isolation (if source has BGM/SFX)

```bash
python scripts/demucs_isolate.py --input video_audio.wav --output speaker_vocals.wav
```

### 3. Data pipeline (raw audio → training inputs)

```bash
cd scripts
python 01_slice_audio.py      --vocals ../speaker_vocals.wav --exp my_speaker
python 02_asr_transcribe.py   --exp my_speaker --lang ja
python 03_extract_features.py --exp my_speaker
python 04_extract_semantic.py --exp my_speaker
```

Each step writes outputs the next consumes. The pipeline lives in
`scripts/` — these are standalone scripts that bypass GPT-SoVITS's
default DDP-based trainer (DDP doesn't work on Windows).

### 4. Train

```bash
python 05_train_sovits_v4.py --exp my_speaker --epochs 20 --lora-rank 32
python 06_train_gpt.py       --exp my_speaker --epochs 15 --pretrained-version v4
```

SoVITS first (slower, ~30 min for 10 min data + 20 epochs), GPT second
(fast, ~10 min). Watch the loss curves — if you see catastrophic
spikes, reduce learning rate.

### 5. Inference

```bash
python 07_inference_v4.py --exp my_speaker --lang ja \
    --text "こんにちは、はじめまして！" \
    --ref-wav ../GPT-SoVITS/logs/my_speaker/0_sliced/0003.wav \
    --ref-text "ここは私に任せて私を選んでくれる" --ref-lang ja \
    --out hello.wav
```

Three things go in: text + reference audio + reference transcript.
Reference clip should be 3-15s of the target speaker — ideally from
the training set so its semantic tokens match what GPT learned.

## v4 vs v2 (why this guide standardized on v4)

| Aspect | v2 | v4 |
|---|---|---|
| Sample rate | 32 kHz | **48 kHz** (clear audio quality difference) |
| Vocoder | Built into SoVITS decoder | Separate pass (allows higher fidelity) |
| Training method | Full fine-tune (~50 MB per voice) | **LoRA (~5 MB per voice)** |
| Training speed | Baseline | Faster (smaller param surface) |
| Inference RTF | Similar | Similar |
| Voice fidelity | Good | **Better** (subjective A/B confirms) |
| Setup complexity | Lower | Slightly higher (LoRA adapter loading) |

Use v4 unless your training data is noisy (v2 is more forgiving). Use
v2 if you specifically need 32 kHz output for legacy pipeline
compatibility.

## Pros

- **MIT license** — clean commercial use, no restrictions
- **Best per-voice quality** for fine-tune models in this guide
- **Active community** (RVC-Boss + downstream forks)
- **Multi-language support** (zh/ja/en/ko/yue)
- **Small per-voice adapter** (LoRA ~5 MB) — ship N character voices
  cheaply
- **Windows-native** with this repo's standalone scripts (default
  trainer's DDP requirement bypassed)

## Cons

- **Requires per-voice training** (~30-60 min RTX 5080 per character)
- **Training data prerequisite** — needs 10-30 min of clean audio
- **JA pitch-accent imperfect** compared to native-JA Style-Bert-VITS2
  (multilingual model trained on JA+ZH+EN, doesn't optimize JA prosody
  specifically)
- **Inference RTF ~0.5** — slower than Style-Bert-VITS2 or Kokoro,
  comparable to other AR models
- **Pipeline complexity** — 7-step training workflow (slice → ASR →
  features → semantic → SoVITS-train → GPT-train → inference). Each
  step's outputs feed the next; debugging requires understanding the
  full chain.

## Extending the baseline (after you have working voice)

Diminishing returns around 30 min of clean speech. Beyond that, focus
on **diversity** not volume:

- **Emotional range**: angry, calm, excited, sad, whispered. The model
  can only generate styles it's seen during training.
- **Sentence length variety**: short interjections + long narrative
  passages.
- **Prosodic variety**: declarative + questioning + exclamative +
  hesitant.

For multi-emotion voices, train one LoRA per emotion and switch at
inference time. Or pool data across emotions if the styles aren't
distinct enough to warrant separation.

## Deployment difficulty

**3/5.** The training pipeline is well-documented but multi-step. The
inference path is simple (one Python script). Sidecar wrapper around
inference follows the same TTS-Provider-Spec pattern as the other
models (see [../12-integration.md](../12-integration.md)).

## Production integration

For a single-character voice, GPT-SoVITS makes sense as a long-running
sidecar:

```
companion-server
       ↓
tts-sidecar:9890 (GPT-SoVITS v4 + a per-character LoRA loaded)
```

For multi-character or multi-language deployments, GPT-SoVITS sits
alongside other engines in the multi-engine router:

```
tts-router:9890
       ├─→ tts-target:9891  (GPT-SoVITS v4 + a per-character LoRA, JA)
       ├─→ tts-cosyvoice:9892  (CosyVoice 3, ZH)
       └─→ tts-higgs:9893     (Higgs Audio v2.5, EN)
```

See [../deployment/multi-engine.md](../deployment/multi-engine.md).

## Known failure modes

| Symptom | Cause | Fix |
|---|---|---|
| AR truncation ("only said 4 words") | GPT predicts EOS too early on certain inputs (stochastic; worse with digits, "～" tilde, multi-sentence) | Per-sentence split + re-roll in wrapper. See `tts_lab` memory `project_tts_ar_truncation.md` |
| Multi-second silence followed by speech | SoVITS spinning up — first call after model load is slow | Warmup synth at boot |
| JA prosody flat / unexpressive | Multilingual training compromises JA specifically | Use Style-Bert-VITS2 if JA is the only language you need |
| Reference clip drifts voice | Reference semantic tokens don't match LoRA's learned prosody | Use a reference clip from the training set, not an arbitrary new clip |
| OOM at training | LoRA rank too high or batch too large | Reduce `--lora-rank` to 16 or `--batch-size` |

## Validated configurations

| | Hardware | OS | Python | Torch | Result |
|---|---|---|---|---|---|
| Training | RTX 5080 (Blackwell, 16 GB) | Win 11 | 3.10 | 2.11+cu128 | ~30-50 min per voice (10 min data, 20 epochs SoVITS, 15 GPT) |
| Inference | RTX 5080 | Win 11 | 3.10 | 2.11+cu128 | RTF ~0.5 |

## See also

- [`scripts/`](../../scripts/) — the standalone Windows-friendly training + inference scripts (canonical implementation; this page is the conceptual deep-dive that complements them)
- [`../01-theory.md`](../01-theory.md) — *why* fine-tuning works (transfer learning, two-stage design, info bottleneck — universal TTS theory framed via GPT-SoVITS)
- [`../07-windows-guide.md`](../07-windows-guide.md) — Windows-specific quirks (NCCL, torchcodec, audio I/O backends)
- [`../12-integration.md`](../12-integration.md) — wrapping GPT-SoVITS as a TTS Provider Spec sidecar
- [Upstream repo](https://github.com/RVC-Boss/GPT-SoVITS)

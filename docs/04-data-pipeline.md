# 04 — Data Pipeline

How raw audio becomes training inputs. The pipeline runs once before training and produces a directory layout that the SoVITS and GPT trainers consume directly.

```
your_audio.wav  ──►  Demucs (optional, if mixed source)
                         │
                         ▼
                vocals_clean.wav
                         │
                         ▼
        ┌────────────────┴────────────────┐
        │  01_slice_audio.py              │  WebRTC VAD-based slicing
        │  → logs/<exp>/0_sliced/*.wav    │  3-15s segments at 32 kHz
        └────────────────┬────────────────┘
                         ▼
        ┌────────────────┴────────────────┐
        │  02_asr_transcribe.py           │  faster-whisper transcription
        │  → asr.list (raw)               │  + phonemization via clean_text()
        │  → 2-name2text.txt (TAB)        │
        └────────────────┬────────────────┘
                         ▼
        ┌────────────────┴────────────────┐
        │  03_extract_features.py         │  CNHuBERT @ 16kHz → 768-dim
        │  → 4-cnhubert/*.pt              │  + zh-only BERT features
        │  → 3-bert/*.pt (zh)             │  + 32kHz normalized copies
        │  → 5-wav32k/*.wav               │
        └────────────────┬────────────────┘
                         ▼
        ┌────────────────┴────────────────┐
        │  04_extract_semantic.py         │  Quantize HuBERT → 1024-vocab
        │  → 6-name2semantic.tsv          │  semantic tokens @ 25 Hz
        └─────────────────────────────────┘
```

## Audio requirements

The model is forgiving about exact sample rate (the slicer resamples) but unforgiving about *content quality*. A 4-minute clip of clean vocals beats a 30-minute clip with background music.

What "clean" means concretely:

- **No background music** — even quiet BGM bleeds into the model and shows up as a constant hum in outputs.
- **No reverb / large rooms** — the model will replicate the room as part of the speaker's "voice."
- **Minimal sound effects, applause, or other speakers** — the slicer can't distinguish, and you'll get cross-speaker artifacts.
- **Consistent recording conditions** within the source. Switching between studio-quality and phone-quality clips confuses the speaker embedding.

If your source has any of these issues, run the optional **Demucs vocal isolation** step (`scripts/demucs_isolate.py`). Demucs separates vocals from music with high quality. It can't recover from severe reverb or fix multi-speaker scenes — those are upstream problems.

## Step 1: Slicing

The [Slicer](https://github.com/RVC-Boss/GPT-SoVITS/blob/main/tools/slicer2.py) detects silences below `-40 dB` and splits the file at those boundaries. Default constraints:

- Min slice length: 4000 ms
- Min silence interval that triggers a split: 300 ms
- Max silence kept around a slice boundary: 500 ms

The output is mono 32 kHz int16 PCM, normalized to a peak of 0.9 with mild dynamic-range compression. Files are named `0001.wav`, `0002.wav`, ... in `logs/<exp>/0_sliced/`.

You'll typically end up with one slice per ~5 seconds of source. From 16 minutes of audio expect ~150-200 slices.

**Tuning**: if your source has very short utterances (game NPC barks, dialogue snippets) the default `min-length-ms=4000` may discard them. Lower it to 2000 if you have many short clips. Conversely, for narration with long sentences, raise to 6000 to keep more context per slice.

## Step 2: ASR + phonemization

[faster-whisper](https://github.com/SYSTRAN/faster-whisper) (large-v3) transcribes each slice. The `--lang` flag tells it which language to expect — passing the wrong language produces garbage transcriptions.

We force CPU mode (`compute_type=int8`) by default because CTranslate2's CUDA build is sensitive to driver/CUDA version mismatches. On a fast CPU this transcribes ~5 seconds per slice; for 200 slices, plan ~15-20 min.

**Phonemization** runs on the ASR output, converting text to phoneme IDs via `GPT_SoVITS/text/cleaner.py`:
- Japanese: `pyopenjtalk` + G2P → mora-level phonemes.
- English: CMUDict-based g2p_en → ARPABET phonemes.
- Chinese: pinyin + tone marks via `g2pw`.

Output goes to `2-name2text.txt` (TAB-separated):
```
0001.wav    h e r o w a    speaker    ja
0002.wav    o n e g a y sh i m a s u    speaker    ja
```

Slices that fail phonemization (rare — usually empty transcriptions) are skipped silently. The log line "Saved N entries (skipped M)" tells you the actual training set size.

## Step 3: Feature extraction (HuBERT + BERT)

CNHuBERT consumes 16 kHz audio and outputs `[768, T_ssl]` per slice (T_ssl = audio length / 320 samples). Each tensor goes to `4-cnhubert/<wav>.pt`.

Side effect: a 32 kHz normalized copy of the audio is written to `5-wav32k/`. This is what the SoVITS data loader reads (it does its own spec computation). Without these the SoVITS dataset will silently skip slices.

**Why two sample rates?** HuBERT was pretrained on 16 kHz audio; running it at 32 kHz produces nonsense features. SoVITS operates at 32 kHz because that's the v2 vocoder rate.

**Precision**: `--no-fp16` falls back to fp32 if you see NaN warnings. With clean source data and a healthy GPU, fp16 is reliable.

For Chinese text, BERT features (1024-dim, last-3 hidden states from Chinese-RoBERTa-wwm-ext-large) are extracted and aligned to phoneme positions via the `word2ph` mapping. Output goes to `3-bert/<wav>.pt`.

For Japanese and English text, the BERT directory is created but no features are written. The inference scripts produce zero tensors of the correct shape on the fly. **Do not try to substitute a Japanese or English BERT** — the network architecture expects this specific Chinese model's dim and won't tolerate a different one.

## Step 4: Semantic token extraction

Loads the pretrained SoVITS generator (frozen — we use the published v2 weights, not your fine-tuned ones, since this step runs before training) and calls `extract_latent(ssl)` on each HuBERT tensor.

The result is a sequence of 1024-vocab integer codes at 25 Hz, written as space-separated tokens to `6-name2semantic.tsv`:

```
0001.wav    523 891 17 552 ...
0002.wav    412 1003 88 7 ...
```

This file is the **target** for GPT training. The GPT learns to produce these sequences from text.

## Final directory layout

```
GPT-SoVITS/logs/<exp>/
├── 0_sliced/         # Step 1: 32kHz mono PCM slices
│   ├── 0001.wav
│   └── ...
├── asr.list          # Step 2: raw ASR (pipe-separated)
├── 2-name2text.txt   # Step 2: phonemized (TAB-separated) — TRAINING INPUT
├── 3-bert/           # Step 3: zh BERT features (.pt)
├── 4-cnhubert/       # Step 3: HuBERT features (.pt)
├── 5-wav32k/         # Step 3: normalized 32kHz wavs (read by SoVITS loader)
└── 6-name2semantic.tsv  # Step 4: semantic tokens — GPT TRAINING TARGET
```

If any of these are missing or partial, the trainers will skip the affected slices silently. After running the full pipeline, do a sanity check:

```bash
wc -l logs/<exp>/2-name2text.txt logs/<exp>/6-name2semantic.tsv
ls logs/<exp>/4-cnhubert | wc -l
ls logs/<exp>/5-wav32k   | wc -l
```

These four counts should be approximately equal (within a couple, accounting for any ASR-empty or NaN-skipped slices).

## Common failures

- **`No slices in 0_sliced/`**: input vocals file is empty or all below -40 dB. Check audio playback.
- **ASR produces garbage** (lots of `<unk>` or wrong-language characters): wrong `--lang` flag, or audio quality is too poor for Whisper.
- **`Phonemize failed for X`**: the ASR output contained characters the phonemizer doesn't know (rare punctuation, emoji). Check the asr.list file.
- **Many slices in HuBERT skipped due to clipping (`tmp_max > 2.2`)**: your slice normalization upstream is wrong. Re-run Demucs or check the input gain.
- **NaN in HuBERT features**: usually a fp16 issue. Add `--no-fp16` to step 3.

The next document, [05 — Training](05-training.md), covers the actual fine-tuning step.

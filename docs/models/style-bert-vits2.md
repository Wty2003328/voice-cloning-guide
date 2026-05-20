# Style-Bert-VITS2 (JP-Extra)

| Field | Value |
|---|---|
| **Family** | VITS-based (VAE + flow + duration predictor) with BERT prosody conditioning |
| **License (code)** | AGPL-3.0 |
| **License (weights)** | JP-Extra base: trained on JSUT (Apache audio data); derivative models inherit |
| **Best language** | Japanese (JP-Extra variant) |
| **Voice cloning** | **Per-voice fine-tune** (~1-2 days on RTX 5080). Zero-shot also supported but quality much lower than fine-tune. |
| **Phonemization** | External: pyopenjtalk + Unidic dictionary (rule-based pitch accent) |
| **Params** | ~165M (smaller than Qwen3-TTS's 1.7B) |
| **VRAM** | ~2 GB (fits comfortably alongside other engines) |
| **RTF (RTX 5080)** | ~0.15-0.25 (very fast — small model + non-AR architecture) |
| **Best-in-class for** | Japanese chat / character voices where pitch accent matters |
| **Status in this repo** | 🚧 Prototype planned (per-character voice fine-tune) |

## When to pick Style-Bert-VITS2

- You need the BEST Japanese voice quality possible from open-source
  models
- You have 10-30 minutes of training audio for your target voice (or
  a ~150-200 clip pool of your target voice already in this workspace)
- You can spend 1-2 days training per voice (one-time per character)
- AGPL-3.0 commercial use is acceptable via IPC sidecar isolation
  (your closed-source product talks to the sidecar over HTTP/IPC; the
  sidecar itself stays AGPL-isolated)

## When NOT to pick it

- You need zero-shot voice cloning from 3-30s of reference audio
  (Qwen3-TTS or Higgs Audio v2.5 do this; SBV2's zero-shot mode
  exists but is far behind its fine-tuned mode)
- You only need English / Chinese (SBV2 is JP-focused)
- AGPL-3.0 license is a hard blocker for your distribution model

## Why it wins for Japanese

**Rule-based pitch accent.** This is the decisive factor. All
multilingual TTS models try to *learn* Japanese pitch accent from
training data — and they get it wrong on lexical-distinction pairs
(橋 hashi-bridge vs 箸 hashi-chopsticks) because the model conflates
the homophonous syllables.

SBV2 uses pyopenjtalk + Unidic dictionary to **mark the pitch contour
explicitly** before the neural network sees the text. The model
doesn't have to "figure out" 橋 vs 箸 — it gets the correct
high-low/low-high contour as input.

Empirical result: **MOS 4.37 on character-style JA**, essentially
matching the human reference at 4.38 (per the research report). Most
listeners can't reliably distinguish SBV2 output from a real recording
of the same speaker in blind A/B tests.

## Architecture in brief

```
text → pyopenjtalk → phonemes + pitch-accent labels + BERT embeddings
                 → VITS encoder → latent z + duration predictor
                 → flow + decoder → mel
                 → vocoder (HiFi-GAN) → waveform
```

Non-autoregressive end-to-end — no per-token decode loop, no AR
runaway risk. Each utterance generates in one forward pass.

## Quickstart (zero-shot mode)

For exploration, before committing to a fine-tune:

```bash
# Style-Bert-VITS2 setup
git clone https://github.com/litagin02/Style-Bert-VITS2.git
cd Style-Bert-VITS2
pip install -e .

# Download JP-Extra base model
python -m style_bert_vits2.download --model jp-extra
```

Zero-shot inference uses one of the pre-trained canonical voices. To
clone a custom voice (the target speaker), proceed to the fine-tune path.

## Fine-tune recipe for a custom voice

1. **Audio pool.** Need 10-30 min of clean audio in the target voice.
   Use a ~150 clip dataset under
   `GPT-SoVITS/logs/target_combined/0_sliced/` (already validated for
   GPT-SoVITS training; same data quality applies).

2. **Transcribe + normalize.** Use the existing
   `asr.list` (per-clip transcripts) in the same dir. SBV2 expects
   a per-clip JSON dataset format; convert from asr.list.

3. **Train.**
   ```bash
   python train_ms.py \
     --config configs/jp_extra/target.json \
     --model target-jp \
     --num_workers 4
   ```
   ~1-2 days on RTX 5080 for converged quality. Earlier checkpoints
   (epoch 30-50) are usable for testing.

4. **Inference test.**
   ```bash
   python infer.py --model target-jp \
     --text "今日はとてもいい天気ですね。" \
     --out target_out.wav
   ```

5. **Wrap as sidecar.** Implement the
   [TTS Provider Spec v1](../../../zeroclaw-companion/docs/TTS-PROVIDER-SPEC.md)
   `/v1/audio/speech` endpoint around the inference call. Sidecar
   runs as its own process (AGPL boundary respected).

## Pros

- **Best-in-class JA quality** with explicit pitch accent
- **Small + fast** (~2 GB VRAM, RTF ~0.15-0.25)
- **Non-autoregressive** — no runaway loops, no stochastic EOS
- **Active JA community** (litagin02 + Aivisspeech fork)
- **Per-voice fine-tune works on the same data** you'd use for
  GPT-SoVITS (a ~150-200 clip pool of your target voice is reusable)

## Cons

- **AGPL-3.0 code license** — your closed-source app can't statically
  link or share-process with SBV2; must communicate over IPC (HTTP,
  gRPC, etc.). The sidecar pattern handles this cleanly.
- **Per-voice training required** for best quality (1-2 days per voice)
- **JA-focused** — supports other languages via different base models
  but quality drops outside Japanese
- **Smaller speaker variety** than zero-shot models (you only get the
  voices you train)
- **Maintainer-dependent for base-model license clarity** —
  JP-Extra base trained on JSUT (Apache audio), should be commercial-OK
  for derivatives, but worth explicit clarification from
  litagin02 before shipping

## Deployment difficulty

**2/5 — easy install, AGPL-aware deployment required.**

```
pip install style-bert-vits2
python -m style_bert_vits2.download --model jp-extra
```

Runs on Windows + RTX 5080 native. Sidecar isolation for AGPL
compliance: ~50 LOC FastAPI wrapper around the inference function.

## Production integration

Per [multi-engine.md](../deployment/multi-engine.md), SBV2 runs as the
JA-only engine in a multi-sidecar router architecture:

```
companion-server
       ↓
tts-router:9890
       ├─→ tts-ja:9891   (Style-Bert-VITS2 fine-tuned on the target speaker)
       ├─→ tts-zh:9892   (CosyVoice 3)
       └─→ tts-en:9893   (Higgs Audio v2.5)
```

Router dispatches by `x_companion.language`. SBV2 always-resident
(~2 GB VRAM is cheap).

## See also

- [../per-language/japanese.md](../per-language/japanese.md) — why SBV2 over alternatives
- [../deployment/multi-engine.md](../deployment/multi-engine.md) — sidecar router architecture
- [Upstream repo](https://github.com/litagin02/Style-Bert-VITS2)
- [Aivisspeech fork](https://github.com/aivisspeech/aivisspeech) — JA-focused fork with web UI

# 03 — Architecture Deep Dive

GPT-SoVITS is two networks plus a few peripheral models, wired together by a discrete intermediate representation (semantic tokens). This document walks through each component with concrete shapes and the loss functions used in fine-tuning.

If you haven't read [01 — Theory](01-theory.md), do that first — it explains *why* the architecture is shaped this way.

## End-to-end picture

```
                                   ┌── BERT (zh only) ───────┐
                                   │                          │
Input text ──► Phonemizer ─► Phone IDs ──► GPT (s1) ──► Semantic tokens ──► SoVITS (s2) ──► Audio (32kHz)
                                                ▲                                ▲
                                          Reference                       Reference
                                          semantic prefix                 mel spectrogram
                                          (50 tokens)                     (full clip)
                                                ▲                                ▲
                                                └────── Reference audio ─────────┘
                                                        (3-15 sec clip)
```

Two nets are trained: **s1** (GPT) and **s2** (SoVITS). Everything else is frozen.

## Stage 1: GPT (Text2SemanticDecoder)

**Purpose**: Predict a sequence of semantic tokens given phoneme IDs and BERT features. This is a language model over a 1024-vocab discrete code.

**Architecture**:
- 24-layer Transformer decoder
- Hidden dim 512, 16 attention heads, FFN 2048
- Vocab size 1025 (1024 codes + 1 EOS)
- Phoneme vocab 732 (covers zh + ja + en + yue + ko phoneme inventories)

**Forward signature** (from `forward_old`):
```python
loss, accuracy = gpt.forward_old(
    phoneme_ids,        # [B, T_phone]   integer phoneme IDs
    phoneme_lens,       # [B]
    semantic_ids,       # [B, T_sem]     target semantic tokens
    semantic_lens,      # [B]
    bert_feature,       # [B, 1024, T_phone]  zeros for non-zh
)
```

**Training objective**: cross-entropy with sum reduction. Each predicted token is penalized for not matching the ground-truth semantic token at that position.

**Inference**: autoregressive decoding with top-k sampling. We feed in:
1. The reference audio's first 50 semantic tokens as a prompt prefix.
2. The text's phoneme IDs and BERT features.

Then the model generates additional tokens until it emits EOS or hits the max-length safety limit.

The GPT learns the *distribution* of speech timing, rhythm, and prosodic patterns for the target speaker. After fine-tuning, given Asuna-like text, it produces Asuna-like token sequences.

## Stage 2: SoVITS (SynthesizerTrn)

**Purpose**: Decode semantic tokens + phonemes + reference audio into a 32 kHz waveform. This is a generative audio synthesis model based on VITS.

**Architecture** (5 main components):

1. **Text encoder**. Maps phoneme IDs to a hidden representation. This is the most language-dependent part of the model and the one we *don't* want to overfit during fine-tuning — hence the lower learning rate for these layers in [05_train_sovits.py](../scripts/05_train_sovits.py).

2. **Semantic encoder**. Embeds semantic tokens into the same hidden space. This is what brings GPT's output into SoVITS.

3. **Posterior encoder**. Extracts a latent representation from a reference mel spectrogram during training. At inference time this is what tells SoVITS what speaker to imitate.

4. **Flow layers** (normalizing flow). Map the prior distribution to the posterior, conditioned on the encoded text+semantic. This is the speaker-conditioning mechanism — different reference clips produce different posteriors which produce different audio.

5. **HiFi-GAN decoder**. Upsamples the latent representation to a waveform via transposed convolutions. Upsampling rates: [10, 8, 2, 2, 2] = 640× total, taking the latent's hop-length features up to sample-rate audio.

**Multi-period discriminator** (training only): adversarial classifier that tries to distinguish real from generated waveforms at multiple periodicities. Used for the GAN loss.

**Training objective** (in `05_train_sovits.py`):
```
Loss_G = L1_mel + λ_kl · KL(z_p ∥ posterior)
       + λ_gen · adversarial_generator_loss
       + λ_fm · feature_matching_loss
       + KL_ssl  (auxiliary alignment of semantic features)

Loss_D = standard hinge-loss discriminator objective
```

The mel reconstruction loss is the dominant term. KL keeps the flow-induced prior close to the posterior. Adversarial + feature-matching losses sharpen the audio (prevent the muffled-mel-loss-only failure mode that vanilla VITS suffers from).

**Inference**: single forward pass.
```python
audio = vits.decode(
    pred_semantic,   # [1, 1, T_sem]   tokens from GPT
    phoneme_ids,     # [1, T_phone]
    [ref_spec],      # [1, 1025, T_ref]  mel from reference clip
    speed=1.0,
)
```

`extract_latent(ssl)` is a separate forward used only at data-prep time to *encode* HuBERT features into semantic tokens for training.

## Peripheral models (frozen)

| Model | Role | Output dim | Frame rate |
|---|---|---|---|
| **CNHuBERT** | Self-supervised speech rep extractor | 768 | 50 Hz |
| **Chinese-RoBERTa-wwm-ext-large** | BERT features for zh text | 1024 | per-character |
| **Faster-Whisper large-v3** | ASR transcription | string | (offline use only) |

CNHuBERT is critical: its 768-dim features are what the SoVITS quantizer turns into the 1024-vocab semantic tokens. The vocabulary is essentially "common patterns in HuBERT space across the pretraining data."

Chinese-RoBERTa is *only* used for Chinese text. For Japanese and English, the BERT branch receives a zero tensor. This is sometimes confusing — the network has the capacity to use BERT features, but for non-zh languages it's effectively unused. Don't try to substitute a Japanese or English BERT here; the dimensionality and tokenization don't match.

## Data flow during fine-tuning

```
For each training sample:
  audio.wav ──► CNHuBERT ──► [768, T_ssl] ──► quantize ──► semantic_ids: [1024-vocab, T_sem]
       │                                                              │
       │                                                              ▼
       └─► spectrogram ──► spec: [1025, T_spec]               GPT loss target
                                  │                                   ▲
                                  ▼                                   │
                          SoVITS loss target ◄── y_hat ◄── SoVITS forward
                                                                      │
                          phoneme_ids, bert ─────────────────────────┘

  → SoVITS loss = L1(spec_real, spec_pred) + KL + adversarial + feature_matching
  → GPT loss    = CE(predicted_semantic, target_semantic)
```

Both stages are trained **simultaneously** (well, sequentially in our pipeline — first SoVITS for 20 epochs, then GPT for 15) but they don't directly share gradients. They're two independent networks coupled by the semantic-token interface.

## What changes in v3 / v4

The architecture documented above is for **v2**. v3 and v4 use a different SoVITS — a flow-matching DiT (Diffusion Transformer) instead of VITS — and produce mel spectrograms that are then run through a separate vocoder for the final waveform.

Implications:
- v3/v4 do not have a discriminator during fine-tuning (the loss is a single conditional flow-matching objective).
- v4 outputs 48 kHz natively (vs 32 kHz for v2), via a custom vocoder.
- The GPT (s1) is largely unchanged — same architecture, same semantic-token interface.

This guide currently focuses on v2 because the training loop is simpler and Windows-friendly. v4 is on the roadmap.

## Further reading

- VITS paper: https://arxiv.org/abs/2106.06103 — the architectural ancestor of SoVITS s2.
- HuBERT paper: https://arxiv.org/abs/2106.07447 — the SSL model behind the 768-dim features.
- HiFi-GAN paper: https://arxiv.org/abs/2010.05646 — the vocoder/decoder pattern.
- VQ-VAE: https://arxiv.org/abs/1711.00937 — for the discrete representation idea.

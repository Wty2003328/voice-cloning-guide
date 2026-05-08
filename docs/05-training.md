# 05 — Training

Two networks, two trainers, two loss curves to watch. SoVITS first (slower), GPT second (fast).

## SoVITS training (s2)

Run [`05_train_sovits.py`](../scripts/05_train_sovits.py) after the data pipeline completes:

```bash
python 05_train_sovits.py --exp my_speaker --epochs 20
```

**Defaults**:
- 20 epochs
- batch size 4
- LR 1e-4 (`AdamW`, `betas=(0.8, 0.99)` from the s2.json config, exponential decay)
- Save every 5 epochs
- fp16 enabled

**What it does per step**:
1. Forward through SoVITS generator → predicted waveform `y_hat`.
2. Compute mel spectrograms from real and predicted waveforms.
3. Discriminator forward on `(y, y_hat.detach())` → discriminator loss → backward → step optimizer_D.
4. Discriminator forward again on `(y, y_hat)` (no detach) → adversarial generator loss + feature matching.
5. Total generator loss = mel + KL + KL_ssl + adversarial + feature_matching → backward → step optimizer_G.

**Loss interpretation**:
- `d` (discriminator): oscillates 2-4 throughout training. Stable values mean the GAN is balanced.
- `g` (generator adversarial): typically 1.5-3. Higher = generator hasn't fooled discriminator (still has work).
- `mel` (L1 on mel spectrogram, scaled by `c_mel=45`): the most important number. Should drop from ~25 at start to ~17-20 by mid-training. **If mel doesn't decrease, training failed** — check your data pipeline.
- `kl` (KL divergence, latent prior vs posterior): drops from ~2-3 to ~1-2.
- `kl_ssl` (auxiliary): typically 0 for our use case (feature already aligned).

**Why two LR groups**: the text encoder, encoder_text, and MRTE layers learn cross-speaker phoneme→audio mappings — fine-tuning them aggressively destroys multi-lingual capability. They get `LR × text_low_lr_rate` (typically 0.1×). The rest of the network (decoder, posterior encoder, flow) gets the full learning rate.

**Compute**: with ~150 slices and batch 4 you'll see ~40 batches/epoch. On RTX 5080 each epoch is ~70 seconds → 20 epochs in ~25 minutes. Older / smaller GPUs might take 2-3× longer.

**When to stop**: watch the `mel` loss specifically. Once it bottoms out and starts oscillating in a 3-4 unit range, additional epochs aren't helping. Earlier checkpoints (E5, E10) often produce more natural prosody than later ones; later checkpoints are sometimes "tighter" but artifact-prone. **Test multiple checkpoints in [07_inference.py](../scripts/07_inference.py) and pick by ear.**

## GPT training (s1)

Run [`06_train_gpt.py`](../scripts/06_train_gpt.py):

```bash
python 06_train_gpt.py --exp my_speaker --epochs 15
```

**Defaults**:
- 15 epochs
- batch size 4
- LR 1e-4 (warmup 200 steps, no decay)
- AdamW, weight decay 0.01

**What it does per step**: cross-entropy of predicted-vs-target semantic token sequence. Uses the model's `forward_old` method, which returns both loss and accuracy.

**Loss interpretation**:
- `loss`: starts around 1500-2400 (sum reduction over the full sequence — large numbers are normal). Should drop to ~250-500 by E15.
- `acc`: starts ~25-30%, climbs to 95-99%. **Don't aim for 100%** — that's pure memorization. If your final accuracy is 99.5%+, your model is overfitting and will produce stiff, repetitive prosody. 95-97% generalizes better.

**Compute**: each epoch is ~2-3 seconds on RTX 5080 (small model, small dataset). Total ~30-60 seconds. This is essentially free.

**Why so fast**: the GPT only sees phoneme IDs and BERT features as input, predicts semantic tokens as output. No audio processing in the loop, just a transformer doing language modeling.

## Hyperparameter cheat sheet

| Knob | Default | When to change |
|---|---|---|
| `--epochs` (sovits) | 20 | Lower (10-15) if training set > 30 min — diminishing returns. Higher rarely helps. |
| `--epochs` (gpt) | 15 | Lower if `acc` saturates at 99%+ — usually means you can stop at E10 or earlier. |
| `--batch-size` | 4 | Lower to 2 if you OOM. Higher (8) if you have abundant VRAM and >300 slices. |
| `--lr` | 1e-4 | Stable for fine-tuning. Don't change unless you know what you're doing. |
| `--save-every` | 5 | Lower to 1 if you want checkpoints at every epoch for selection. Costs disk space. |
| `--warmup-steps` (gpt) | 200 | Increase if loss oscillates wildly early on. |

## Picking checkpoints

After training, you'll have multiple SoVITS and GPT checkpoints. The naive choice is "the last one" — and that's often wrong.

**SoVITS**: test `_e5_`, `_e10_`, `_e15_`, `_e20_` outputs side by side. Listen for:
- **Voice match**: does it sound like the target speaker?
- **Naturalness**: prosody flowing or stiff?
- **Artifacts**: any robotic glitches, hisses, or background noise?

Earlier epochs may sound less "tight" but be more natural. Later epochs can introduce overfit artifacts (the model has memorized specific prosodic patterns from training and applies them inappropriately).

**GPT**: usually the last 2-3 checkpoints are interchangeable. If you see degradation at the last epoch (worse prosody than E13/E14), you trained too long.

The `--sovits-ckpt` and `--gpt-ckpt` flags in [`07_inference.py`](../scripts/07_inference.py) let you override the auto-selected latest checkpoint. Use them for A/B comparisons.

## Reading the training output

Sample SoVITS log line:
```
E7 [10/44] d=2.487 g=2.048 mel=17.4 kl_ssl=0.000 kl=1.733
```

This is epoch 7, batch 10 of 44, with discriminator loss 2.487, generator adversarial 2.048, mel L1 17.4 (capped display at 75 to avoid huge numbers when training fails), KL_ssl 0, KL 1.733. Healthy training looks like a slow downward trend on `mel` and `kl` with `d` and `g` oscillating in a stable range.

Sample GPT log line:
```
=== Epoch 12: avg_loss=419.8 avg_acc=0.947 (2s) ===
```

Average epoch loss 419.8, accuracy 94.7%, took 2 seconds. Healthy training shows monotonic loss decrease and accuracy increase per epoch.

## Common training failures

- **OOM during SoVITS**: batch is too big or sequences are too long. Drop batch to 2, or check that no slice in your data set is > 15 seconds.
- **mel loss not decreasing past E2**: data pipeline produced bad features. Verify `4-cnhubert/` and `5-wav32k/` counts match `2-name2text.txt`.
- **GPT loss explodes at first step**: pretrained checkpoint not loaded. Check the "Pretrained loaded" log line at startup.
- **GPT acc plateaus at 30-40%**: usually means the semantic token vocabulary doesn't match what the GPT expects. Re-run step 4 (semantic extraction) to make sure tokens are in [0, 1023].

## What's next

Once both trainers complete, you have fine-tuned weights at:
- `GPT-SoVITS/SoVITS_weights_v2/<exp>_e*_s*.pth`
- `GPT-SoVITS/GPT_weights_v2/<exp>-e*.ckpt`

Move on to [06 — Inference](06-inference.md) to actually generate speech.

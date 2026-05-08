# 08 — Extending the Model

After you have a working baseline, here's where to go next.

## More training data

Diminishing returns kick in around 30 minutes of clean speech. Beyond that, you mostly need *diverse* data, not more data. Useful axes of diversity:

- **Emotional range**: angry, calm, excited, sad, whispered. The model can only generate styles it's seen.
- **Sentence length variety**: short interjections + long narrative.
- **Pitch range**: include low murmurs and high exclamations if the speaker uses them.
- **Speaking modes**: declarative, interrogative, imperative.

Adding 5 minutes of varied emotion to a 30-minute baseline often improves outputs more than another 30 minutes of monotone narration.

**How to add data**: drop additional `.wav` files into your source directory, concatenate them into the existing vocals file, then re-run the pipeline starting from step 1 (slicing). Existing `4-cnhubert/`, `3-bert/` files are preserved (re-extraction skips files that already exist), but it's cleanest to delete the `logs/<exp>/` directory and start fresh.

## Multi-speaker training

The data format supports multiple speakers via the `speaker` column in `2-name2text.txt`. To train multi-speaker:

1. Run the pipeline once per speaker with `--speaker speaker_a`, `--speaker speaker_b`, etc.
2. Concatenate the per-speaker `2-name2text.txt`, `6-name2semantic.tsv`, and copy all `4-cnhubert/`, `5-wav32k/`, `3-bert/` contents into one `logs/<combined_exp>/`.
3. Train the combined experiment.

At inference time, switch speakers by changing the `--ref-wav`. The model conditions on the reference's mel spectrogram, so different references produce different speakers — even from the same fine-tuned model.

This works best when speakers have similar speech styles (e.g., same gender, similar accents). Mixing wildly different speakers in one fine-tune can confuse the model.

## Training more epochs vs longer per epoch

If your `mel` loss is still decreasing at E20, try `--epochs 30`. If it plateaued by E10, stop earlier — extra epochs add overfitting without quality gains.

The GPT trains so fast that running 25 epochs vs 15 costs almost nothing. Pick by accuracy: target 95-97% final accuracy. Higher = overfitting; lower = under-trained.

## Mixing languages in training

Including multiple languages in a single training set produces better cross-lingual outputs.

Recipe:
1. Slice and ASR each language separately (different `--lang` flag per run).
2. Combine the resulting `2-name2text.txt` files manually (TAB-separated, easy to concat).
3. Re-run feature extraction with the combined file.
4. Train normally.

Caveats:
- Phoneme inventories differ per language but the model handles this — phoneme IDs are unified across languages in the v2 phonemizer.
- If you have only 5 minutes of English mixed into 30 minutes of Japanese, English output will still feel Japanese-accented. Aim for >30% of the secondary language for a noticeable accent shift.

## LoRA fine-tuning

The upstream GPT-SoVITS supports LoRA fine-tuning for v3/v4 — much smaller adapter checkpoints (~10 MB instead of 700 MB). v2 doesn't have LoRA support out of the box.

If model size matters (deploying many speaker variants), upgrading to v4 + LoRA is the path. v4 support in this repo is on the roadmap.

## Real-time TTS / streaming

The current `07_inference.py` waits for the full GPT generation to complete before SoVITS decoding. For streaming TTS:

- Modify the GPT loop to yield semantic tokens incrementally.
- Buffer 25-50 tokens (1-2 seconds of audio) and call SoVITS decode in chunks.
- Concatenate chunks with crossfade at boundaries to avoid clicks.

The GPT is fast enough that real-time playback is feasible if you can keep the SoVITS decoder pipelined.

## Integrating with VTuber pipelines

A common use case is wiring this into a VTuber stack:

1. **LLM** generates dialogue text → 2. **GPT-SoVITS** synthesizes voice → 3. **Live2D / VRM** model animates with lipsync → 4. Audio + video out to OBS.

For lipsync, you can either:
- Use audio-to-mouth movement libraries like [VTube Studio's automatic lip sync](https://denchisoft.com/) (works on output WAV).
- Or extract phoneme timing from the SoVITS output (semantic tokens are roughly aligned at 25 Hz) and drive Live2D parameters directly.

The 0.5-2 second per-sentence latency limits this to non-real-time use unless you implement streaming.

## Voice conversion via SoVITS

SoVITS can do voice *conversion* (warp existing audio to a target speaker) in addition to text-to-speech. The `extract_latent` method gives you semantic tokens from arbitrary input audio, which you can then feed back through `decode` with a different reference.

This is similar to RVC but uses the SoVITS model directly. Quality is comparable to RVC for the same compute. The advantage: one model handles both TTS and VC for a given speaker.

## Better quality via more pretraining

If you have a custom pretraining dataset (e.g., 100+ hours of a specific domain), you can pretrain SoVITS from scratch and then fine-tune. This is rarely worth it for individual users, but for production work on a specific accent or speaking style (e.g., medical narration, audiobook reading) it can yield a stronger base for fine-tuning.

The v2 base model was trained on ~5,000 hours; matching that scale costs serious compute (multi-GPU, days). For most users, fine-tuning the existing pretrained weights is the right move.

## Troubleshooting recipes

**"My fine-tuned model sounds like the original speaker (no change)"**: training failed to update SoVITS. Check that `05_train_sovits.py` actually saved a checkpoint and you're loading it (not the pretrained `s2G2333k.pth`). The checkpoint filename should start with your `--exp` name.

**"My fine-tuned model lost the ability to speak other languages"**: text encoder over-fine-tuned. Drop `text_low_lr_rate` from 0.4 to 0.1 (in `s2.json` config) and retrain. Or use earlier checkpoints.

**"Output volume is wildly inconsistent"**: training data wasn't normalized. Re-run step 3 with the default normalization scaling (`maxx=0.95, alpha=0.5`).

**"Output is missing punctuation pauses"**: the phonemizer collapsed punctuation. Check your `2-name2text.txt` — the phoneme sequence should include rest tokens for commas/periods. If it doesn't, your input transcripts didn't have punctuation.

## Closing

GPT-SoVITS v2 is a flexible base. The architecture's clean separation of GPT (prosody) and SoVITS (timbre) makes most modifications local to one stage. Want better timbre? Improve SoVITS training data and epochs. Want better prosody? Add varied emotional examples to GPT training. The two stages compose cleanly.

Open issues / PRs at this repo welcome — particularly for bugfixes, new docs, and v4 support.

# 06 — Inference

How to actually generate speech from a fine-tuned model.

## The mental model

Inference takes three things:

1. **Text** to synthesize, plus a language tag.
2. **Reference audio** (3-15 seconds of the target speaker) plus its transcript.
3. The fine-tuned **GPT** + **SoVITS** checkpoints.

The reference audio plays two distinct roles:

- The reference's **first 50 semantic tokens** are fed as a *prompt prefix* to the GPT. This biases the GPT to continue in the speaker's prosodic style — think of it as priming an LLM with a few examples before asking the real question.
- The reference's **full mel spectrogram** is passed to the SoVITS posterior encoder. This is what locks in speaker identity (timbre).

If you skip the reference, the model has nothing to anchor speaker identity to and produces generic averaged-speaker output. The reference is mandatory.

## Running it

```bash
python 07_inference.py \
    --exp my_speaker \
    --text "Hello! I'm your character. Nice to meet you." \
    --lang en \
    --ref-wav ../GPT-SoVITS/logs/my_speaker/0_sliced/0003.wav \
    --ref-text "ここは私に任せて私を選んでくれる" \
    --ref-lang ja \
    --out hello.wav
```

The text and reference can be in **different languages**. Train on Japanese, prompt in Japanese, generate in English — this works because the SoVITS speaker conditioning is language-agnostic at the timbre level.

By default the script picks the latest checkpoint matching the `--exp` name. Override with `--sovits-ckpt` and `--gpt-ckpt` for A/B comparisons.

## Choosing a good reference clip

This matters more than people expect. A bad reference will sabotage even a perfectly-trained model.

**Good reference** characteristics:
- 5-10 seconds long (3 is too short; 15+ is overkill).
- Single sentence, clean delivery, neutral emotion.
- Clear speech, no background noise, no laughter / coughs / sighs.
- The transcript matches the audio exactly. Misalignment hurts.
- Same recording conditions as your training data.

**Bad references** that produce artifacts:
- Background music or reverb (the model copies these into outputs).
- Very emotional delivery (screaming, whispering) — biases prosody dramatically.
- Multi-speaker clips.
- Clips with non-speech vocalizations (sighs, gasps, laughter at boundaries).

Pick from your `0_sliced/` directory by listening to a few candidates. `0003.wav` to `0010.wav` is usually a good range — you avoid the very first/last slices which often have edge artifacts.

## Sampling parameters

GPT inference is autoregressive sampling, controlled by three knobs:

| Parameter | Default | What it does |
|---|---|---|
| `--top-k` | 15 | Sample from the top 15 tokens at each step. Lower → more deterministic / repetitive. Higher → more variation, more risk of weird outputs. |
| `--temperature` | 1.0 | Sampling sharpness. <1.0 is more conservative; >1.0 is more creative. >1.5 usually breaks things. |
| `--repetition-penalty` | 1.35 | Penalizes repeated token n-grams. Prevents the model from getting stuck saying the same syllable. |

The defaults are well-tuned for typical use. Reasons to deviate:

- **Output sounds robotic / monotone**: increase `--top-k` to 25-50, or `--temperature` to 1.1-1.2.
- **Output is unstable / hallucinated**: decrease `--top-k` to 5-10, `--temperature` to 0.8.
- **Same syllable repeats**: increase `--repetition-penalty` to 1.5+.

The `early_stop_num` is hardcoded to 54 seconds worth of tokens. The model emits an EOS token when it's "done" with the input text — usually long before this limit.

## Inference speed

On RTX 5080:
- Model loading: ~5-10 seconds (one-time per process).
- GPT autoregressive generation: ~150 tokens/second → most sentences finish in 0.5-2 seconds.
- SoVITS decode: ~0.1-0.3 seconds.
- Total per sentence: **~0.5-2 seconds**, faster than real-time playback.

The `07_inference.py` script generates one sentence per invocation. For batch generation, hold the model in memory and call `synthesize` repeatedly — see how `compare_v2_vs_combined.py` (in the project's history) does it.

## Failure modes and fixes

**Output is silence or near-silence (< 1 second)**:
The GPT emitted EOS too early. Causes and fixes:
- The reference's semantic prefix happens to end on an EOS-like token. Pick a different `--ref-wav`.
- Top-k 15 occasionally samples EOS by chance. Increase `--top-k` to 25 and try again.

**Output sounds like a different speaker**:
- Wrong reference clip. Verify `--ref-wav` is from the target speaker.
- SoVITS checkpoint not loaded. Check the "Loading SoVITS" line shows your fine-tuned `.pth`, not a pretrained one.

**Output has background noise / hum**:
- Training data wasn't actually clean — re-run Demucs.
- Reference clip has noise — try a cleaner reference.

**Output truncates mid-sentence**:
- Increase `early_stop_num` in `07_inference.py` (currently `sampling_rate / hop_length * 54` ≈ 1500 tokens).
- For very long sentences (>20 seconds), split into shorter chunks and concatenate.

**Output sounds tonally off (Chinese)**:
- The Chinese phonemizer is tone-aware but the model can sometimes flatten tones, especially for speakers who didn't train on Chinese. This is a known limitation; running with `--temperature 1.1` sometimes helps.

**Output sounds too fast / too slow**:
- Pass `speed=` to the SoVITS decode call (currently hardcoded to 1.0). Values 0.8-1.2 work; outside that range introduces artifacts.

## Multi-language tricks

GPT-SoVITS handles cross-lingual generation reasonably well, but quality varies:

- **Train on JP, generate JP**: best quality.
- **Train on JP, generate EN**: works; English will have a Japanese accent character.
- **Train on JP, generate ZH**: works but tones can be inaccurate.
- **Train on multiple languages**: better cross-lingual quality but more training data needed (15+ min mixed).

If you want strong English output, include English clips in your training set. Even 2-3 minutes of mixed English can dramatically improve the English voice.

## What's next

- [07 — Windows guide](07-windows-guide.md) for OS-specific gotchas you might hit.
- [08 — Extending](08-extending.md) for ideas on improving quality further.

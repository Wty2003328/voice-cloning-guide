# 10 — Zero-Shot Voice Cloning (Qwen3-TTS)

Clone any voice from 3–30 seconds of audio. No GPU training, no
LoRA, no hyperparameter tuning. This guide walks through the whole
end-to-end workflow using **Qwen3-TTS-12Hz-1.7B-Base**, which we
picked over CosyVoice 3 / IndexTTS-2 / Higgs / ChatterBox after a
multi-round listening test (see [02-comparison.md](02-comparison.md)).

## What you'll have at the end

A working Python wrapper that takes a target text + your reference
clip and emits a WAV in the cloned voice. Plus an OpenAI-compatible
HTTP sidecar so the same model plugs into any chat/companion app
through the [universal TTS spec](12-integration.md).

## Prerequisites

- Python 3.10 or newer
- ~10 GB free disk (model weights are ~4 GB + caches)
- NVIDIA GPU with ≥4 GB VRAM (CPU works at ~3-5× real-time)
- A reference audio file of the voice you want to clone — see [§Reference clip](#reference-clip) for what makes a good one

## Install

```bash
# Create a clean env (recommended)
conda create -n qwen3-tts -y python=3.10
conda activate qwen3-tts

# Install qwen-tts (pulls in transformers, accelerate, etc.)
pip install -U qwen-tts huggingface_hub soundfile

# PyTorch with CUDA — match your CUDA version (12.8 example)
pip install torch --index-url https://download.pytorch.org/whl/cu128

# Optional, faster inference (sometimes finicky to build on Windows)
pip install -U flash-attn --no-build-isolation
```

## Download model weights

```bash
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-Base --local-dir ./qwen3-tts-1.7b-base
huggingface-cli download Qwen/Qwen3-TTS-Tokenizer-12Hz --local-dir ./qwen3-tts-tokenizer
```

This downloads ~4 GB. The tokenizer is needed for the streaming codec
path; download both even if you only plan to use non-streaming.

## Reference clip

The voice identity comes **entirely** from the reference clip. Picking
it well is the most consequential single choice.

### Length

- **Minimum:** ~3 seconds. The model docs claim "rapid clone" at 3s
  and that's true, but you'll lose nuance.
- **Sweet spot:** **20–32 seconds** of *multi-clip* reference. A long
  reference gives the speaker encoder a richer fingerprint and stops
  the model from over-fitting to whatever prosody happens to be in a
  single short clip.
- **Maximum:** soft ceiling around 32 seconds. We measured AR
  hallucinations (audio looping, breathy non-utterance) above ~49s.
  **Don't exceed 32s.**

### Cleanliness

- No background music, no overlapping speakers, no laugh tracks.
- Trim leading/trailing silences.
- Normalize loudness so peak amplitude is ~0.8 (loud but not clipped).
  Quiet refs (<0.3 peak) give weak speaker embeddings.

### Prosody diversity

Critical for cross-utterance consistency. If your reference is all
*declarative monologue*, the model will over-emote on casual targets
(*"こんにちは!"* sounds like Shakespeare). A good multi-clip mix spans:

- A declarative / narrative sentence
- A questioning sentence
- An emotional / emphasized utterance
- A casual / short / filler-rich sentence
- An energetic / playful clip

`scripts/build_reference.py` automates the selection and concatenation
(loudness-normalize + silence-trim + 0.3s gaps between clips).

### Transcript

Each reference clip needs an exact transcript. The model uses it to
compute paired prompt features that improve same-language cloning.
Run your reference through faster-whisper or write it manually:

```bash
python -c "
from faster_whisper import WhisperModel
m = WhisperModel('small', device='cuda', compute_type='float16')
segs, _ = m.transcribe('my_reference.wav', language='ja')
print(''.join(s.text for s in segs))
"
```

## Minimal inference

```python
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "./qwen3-tts-1.7b-base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="sdpa",  # or "flash_attention_2" if installed
)

# Build (and cache) the prompt features for this reference clip
prompt = model.create_voice_clone_prompt(
    ref_audio="my_reference.wav",
    ref_text="この間、コンテン神社でポテトチップスを食べてる女の子を見かけたの。",
    x_vector_only_mode=False,  # see §Hybrid mode below
)

# Synthesize
wavs, sr = model.generate_voice_clone(
    text="こんにちは、私は人工知能アシスタントです。",
    language="Japanese",   # one of: Japanese, English, Chinese, Korean,
                           # German, French, Russian, Portuguese, Spanish, Italian
    voice_clone_prompt=prompt,
    temperature=0.4,       # see §Sampling tuning
    top_p=0.85,
    max_new_tokens=240,
)
sf.write("hello_cloned.wav", wavs[0], sr)
```

## Sampling tuning

The `temperature` and `top_p` parameters are the main quality dials.
Both control how strict the model is to the speaker embedding's
distribution.

| Preset | temperature | top_p | max_new_tokens | Use when |
|--------|-------------|-------|----------------|----------|
| `fast` | 0.6 | 0.70 | 200 | Real-time conversation, snappy chunks |
| `balanced` (default) | 0.4 | 0.85 | 240 | Most cases — natural voice fidelity |
| `high` | 0.3 | 0.90 | 320 | Long-form / important responses |

**Lower temperature** = stricter to reference = more in-character but
less prosodic variation. **Higher temperature** = more natural prosody
but voice drift. The 2026-validated sweet spot for character voice
cloning is `temperature=0.4` (picked by user listening test from a
28-sample sweep).

The `max_new_tokens` cap is a **loop safety net** — Qwen3-TTS speech
tokens run at 12 Hz, so 240 tokens caps output at ~20 s of audio. Set
this even when you don't expect long outputs; it prevents runaway
generation.

## Hybrid mode: same-language vs cross-lingual

Qwen3-TTS's `create_voice_clone_prompt` has an `x_vector_only_mode`
toggle:

- **`False` (default):** uses both the speaker embedding AND
  prompt-text-paired features. Best voice clone fidelity when the
  target language matches the reference language.
- **`True`:** uses **only the speaker embedding**. Best for
  cross-lingual targets — paired features carry the reference
  language's phonotactics, which bleed into the target if you don't
  drop them.

**Recipe:** if your reference is JA and you're synthesizing JA, use
`False`. For JA → ZH / KO / EN cross-lingual, use `True`. Cache both
prompts so synthesis-time switching is free:

```python
prompts = {
    "same_lang":     model.create_voice_clone_prompt(..., x_vector_only_mode=False),
    "cross_lingual": model.create_voice_clone_prompt(..., x_vector_only_mode=True),
}

def synth(text, language):
    is_same = (language == reference_language)
    p = prompts["same_lang"] if is_same else prompts["cross_lingual"]
    return model.generate_voice_clone(text=text, language=language,
                                       voice_clone_prompt=p,
                                       temperature=0.4, top_p=0.85)
```

This hybrid pattern was the key win that took our cross-lingual ZH
score from 20% → 90% on a 33-case robustness eval.

## DO NOT pre-normalize loanwords

If you're carrying over text normalization code from a GPT-SoVITS
era (e.g., converting "iPhone" → "アイフォン" before synthesis), **turn
it off for Qwen3-TTS**. The LLM-based text encoder handles Latin
loanwords, digits, and acronyms natively and produces canonical
katakana. Manual normalization actually *hurts* — we measured cases
where forcing "iPhone" → "イフォウン" made the model speak gibberish
that it would have rendered correctly given the raw Latin input.

## Common failure modes & fixes

| Symptom | Cause | Fix |
|---------|-------|-----|
| Output is silent / 0.5 s of breathy noise | `max_new_tokens` was hit on a short non-converging path | Bump temperature 0.1; usually unsticks |
| Output goes into a loop ("好嘞, 好嘞, 好嘞...") | AR diverged with no token-cap stop | Lower `max_new_tokens` — we cap at 240 for 20s audio |
| Voice drifts mid-utterance | Reference clip was too short / had too much prosodic variation | Use longer multi-clip ref; lower temperature 0.1 |
| Casual sentence sounds over-emoted | Reference is all monologue, no casual prosody | Add a casual/short clip to the multi-clip ref |
| Cross-lingual (JA→ZH) sounds JA-ish | x_vector_only mode wasn't switched on for cross-lingual | See §Hybrid mode |
| EN/ZH/KO pronunciation is wrong | You pre-normalized loanwords | Disable normalization (see above) |

## Validation: the 33-case robustness eval

We validate every config change against a 33-case eval covering
casual/loanword/digit/acronym/multi-sentence/special-char/name/llm-output
patterns across ja/en/zh. Pass criteria per case: jaccard ≥ 0.40 to
the gold transcript + length-ratio in [0.5, 1.6].

Production config (`reference=asuna_concat_diverse5`,
`temperature=0.4`, `top_p=0.85`, x_vector_only hybrid mode, no
normalization) scores **32/33 (97%)**:

- JA: 13/13 (100%) — clean + loanword + digit + acronym + name + special + multi all 100%
- EN: 10/10 (100%)
- ZH: 9/10 (90%) — sole failure is a Whisper-side homophone artifact, not a real TTS bug

Reproduce with `scripts/eval_robustness.py`.

## Next steps

- **Multilingual + cross-lingual deep-dive:** [11-multilingual.md](11-multilingual.md)
- **Integration into a chat app:** [12-integration.md](12-integration.md)
- **What if zero-shot isn't enough?** Consider Path B (fine-tune GPT-SoVITS) —
  start with [01-theory.md](01-theory.md) for the theory.

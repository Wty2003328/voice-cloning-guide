# 10 — Zero-Shot Voice Cloning

Zero-shot voice cloning is a one-line idea: **give the model a 5–10 s
reference clip plus its transcript, and it speaks any text in that
voice.** No GPU training, no LoRA, no dataset.

This chapter walks through the modern deploy path — pulling a TTS
container, sending an HTTP request, and tuning the reference clip
before reaching for fine-tuning.

## What "zero-shot" requires

Three inputs at request time:

| Input | What it is | Notes |
|---|---|---|
| `ref_audio` | 5–10 s of clean target-voice audio | Single speaker, no music, peak ~0.8. |
| `ref_text` | Exact transcript of `ref_audio` | The same script the speaker reads — used to anchor prompt features. |
| `input` | The new text you want spoken | Any sentence in any supported language. |

No training data, no hyperparameter sweep. The cost is that voice
fidelity depends entirely on the reference clip — see
[ch. 16 — OmniVoice SFT](16-omnivoice-sft-recipe.md) if you need
character-level timbre.

## The concrete recipe

The recommended deploy in 2026 is **vLLM-Omni in Docker**: one
container, one OpenAI-compatible URL, swap models with a profile flag.
See [ch. 15 — vLLM-Omni Docker](15-vllm-omni-docker.md) for the full
walkthrough; the short version is:

```bash
# 1. Bring up the default OmniVoice service.
docker compose up -d

# 2. Verify the model is loaded.
curl -s http://127.0.0.1:8000/v1/models
# → { "data": [ { "id": "k2-fsa/OmniVoice", ... } ] }
```

A minimal `docker-compose.yml` for the default zero-shot service:

```yaml
services:
  omnivoice:
    image: vllm/vllm-omni:v0.20.0
    ports:
      - "8000:8000"
    volumes:
      - hf-cache:/root/.cache/huggingface
      - ./refs:/refs:ro
    runtime: nvidia
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    ipc: host
    shm_size: '8gb'
    command: >
      vllm serve k2-fsa/OmniVoice
      --omni --host 0.0.0.0 --port 8000
      --gpu-memory-utilization 0.40
      --enforce-eager
      --max-model-len 2048 --max-num-seqs 1
      --allowed-local-media-path /refs

volumes:
  hf-cache:
```

## Sending a synth request

The body is OpenAI-compatible plus three extension fields
(`language`, `ref_audio`, `ref_text`):

```bash
curl -X POST http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d "$(cat <<'JSON'
{
  "model":     "k2-fsa/OmniVoice",
  "input":     "今日もよろしくお願いします。",
  "language":  "Japanese",
  "ref_audio": "data:audio/wav;base64,<base64 of your reference clip>",
  "ref_text":  "<exact transcript of the reference clip>"
}
JSON
)" \
  -o hello.wav
```

Wire-contract gotchas (same as ch. 15):

1. `ref_audio` must be a `data:audio/wav;base64,...` URL. File paths
   are rejected by the media connector.
2. `language` is the English name ("Japanese", "English", "Chinese"),
   not a BCP-47 code.
3. Response is raw 24 kHz mono WAV (`Content-Type: audio/wav`).
   Errors come back as JSON.

A reference Python client lives in `scripts/zero_shot_clone.py`.

## Per-model nuances

The same docker-compose flow serves multiple models behind profile
flags. Each has different strengths for zero-shot:

| Model | Profile | Strength | Caveats |
|---|---|---|---|
| **OmniVoice** (`k2-fsa/OmniVoice`) | _default_ | Japanese (36k+ hr JA pretrain, char-level Qwen3 tokenizer, FLEURS JA CER 5.96). | Mediocre EN baseline; not yet evaluated for ZH at the same depth. |
| **CosyVoice3** (`FunAudioLLM/Fun-CosyVoice3-0.5B-2512`) | `--profile cosy3` | Chinese (CER 0.81% native ZH; SIM 78% beats human ref 75.5%). | Trained on kana-converted text; raw kanji input degrades JA content fidelity. |
| **Qwen3-TTS** (`Qwen/Qwen3-TTS-12Hz-1.7B-Base`) | `--profile qwen` | Multilingual baseline across JA/ZH/EN/KO + 6 EU languages. | Generic — beaten on JA by OmniVoice and on ZH by CosyVoice3. |

For the full empirical comparison and rejection rationale see
[ch. 15 — Picking a vLLM-Omni model](15-vllm-omni-model-selection.md).

## Tuning the reference clip

If the output sounds *close but not right*, vary the reference clip
before reaching for SFT.

### Length

| Length | What happens |
|---|---|
| < 3 s | Speaker fingerprint too thin; over-fits to short prosody. |
| 5–10 s | **Sweet spot.** Single clean utterance. |
| 20–30 s | Multi-clip reference; richer fingerprint, better cross-utterance consistency. |
| > 32 s | AR hallucinations measured above ~49 s; avoid. |

### Cleanliness

- No background music, no overlapping speakers, no laugh tracks.
- Trim leading/trailing silences (~50 ms padding is fine).
- Normalize loudness so peak amplitude is ~0.8 — quiet refs (< 0.3
  peak) give weak speaker embeddings.

### Matching emotional register

The reference clip's prosody leaks into the output. If your reference
is a calm declarative monologue and you ask the model to synthesize a
casual greeting, the greeting will come out over-emoted.

Recipe: build a multi-clip reference that spans the prosody range of
your eventual use case — a declarative line, a question, a casual
short line, an energetic line — concatenated with ~0.3 s gaps and
re-normalized.

### Transcript accuracy

`ref_text` must be the **exact** transcript of `ref_audio`. A
mistranscription (added word, dropped particle, wrong kanji) directly
degrades prompt features. If you don't have the transcript, run
faster-whisper:

```python
from faster_whisper import WhisperModel
m = WhisperModel("large-v3", device="cuda", compute_type="float16")
segs, _ = m.transcribe("my_reference.wav", language="ja")
print("".join(s.text for s in segs))
```

## When zero-shot isn't enough — reach for SFT

Zero-shot zero-shot ceiling is "in the ballpark — listenable, right
gender / age range, plausible timbre." For distinctive character
voices (anime, VTuber, fictional character), the missing 5–10% of
timbre fidelity matters and zero-shot can't close it just by changing
the reference clip.

The next step is **supervised fine-tuning (SFT)** on ~10–20 minutes of
the target voice. Full recipe at
[ch. 16 — OmniVoice SFT](16-omnivoice-sft-recipe.md). End-to-end
wallclock on a 16 GB consumer GPU is ~50 minutes (dataset prep +
8-minute training run + checkpoint deploy).

## Common failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| Output is silent or 0.5 s of breathy noise | Speaker fingerprint too weak | Use a longer / cleaner reference clip. |
| Output loops ("好嘞, 好嘞, 好嘞…") | AR diverged | Switch to a different reference clip; verify `ref_text` matches exactly. |
| Voice drifts mid-utterance | Reference too short, or prosody mismatch | Use a 20–30 s multi-clip reference; match register. |
| Casual sentence sounds over-emoted | Reference is all monologue | Add a casual / short clip to the reference set. |
| Cross-lingual output sounds like the reference language | Phonotactic leak from short reference | See [ch. 14 — cross-lingual limits](14-cross-lingual-limits.md). |

## See also

- [ch. 11 — Multilingual](11-multilingual.md) — serving many languages
  through one or many models.
- [ch. 14 — Cross-lingual limits](14-cross-lingual-limits.md) —
  empirical accent-leak measurements when the reference language differs
  from the target.
- [ch. 15 — vLLM-Omni Docker](15-vllm-omni-docker.md) — full deploy
  walkthrough including troubleshooting and profile switching.
- [ch. 15 — Picking a model](15-vllm-omni-model-selection.md) —
  per-model eval that drove the defaults.
- [ch. 16 — OmniVoice SFT](16-omnivoice-sft-recipe.md) — when
  zero-shot timbre isn't close enough.

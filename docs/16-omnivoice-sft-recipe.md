# 16 — OmniVoice SFT: fine-tuning a character voice

How to lift OmniVoice's base zero-shot timbre to a clearly-better
character match using ~10–20 minutes of clean target-voice audio + the
official k2-fsa training recipe. End-to-end wallclock: **~50 minutes**
on a single 16 GB Blackwell GPU.

This pairs with [ch. 15 — vLLM-Omni Docker](15-vllm-omni-docker.md): the
fine-tuned checkpoint slots straight into the same `docker-compose.yml`.

## Why SFT, not just better reference clips?

Base OmniVoice zero-shot (from a single 5–10 s ref clip) already hits
~0.95 char-jaccard on a 21-prompt content-fidelity battery. The
bottleneck is **timbre fidelity** — does the produced voice actually
*sound* like your target?

OmniVoice's design philosophy is "good ref clip > SFT" — the authors
bet on the 36k-hour pre-train absorbing speaker diversity at inference
time. In practice, for a distinctive character voice:

- Base zero-shot timbre: "in the right ballpark; listenable."
- Same model after 400 SFT steps on ~20 min of one voice: "**that's
  the character**" on subjective A/B.

The SFT shift is small but audible. Worth doing if the character voice
matters; skip it if any clean voice in the right gender / age band is
acceptable.

## Dataset requirements

| Requirement | Minimum | Sweet spot |
|---|---|---|
| Total clean target-voice audio | ~10 min | ~20–30 min |
| Number of clips | ~50 | ~100–200 |
| Per-clip duration | 1 – 10 s | 1 – 8 s |
| Sample rate | any (auto-resampled to 24 kHz) | 22.05 kHz or 44.1 kHz |
| Background noise / music | minimal | none (use Demucs to isolate) |
| Transcript per clip | required | accurate Japanese kana / kanji |

A typical anime / VTuber recording set after slicing yields ~100–200
1–8 s clips and ~15–25 min total — right in the sweet spot.

## Step-by-step

In the rest of this guide we use `mychar` as the placeholder name for
your character. Substitute anything (e.g. `narrator`, `alice`, etc.).

### 0. Prereqs

Same as [ch. 15](15-vllm-omni-docker.md): Docker + nvidia-container-toolkit
+ NVIDIA GPU. Plus a local clone of the OmniVoice repo:

```bash
git clone https://github.com/k2-fsa/OmniVoice
cd OmniVoice
```

And a training image (extends `vllm/vllm-omni:v0.20.0` with
`fugashi+unidic-lite`, `num2words`, `accelerate`, `webdataset`,
`transformers>=5.3`):

```bash
docker build -t omnivoice-train:0.1 \
  -f ../vllm-omni-deploy/Dockerfile.omnivoice-train \
  ../vllm-omni-deploy/
```

### 1. Prepare the dataset

Your input should look like this (path-relative is fine):

```
my_dataset/
├── asr.list            # one line per clip: "audio/0000.wav|speaker|JP|transcript"
└── audio/
    ├── 0000.wav
    ├── 0001.wav
    └── ...
```

`asr.list` format (pipe-separated):

```
audio/0000.wav|mychar|JP|今日は本当にいい天気ですね。
audio/0001.wav|mychar|JP|散歩に行きませんか。
audio/0002.wav|mychar|JP|この前の話なんだけど…
```

If you're starting from a single long recording, the legacy
[`scripts/sovits-finetune/`](../scripts/sovits-finetune/) directory has
the slicing + ASR pipeline that produces this exact format.

### 2. Build JSONL manifests

OmniVoice trains from JSONL with `{id, audio_path, text, language_id}`
rows. Convert `asr.list` → train/dev JSONL:

```python
# build_manifest.py — copy this into OmniVoice/data/
import json, random
from pathlib import Path

DATA = Path("/data/mychar")  # bind-mount target inside the container
OUT  = Path(__file__).parent

rows = []
with open(DATA / "asr.list", encoding="utf-8") as f:
    for line in f:
        rel_path, speaker, lang, text = line.strip().split("|", 3)
        rows.append({
            "id":          Path(rel_path).stem,
            "audio_path":  f"/data/mychar/{rel_path}",
            "text":        text.strip(),
            "language_id": "ja",
        })

random.seed(42)
random.shuffle(rows)
dev = rows[:max(7, len(rows)//20)]
train = rows[len(dev):]

with open(OUT / "mychar_train.jsonl", "w", encoding="utf-8") as f:
    for r in train: f.write(json.dumps(r, ensure_ascii=False) + "\n")
with open(OUT / "mychar_dev.jsonl", "w", encoding="utf-8") as f:
    for r in dev:   f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"train={len(train)} dev={len(dev)}")
```

For a typical ~140-clip dataset: ~130 train, ~7 dev.

### 3. Custom training config

Copy `examples/config/train_config_finetune_sdpa.json` to
`train_config_mychar.json` and tune for your dataset size:

```json
{
  "llm_name_or_path": "Qwen/Qwen3-0.6B",
  "audio_vocab_size": 1025,
  "audio_mask_id":    1024,
  "num_audio_codebook":   8,
  "audio_codebook_weights": [8, 8, 6, 6, 4, 4, 2, 2],
  "drop_cond_ratio":  0.1,
  "prompt_ratio_range": [0.0, 0.3],
  "mask_ratio_range":   [0.0, 1.0],
  "language_ratio": 0.8,

  "init_from_checkpoint": "k2-fsa/OmniVoice",

  "learning_rate":              5e-6,          // ← lower than default 1e-5 (small data)
  "weight_decay":               0.01,
  "max_grad_norm":              1.0,
  "steps":                      400,           // ← was 5000; 20-min data needs ~400
  "seed":                       42,
  "warmup_type":                "ratio",
  "warmup_ratio":               0.05,

  "batch_tokens":               1024,          // ← was 8192; tight VRAM
  "gradient_accumulation_steps": 8,            // ← compensate effective batch
  "num_workers": 2,

  "mixed_precision":            "bf16",
  "allow_tf32":                 true,
  "attn_implementation":        "sdpa",        // ← flex_attention OOMs in backward on Blackwell
  "max_sample_tokens":          1500,
  "min_sample_tokens":          50,
  "max_batch_size":             4,

  "logging_steps":              20,
  "eval_steps":                 100,
  "save_steps":                 100,
  "keep_last_n_checkpoints":    3
}
```

Plus a one-line data config (`examples/config/data_config_mychar.json`):

```json
{
  "train": [ { "manifest_path": ["/workspace/OmniVoice/data/finetune_mychar/tokens/train/data.lst"] } ],
  "dev":   [ { "manifest_path": ["/workspace/OmniVoice/data/finetune_mychar/tokens/dev/data.lst"]   } ]
}
```

### 4. Tokenize audio (Stage 0)

The OmniVoice training script runs in two stages. Stage 0 uses the
Higgs audio tokenizer to convert WAVs → discrete codes, packed into
WebDataset shards:

```bash
docker run --rm --gpus all \
  -v $(pwd):/workspace/OmniVoice \
  -v /path/to/my_dataset:/data/mychar \
  -v hf-cache:/root/.cache/huggingface \
  -e HF_HOME=/root/.cache/huggingface \
  omnivoice-train:0.1 \
  bash -c '
    cd /workspace/OmniVoice
    pip install -e . --no-deps >/dev/null
    stage=0 stop_stage=0 bash examples/run_finetune_mychar.sh
  '
```

For ~140 clips: ~30 s on a Blackwell GPU. Outputs:

```
data/finetune_mychar/tokens/
├── train/  # ~40 shards
│   ├── audios/shard-000000.tar  …
│   ├── txts/shard-000000.jsonl  …
│   └── data.lst
└── dev/    # ~7 shards
```

### 5. Train (Stage 1)

```bash
docker run -d --name omnivoice-train-mychar --gpus all \
  -v $(pwd):/workspace/OmniVoice \
  -v /path/to/my_dataset:/data/mychar \
  -v hf-cache:/root/.cache/huggingface \
  -e HF_HOME=/root/.cache/huggingface \
  --ipc=host --shm-size=8g \
  omnivoice-train:0.1 \
  bash -c '
    cd /workspace/OmniVoice
    pip install -e . --no-deps >/dev/null
    stage=1 stop_stage=1 bash examples/run_finetune_mychar.sh
  '
```

Expect:

```
Step  60 | loss: 3.86 | lr 4.86e-06 | grad_norm 3.10 | epoch 16 | 1.46 steps/sec
Step 200 | loss: 3.51 | lr 3.50e-06 | grad_norm 2.78 | epoch 51
Step 260 | loss: 3.40 | lr 1.50e-06 | grad_norm 4.21 | epoch 73 | 1.42 steps/sec
Step 400 | loss: 3.01 | lr 0.00e+00 | grad_norm 4.71 | epoch 113 | done
```

400 steps × ~1.3 s/step ≈ **8 minutes wallclock** on a 16 GB Blackwell.
Peak VRAM ~15 GB (fully utilized with single GPU).

Output:

```
exp/omnivoice_mychar_sft/
├── checkpoint-200/   ~2.45 GB (model.safetensors) + ~4.6 GB optimizer.bin
├── checkpoint-300/   same
├── checkpoint-400/   same   ← best (final loss ~3.0)
├── initial_config.json
├── tensorboard/
└── train.log
```

You can `rm` `optimizer.bin` from each checkpoint to save ~15 GB of
disk (it's only needed if you want to resume training).

### 6. Audio-tokenizer subdir gotcha

The trainer's checkpoint save does **not** include the
`audio_tokenizer/` subdir that vLLM-Omni needs at inference. Copy it
from the cached base model:

```bash
docker run --rm \
  -v hf-cache:/cache:ro \
  -v /path/to/checkpoint-400:/dst \
  busybox sh -c '
    src=/cache/hub/models--k2-fsa--OmniVoice/snapshots/*/audio_tokenizer
    mkdir -p /dst/audio_tokenizer
    cp -L "$src/config.json"              /dst/audio_tokenizer/
    cp -L "$src/model.safetensors"        /dst/audio_tokenizer/
    cp -L "$src/preprocessor_config.json" /dst/audio_tokenizer/
  '
```

`cp -L` is **required** — the cache files are symlinks to a blob store;
plain `cp` copies broken symlinks and the inference container crashes
with `HFValidationError: Repo id must be in the form 'repo_name'`.

### 7. Deploy the SFT'd checkpoint

Add the service to your `vllm-omni-deploy/docker-compose.yml`:

```yaml
services:
  omnivoice-mychar:
    image: vllm/vllm-omni:v0.20.0
    volumes:
      - hf-cache:/root/.cache/huggingface
      - ../OmniVoice/exp/omnivoice_mychar_sft/checkpoint-400:/models/omnivoice-mychar:ro
      - ./refs:/refs:ro
    ports:
      - "8000:8000"
    runtime: nvidia
    deploy: { resources: { reservations: { devices: [{ driver: nvidia, count: 1, capabilities: [gpu] }] } } }
    ipc: host
    shm_size: '8gb'
    command: >
      vllm serve /models/omnivoice-mychar
      --omni --host 0.0.0.0 --port 8000
      --gpu-memory-utilization 0.40
      --enforce-eager
      --max-model-len 2048 --max-num-seqs 1
      --allowed-local-media-path /refs
      --served-model-name omnivoice-mychar
```

Bring up:

```bash
docker compose up -d omnivoice-mychar
curl http://127.0.0.1:8000/v1/models  # → omnivoice-mychar
```

### 8. Verify content fidelity didn't regress

Run the same eval rig from [ch. 15](15-vllm-omni-model-selection.md):

```bash
cd vllm-omni-tests/
python run_eval.py --endpoint http://127.0.0.1:8000 \
  --whisper-model large-v3 --label "OmniVoice + per-character SFT"
```

Pass criteria: mean jaccard ≥ 0.95 across the 21-prompt battery.

A representative result on a 137-clip / ~20-min training set: mean
jaccard **0.96–0.97** across 3 reference clips (matches / slight-better
than base zero-shot). Worst single edge case (a casual greeting on a
mismatched-register ref) went from 0.68 → **1.00** after SFT — SFT
smoothed out the model's prior on rarely-prompted refs.

### 9. Subjective listen for timbre

Generate a wide 21-prompt × 3-ref sample battery:

```bash
python run_wide.py --endpoint http://127.0.0.1:8000 \
  --out out_wide_sft --label "OmniVoice + per-character SFT"
```

Open `out_wide_sft/by_prompt/<id>/<ref>.wav` and A/B against the same
prompt synthesized from base zero-shot (`out_wide/by_prompt/<id>/<ref>.wav`).
If the SFT version sounds closer to your target → ship.

## Common failure modes

| Symptom | Cause + fix |
|---|---|
| OOM in backward at step 1 | `attn_implementation: flex_attention` exhausts VRAM on Blackwell. Switch to `sdpa`. |
| OOM later (e.g. step 100) | Drop `batch_tokens` to 1024 + `gradient_accumulation_steps` to 8; drop `max_batch_size` to 4. |
| `CUBLAS_STATUS_INTERNAL_ERROR` in backward | OOM in disguise. Same fix as above. |
| Loss plateaus high (≥ 4.0) after 100+ steps | LR too low. Bump to `1e-5` and retry. |
| Loss collapses (≤ 1.5) very fast | LR too high — overfit risk. Drop to `2e-6`. |
| Inference produces silence or garbage post-SFT | `audio_tokenizer/` missing from the checkpoint dir. See step 6. |
| Post-SFT model starts hallucinating transcript-like phrases | Overfit (saw too many epochs of the same data). Lower `steps` to ~200. |

## Cost

400 steps × ~1.3 s/step = 8 min wallclock. ~15 GB peak VRAM. ~7.5 GB
checkpoint disk (kept top 3). No multi-GPU needed.

If you don't have a 16 GB+ consumer GPU: rent ~$0.50 of an A100 hour on
Lambda or RunPod and you'll finish before the hour bills.

## Why this works (in 3 sentences)

OmniVoice's 0.6 B Qwen3 backbone has already absorbed Japanese phonetic
patterns from 36k hours of pre-training. SFT on a single voice just
biases the next-token prior toward that voice's characteristic spectral
+ prosodic envelope. The audio decoder (Higgs codec) is frozen, so the
SFT can't damage acoustic quality — it only adjusts which codec tokens
the LM picks.

## See also

- [`docs/15-vllm-omni-docker.md`](15-vllm-omni-docker.md) — the deploy
  flow your checkpoint slots into.
- [`docs/15-vllm-omni-model-selection.md`](15-vllm-omni-model-selection.md)
  — why OmniVoice over CosyVoice3 / Qwen3-TTS for Japanese.
- [`docs/14-cross-lingual-limits.md`](14-cross-lingual-limits.md) —
  cross-lingual cloning failure modes (your JA-trained SFT model will
  still leak Japanese accent into English; expected).

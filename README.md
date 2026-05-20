# Open-Source TTS Guide (2026)

A hands-on guide to deploying production text-to-speech for a single
character voice. Covers picking a model from the 2026 open-source
landscape, running it under **vLLM-Omni in Docker**, fine-tuning the
voice when you need character-level fidelity, and wiring it to any
OpenAI-compatible client via one URL.

The guide pairs with a reference deploy repo (`vllm-omni-deploy`)
containing the canonical `docker-compose.yml` + Dockerfiles, and an
evaluation harness (`vllm-omni-tests`) for grading content fidelity
across a battery of prompts.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## TL;DR — the production pattern in 2026

| Layer | Choice | Why |
|---|---|---|
| Runtime | **vLLM-Omni in Docker** | OpenAI-compatible API, model-agnostic, one bind URL, ~14 TTS architectures supported out of the box. |
| Model (Japanese) | **OmniVoice** (`k2-fsa/OmniVoice`) | 36,914 hrs JA training, char-level Qwen3 tokenizer (no kanji-byte-fallback trap), FLEURS JA CER 5.96. Best of the vLLM-Omni-native set for JA. |
| Voice fidelity | **Per-character SFT on ~10–20 min audio** | Lifts timbre clearly above zero-shot. ~8 minute training run on a 16 GB Blackwell GPU. |
| Footprint | **~7 GB system VRAM** | Fits a 16 GB consumer GPU with a desktop OS sharing the device. |

For non-Japanese: pick the model from
[`docs/15-vllm-omni-model-selection.md`](docs/15-vllm-omni-model-selection.md);
same deploy pattern.

## Quickstart — production deploy in 10 minutes

You need: Docker + NVIDIA Container Toolkit + WSL2 (on Windows) + a
GPU with ≥ 8 GB VRAM.

```bash
# 1. Clone the deploy repo (docker-compose + reference clips).
git clone https://github.com/<your-deploy-fork>/vllm-omni-deploy
cd vllm-omni-deploy

# 2. Bring up the default service (base OmniVoice — zero-shot from a
#    reference clip you supply).
docker compose up -d

# 3. Verify.
curl http://127.0.0.1:8000/v1/models
#   → { ..., "data": [ { "id": "k2-fsa/OmniVoice", ... } ] }
```

Now any OpenAI-compatible client can synthesize:

```bash
curl -X POST http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d "$(cat <<'JSON'
{
  "model":     "k2-fsa/OmniVoice",
  "input":     "今日もよろしくお願いします。",
  "language":  "Japanese",
  "ref_audio": "data:audio/wav;base64,<base64 of your reference clip>",
  "ref_text":  "<transcript of the reference clip>"
}
JSON
)" \
  -o hello.wav
```

Full deploy walkthrough: [`docs/15-vllm-omni-docker.md`](docs/15-vllm-omni-docker.md).

## Fine-tuning a character voice

When the base model's zero-shot timbre isn't close enough to your
target — common for distinctive character voices — fine-tune the
LM head on your reference data. Recipe:
[`docs/16-omnivoice-sft-recipe.md`](docs/16-omnivoice-sft-recipe.md).

Per-step time on a 16 GB Blackwell GPU (single GPU):

| Step | Wallclock | Output |
|---|---|---|
| Slice + ASR + transcript clean | ~30 min | ~100–200 clips + manifest |
| Audio tokenization (Higgs codec) | ~30 s | WebDataset shards |
| Full-FT 400 steps (bf16 SDPA) | **~8 min** | `checkpoint-400/` (~2.3 GB) |
| Smoke-test against eval rig | ~5 min | jaccard ≥ 0.96 on 21 prompts |
| **Total** | **~50 min** | Ship-ready character voice |

## What this guide answers

| Question | Where to look |
|---|---|
| "How do I deploy production TTS today?" | [`docs/15-vllm-omni-docker.md`](docs/15-vllm-omni-docker.md) |
| "Which model fits my language + VRAM?" | [`docs/15-vllm-omni-model-selection.md`](docs/15-vllm-omni-model-selection.md), [`docs/per-language/`](docs/per-language/) |
| "How do I fine-tune a custom character voice?" | [`docs/16-omnivoice-sft-recipe.md`](docs/16-omnivoice-sft-recipe.md) |
| "Why not Qwen3-TTS / CosyVoice / Higgs / Fish-Speech?" | [`docs/15-vllm-omni-model-selection.md`](docs/15-vllm-omni-model-selection.md) — empirical eval with concrete failure modes |
| "What's the wire contract my client should speak?" | [`docs/15-vllm-omni-docker.md`](docs/15-vllm-omni-docker.md) §"Sending a synth request" |
| "What goes wrong on cross-lingual cloning?" | [`docs/14-cross-lingual-limits.md`](docs/14-cross-lingual-limits.md) |
| "Can I still do GPT-SoVITS fine-tuning?" | [Legacy path](#legacy-path-gpt-sovits-fine-tune) — yes, in [`scripts/sovits-finetune/`](scripts/sovits-finetune/). |

## Picking a different model

vLLM-Omni supports a broad model family — VoxCPM2 (30 langs), CosyVoice3
(multilingual), Qwen3-TTS, Voxtral-TTS, MossTTS-Nano, and others. They
all plug into the same `docker compose --profile <name> up` flow with
your client only changing one config field (the served model id).

| Profile | Model | When to pick |
|---|---|---|
| _default_ | `k2-fsa/OmniVoice` | Japanese voice cloning (best of the JA-trained models). |
| `--profile cosy3` | `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` | Chinese (CER 0.81% native ZH). JA mediocre per our eval. |
| `--profile qwen` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | Multilingual baseline (JA/ZH/EN), Apache-2.0. |
| `--profile fish` | `fishaudio/s2-pro` | JA SOTA on paper (100k hrs) — currently blocked by upstream import bug. |

See [`docs/15-vllm-omni-model-selection.md`](docs/15-vllm-omni-model-selection.md)
for the full per-model eval and rationale.

## Per-language picks

The deep-dive pages survive any individual model pivot (they're
architecture surveys, not snapshots):

| Language | Top pick (2026-05) | Notes |
|---|---|---|
| **Japanese** | OmniVoice + per-character SFT | 36k-hr JA pretrain + ~20-min character SFT. |
| **Chinese** | CosyVoice3 | Apache, CER 0.81% native ZH. |
| **English** | Open — OmniVoice EN under-tested, Higgs Audio over the 16 GB budget. |
| **Multilingual (1 model, all 3)** | OmniVoice for JA/ZH; VoxCPM2 if you need 30-lang reach. |

Detailed pages: [`docs/per-language/japanese.md`](docs/per-language/japanese.md),
[`docs/per-language/chinese.md`](docs/per-language/chinese.md),
[`docs/per-language/english.md`](docs/per-language/english.md).

## Documentation map

### Production deploy (start here)
- [**`docs/15-vllm-omni-docker.md`**](docs/15-vllm-omni-docker.md) — Docker compose walkthrough; the canonical deploy path.
- [**`docs/15-vllm-omni-model-selection.md`**](docs/15-vllm-omni-model-selection.md) — Which vLLM-Omni-native model to pick; full empirical eval.
- [**`docs/16-omnivoice-sft-recipe.md`**](docs/16-omnivoice-sft-recipe.md) — Fine-tune OmniVoice on ~20 min of your character voice.
- [`docs/14-cross-lingual-limits.md`](docs/14-cross-lingual-limits.md) — Empirical cross-lingual failure modes (JA→EN accent leak etc.).

### Per-language picks
- [`docs/per-language/japanese.md`](docs/per-language/japanese.md)
- [`docs/per-language/chinese.md`](docs/per-language/chinese.md)
- [`docs/per-language/english.md`](docs/per-language/english.md)
- [`docs/per-language/multilingual.md`](docs/per-language/multilingual.md)

### Model deep dives
- [`docs/models/`](docs/models/) — per-model deep dives (Style-Bert-VITS2, CosyVoice3, Higgs, Qwen3-TTS, GPT-SoVITS v4).

### Reference
- [`docs/00-landscape-2026.md`](docs/00-landscape-2026.md) — Architecture taxonomy + license matrix.
- [`docs/01-theory.md`](docs/01-theory.md) — TTS theory (transfer learning, two-stage design).
- [`docs/02-comparison.md`](docs/02-comparison.md) — Cross-model decision tree.
- [`docs/07-windows-guide.md`](docs/07-windows-guide.md) — Windows / Blackwell / cu128 gotchas.
- [`docs/13-inference-optimization.md`](docs/13-inference-optimization.md) — Kernel-level inference optimization (pre-vLLM-Omni era; tactics still apply).

## Legacy path — GPT-SoVITS fine-tune

The pre-vLLM-Omni recipe. Still works; kept for users who specifically
want LoRA-on-GPT-SoVITS-v4 (e.g. you already have a v4 weights set, or
you want the SoVITS-specific dataset pipeline).

```bash
cd scripts/sovits-finetune/
python demucs_isolate.py      --input video_audio.wav --output speaker_vocals.wav
python 01_slice_audio.py      --vocals ../../speaker_vocals.wav --exp my_speaker
python 02_asr_transcribe.py   --exp my_speaker --lang ja
python 03_extract_features.py --exp my_speaker
python 04_extract_semantic.py --exp my_speaker
python 05_train_sovits_v4.py  --exp my_speaker --epochs 20 --lora-rank 32
python 06_train_gpt.py        --exp my_speaker --epochs 15 --pretrained-version v4
python 07_inference_v4.py     --exp my_speaker --lang ja --text "..." ...
```

Full tutorial: [`docs/models/gpt-sovits-v4.md`](docs/models/gpt-sovits-v4.md)
+ [`scripts/sovits-finetune/README.md`](scripts/sovits-finetune/README.md).
For new projects: **prefer the OmniVoice SFT path** above — same data
requirement (~20 min), shorter wallclock, drops straight into the
production Docker compose.

## Validated on

| Track | Hardware | OS | GPU mem | Versions |
|---|---|---|---|---|
| OmniVoice base + SFT | RTX 5080 (Blackwell, sm_120) | Windows 11 + WSL2 + Docker Desktop | 16 GB | vllm/vllm-omni:v0.20.0, torch 2.11+cu130, k2-fsa/OmniVoice |
| GPT-SoVITS v4 (legacy) | RTX 5080 | Windows 11 native | 16 GB | GPT-SoVITS v4 (LoRA), pretrained s2Gv4 |

## How to contribute

1. **A new model page**: copy any existing `docs/models/<model>.md` as a
   template, fill in license + benchmarks + VRAM + RTF + failure
   modes, open a PR.
2. **A new language pick**: add `docs/per-language/<lang>.md`, link
   from the README.
3. **A new vLLM-Omni profile**: add a service to the `vllm-omni-deploy`
   docker-compose, and append an entry to the
   [picking-a-model](docs/15-vllm-omni-model-selection.md) table here.

## Project history

- **2024–2025**: started as a GPT-SoVITS v4 fine-tuning tutorial.
- **2026-Q1**: expanded with Qwen3-TTS zero-shot + per-language picks;
  production stack was per-engine Python sidecars (Style-Bert-VITS2 for
  JA, Chatterbox-MTL for ZH/EN).
- **2026-05**: pivoted to **vLLM-Omni Docker**. Evaluated every
  vLLM-Omni-native model on a 21-prompt JA battery, picked OmniVoice
  as the production base, documented the SFT recipe. This README leads
  with the new flow; legacy paths kept where still useful.

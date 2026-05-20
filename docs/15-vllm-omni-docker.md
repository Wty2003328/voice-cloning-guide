# 15 — vLLM-Omni Docker: the production deploy

This is the recommended way to run open-source TTS in 2026. One Docker
container, one OpenAI-compatible URL, swap the model by changing one
config line.

## Why Docker + vLLM-Omni

The pre-2026 pattern was a per-engine Python sidecar: one process for
GPT-SoVITS, another for Style-Bert-VITS2, another for CosyVoice — each
with its own conda env, its own install gotchas, its own healthz route,
its own request format. Your client had to know which engine was on
the other end to format the body correctly.

**vLLM-Omni in Docker** consolidates all that:

- One container image (`vllm/vllm-omni:v0.20.0`) supports ~14 TTS model
  architectures out of the box: OmniVoice, CosyVoice3, Qwen3-TTS,
  VoxCPM2, Voxtral-TTS, Fish-Speech, MOSS-TTS-Nano, Qwen2.5/3-Omni, etc.
- Every model is served through the same OpenAI-compatible
  `POST /v1/audio/speech` endpoint.
- No per-engine Python env on the host; no Windows-specific CUDA build
  for every engine. The image ships torch 2.11 + cu130 with sm_120
  (Blackwell) support baked in.
- Switching models is a `docker compose --profile <name> up` away.

## Prerequisites

| Requirement | How |
|---|---|
| Docker Desktop | Default Linux containers on Windows / macOS / Linux. |
| NVIDIA Container Toolkit | `apt install nvidia-container-toolkit` (Linux) or comes with Docker Desktop on Windows. |
| WSL2 + GPU passthrough (Windows only) | Docker Desktop → Settings → General → "Use WSL 2 based engine"; verify with `docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi`. |
| ≥ 8 GB GPU memory | Tighter for 7B+ models like Qwen2.5-Omni-7B; OmniVoice fits in ~3 GB container delta. |
| ≥ 30 GB free disk | Image (~32 GB pulled + extracted) + HF model cache (~5-10 GB per loaded model). |

## The deploy repo

A reference deploy repo lives alongside this guide:
[`vllm-omni-deploy`](https://github.com/<your-deploy-fork>/vllm-omni-deploy).
You can use it as-is or copy the `docker-compose.yml` into your own
project.

Layout:

```
vllm-omni-deploy/
  docker-compose.yml      # the one file you usually edit
  Dockerfile.omnivoice-train  # extends base image with SFT deps
  Dockerfile.cosy-ja      # extends base image with the kanji-frontend
                          # patch for the CosyVoice3 service (see ch. 15-selection)
  refs/                   # bundled reference audio clips (your character)
  out/                    # request body templates for manual curl tests
  README.md
```

Clone + bring up:

```bash
git clone https://github.com/<your-user>/vllm-omni-deploy
cd vllm-omni-deploy
docker compose up -d
```

Default service is the base `k2-fsa/OmniVoice` zero-shot model. First
boot downloads the weights (~3 GB) into the `hf-cache` named volume;
subsequent boots are ~10s. Add an `omnivoice-mychar` service later if
you fine-tune your own character voice — see
[ch. 16](16-omnivoice-sft-recipe.md).

## Verifying it works

```bash
# 1. Container running?
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
#   omnivoice   Up 15 minutes   0.0.0.0:8000->8000/tcp

# 2. Model loaded?
curl -s http://127.0.0.1:8000/v1/models | python -m json.tool
#   { "data": [ { "id": "omnivoice", ... } ] }

# 3. VRAM footprint
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
#   7174 MiB, 16303 MiB   ← Windows desktop (~4 GB) + container (~3 GB)
```

## Sending a synth request

The request body is OpenAI-compatible plus three extension fields
(`language`, `ref_audio`, `ref_text`):

```json
{
  "model":     "k2-fsa/OmniVoice",
  "input":     "今日もよろしくお願いします。",
  "language":  "Japanese",
  "speed":     1.0,
  "ref_audio": "data:audio/wav;base64,<...>",
  "ref_text":  "<transcript of the reference clip>"
}
```

Wire-contract gotchas:

1. **`ref_audio` must be a base64 data URL.** OmniVoice's media
   connector rejects `file://` paths even when the container is
   launched with `--allowed-local-media-path /refs`. Always send
   `data:audio/wav;base64,...`.
2. **`language` is the English name**, not BCP-47. Use "Japanese",
   "English", "Chinese". Wrap a short BCP-47 → English-name mapping in
   your client if you prefer to configure with codes like "ja".
3. **`speed = 1.0` should be elided** to keep the body minimal. Send
   `speed` only when it ≠ 1.0.
4. **Response is raw WAV bytes** (Content-Type: `audio/wav`), 24 kHz
   mono for OmniVoice. Errors come back as JSON `{ "error": ... }`.

A Python client that handles all of this is at
[`scripts/zero_shot_clone.py`](../scripts/zero_shot_clone.py) (use it
as a reference if you're writing your own client).

## Swapping the model

The compose has multiple services behind Docker profiles:

```yaml
services:
  omnivoice:                # default — base, zero-shot from ref_audio
    image: vllm/vllm-omni:v0.20.0
    command: vllm serve k2-fsa/OmniVoice --omni --host 0.0.0.0 --port 8000 ...

  # Optional: a fine-tuned character voice on top of the base.
  # Mount your checkpoint dir and serve it by path.
  # omnivoice-mychar:
  #   profiles: ["mychar"]
  #   image: vllm/vllm-omni:v0.20.0
  #   volumes:
  #     - ../OmniVoice/exp/mychar_sft/checkpoint-400:/models/omnivoice-mychar:ro
  #   command: vllm serve /models/omnivoice-mychar ...

  cosyvoice3:               # multilingual; needs kanji-frontend patch for JA
    profiles: ["cosy3"]
    image: vllm-omni-cosy-ja:v0.20.0
    command: vllm serve FunAudioLLM/Fun-CosyVoice3-0.5B-2512 ...

  qwen3-tts:                # multilingual baseline
    profiles: ["qwen"]
    command: vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-Base ...
```

Switch with:

```bash
docker compose down                          # stop current
docker compose --profile cosy3 up -d         # bring up CosyVoice3
```

See [Picking a model](15-vllm-omni-model-selection.md) for the full
empirical eval that drove these defaults.

## Performance tuning

Default flags in the deploy compose are tuned for an RTX 5080 (16 GB
VRAM with Windows desktop sharing the GPU):

```
--gpu-memory-utilization 0.40   # 40% × 16 GB = 6.4 GB ceiling
--enforce-eager                  # skip torch.compile, faster cold start
--max-model-len 2048             # OmniVoice utterances rarely exceed this
--max-num-seqs 1                 # single concurrent request
--allowed-local-media-path /refs # required for file:// (unused on OmniVoice)
```

For a dedicated GPU machine (no desktop sharing), raise
`--gpu-memory-utilization` to ~0.85 and `--max-num-seqs` to ~8 for
batched throughput.

## Stopping / restarting

```bash
docker compose down              # stop + remove containers
docker compose down -v           # also wipe hf-cache volume (forces re-download)
docker compose logs -f           # stream logs
docker compose exec omnivoice nvidia-smi  # GPU usage from inside container
```

The application never spawns or supervises the container — `docker
compose up` is how you start TTS, full stop.

## Troubleshooting

| Symptom | Likely cause + fix |
|---|---|
| `Cannot load local files without --allowed-local-media-path` | OmniVoice ignores this flag for its connector path. Send `ref_audio` as a `data:audio/wav;base64,...` URL instead. |
| `Repo id must be in the form 'repo_name'` at boot | Your local checkpoint dir is missing `audio_tokenizer/`. Copy it via `cp -L` from the base HF snapshot — see [ch. 16](16-omnivoice-sft-recipe.md#audio-tokenizer-quirk). |
| HTTP 500 + `EngineCore encountered an issue` mid-request | Out-of-memory in backward (training) or runtime. Drop `--gpu-memory-utilization` or `--max-num-seqs`. |
| `ImportError: cannot import name 'split_routed_experts'` at boot | You're on `v0.21.0rc1` / `latest`; vllm-omni HEAD imports a name removed from upstream vllm. **Pin `v0.20.0`** until upstream issue #3663 closes. |
| `ModuleNotFoundError: No module named 'fish_speech'` at boot for Fish-Speech S2 Pro | Known upstream bug in v0.20.0 (issue #2404). No workaround as of 2026-05; use a different model. |

More gotchas in [`docs/07-windows-guide.md`](07-windows-guide.md).

## See also

- [`docs/15-vllm-omni-model-selection.md`](15-vllm-omni-model-selection.md) — Why OmniVoice; what we evaluated and rejected.
- [`docs/16-omnivoice-sft-recipe.md`](16-omnivoice-sft-recipe.md) — Fine-tune your own character voice on top of the base.
- [`docs/14-cross-lingual-limits.md`](14-cross-lingual-limits.md) — Cross-lingual cloning failure modes empirically measured.

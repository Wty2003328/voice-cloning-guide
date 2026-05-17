# 12 — Integration via the Universal TTS Spec

Both the zero-shot path (Qwen3-TTS) and the fine-tune path (GPT-SoVITS)
expose the same HTTP contract — the **OpenAI-compatible TTS Provider
Spec v1** — so the model you trained drops into any chat app /
VTuber rig / audiobook generator / companion that speaks the spec.

This decouples the *voice* from the *app*: you can swap Qwen3-TTS for
GPT-SoVITS / OpenAI cloud TTS / kokoro-fastapi without touching the app
code, and you can switch the app you're integrating into without
retraining anything.

## The spec in one paragraph

A TTS provider is an HTTP server with four routes:

| Route | Purpose |
|-------|---------|
| `POST /v1/audio/speech` | Synthesize speech (OpenAI-compatible body) |
| `GET  /v1/audio/voices` | List registered voices |
| `POST /v1/audio/voices/clone` | Register a reference audio → voice_id (multipart) |
| `GET  /healthz` | Liveness probe |

Body schema for `/v1/audio/speech`:

```json
{
  "model":           "qwen3-tts-1.7b",       // optional, engine identifier
  "input":           "Hello, world.",
  "voice":           "asuna",                // pre-registered voice_id
  "response_format": "wav",                  // wav | mp3 | opus | pcm
  "speed":           1.0,
  "stream_format":   "audio",                // or "sse" for chunked
  "x_companion": {                           // extension fields
    "language":   "ja",                      // BCP-47
    "quality":    "balanced",                // fast | balanced | high
    "reference_id": null,                    // alt to "voice" for ad-hoc
    "advanced":   {}                         // engine-specific overrides
  }
}
```

Response is raw audio bytes by default, or SSE `audio.chunk` events
when `stream_format: "sse"`.

The `x_companion` extension is namespaced so unknown fields are
ignored by real OpenAI servers — your requests stay forward-compatible
with the cloud OpenAI TTS API.

For the full schema (including SSE event format, error shapes,
multipart upload schema), see the source spec:
**[zeroclaw-companion/docs/TTS-PROVIDER-SPEC.md](https://github.com/Wty2003328/waifu-companion/blob/main/docs/TTS-PROVIDER-SPEC.md)**.

## Reference sidecar implementations

Two ready-to-run sidecars ship alongside the companion (linked above):

### Qwen3-TTS sidecar (`qwen3_tts_sidecar.py`)

FastAPI server wrapping the model from this guide's Path A. ~180 LOC.
Reads voice registry from a `voices.toml`, supports streaming, voice
cloning at runtime via the `/clone` endpoint. Launch:

```bash
export TTS_PORT=9890
export TTS_MODEL_DIR=/path/to/qwen3-tts-1.7b-base
export TTS_VOICES_CONFIG=/path/to/voices.toml
export TTS_ATTN_IMPL=sdpa
export TTS_DTYPE=bf16
python qwen3_tts_sidecar.py
```

The `voices.toml` registers voices at startup:

```toml
[[voice]]
id              = "asuna"
name            = "Asuna"
language        = "ja"
reference_audio = "/path/to/asuna_concat_diverse5.wav"
reference_text  = "今日はあの…"
```

### OpenAI proxy sidecar (`openai_proxy_sidecar.py`)

Forwards `/v1/audio/speech` requests to `api.openai.com`. Useful as a
no-GPU fallback or for testing the abstraction. Same client code,
different backend. Launch:

```bash
export TTS_PORT=9890
export OPENAI_API_KEY=sk-...
python openai_proxy_sidecar.py
```

The `voice_id` becomes one of OpenAI's six preset voices
(alloy / echo / fable / onyx / nova / shimmer). `/v1/audio/voices/clone`
returns 501 (OpenAI doesn't support reference cloning).

## Wiring a sidecar from this guide

If you trained a GPT-SoVITS voice (Path B) or want to use a Qwen3-TTS
voice from a custom reference clip, here's the integration pattern:

### Path A integration (Qwen3-TTS — zero-shot)

1. Pick or build your reference WAV (per [10-zero-shot-cloning.md](10-zero-shot-cloning.md)).
2. Write its transcript.
3. Add a `[[voice]]` entry to `voices.toml`:
   ```toml
   [[voice]]
   id              = "my_character"
   name            = "My Character"
   language        = "ja"
   reference_audio = "./reference_clips/my_character.wav"
   reference_text  = "<exact transcript>"
   ```
4. Launch `qwen3_tts_sidecar.py` (per above).
5. Point your chat app's TTS URL at `http://127.0.0.1:9890`.

That's it — no code changes in the app.

### Path B integration (GPT-SoVITS — fine-tuned)

GPT-SoVITS pre-dates the universal spec. The classic launcher in
[zeroclaw-companion/tools/avatar/gptsovits_tts_server.py](https://github.com/Wty2003328/waifu-companion/blob/main/tools/avatar/gptsovits_tts_server.py)
serves the legacy `POST /tts` API, which spec-compliant clients also
accept as a compatibility alias. So existing fine-tuned voices keep
working.

If you want a fine-tuned GPT-SoVITS voice to serve the new spec
natively, write a thin FastAPI wrapper that:

1. Boots GPT-SoVITS inference (`07_inference_v4.py`-style)
2. Exposes the 4 spec endpoints
3. Maps `x_companion.quality` to its native knobs:
   ```python
   QUALITY_PRESETS = {
       "fast":     {"cfm_sample_steps": 8},
       "balanced": {"cfm_sample_steps": 16},   # default
       "high":     {"cfm_sample_steps": 32},
   }
   ```

The Qwen3-TTS sidecar at `qwen3_tts_sidecar.py` is a copy-paste-ready
template for this. ~200 LOC total.

## Client side — minimal example

```python
import requests

# Synthesize
resp = requests.post("http://127.0.0.1:9890/v1/audio/speech", json={
    "input": "こんにちは、私は人工知能アシスタントです。",
    "voice": "my_character",
    "response_format": "wav",
    "x_companion": {"language": "ja", "quality": "balanced"},
}, timeout=60)

with open("out.wav", "wb") as f:
    f.write(resp.content)

# Streaming variant (SSE)
import json
with requests.post("http://127.0.0.1:9890/v1/audio/speech", json={
    "input": "Hello. This will arrive in chunks. Goodbye.",
    "voice": "my_character",
    "stream_format": "sse",
    "x_companion": {"language": "en"},
}, stream=True) as r:
    for line in r.iter_lines():
        if line.startswith(b"data: "):
            evt = json.loads(line[6:])
            # evt["audio"] is base64-encoded WAV chunk
            # ... decode + queue for playback ...
```

## Why an HTTP sidecar (vs in-process inference)

- **Process isolation**: TTS models are heavy. A model crash shouldn't
  take the host app with it.
- **Language-independence**: app code can be Rust, Go, TypeScript, etc.
  — the sidecar is Python where the ML lives.
- **Hot-swap**: change `voice` config and restart the sidecar without
  rebuilding the app.
- **Distributed**: the sidecar can run on a different machine with a
  GPU. App stays on the user's laptop.
- **Spec-compliance**: any future / alternative model just needs to
  speak the spec — no code branching by engine type.

## When NOT to use a sidecar

- **One-shot batch synthesis** (e.g., generating an audiobook
  overnight) — call `Qwen3TTSModel` directly in Python.
- **Real-time interactive demos** where you want zero HTTP overhead —
  same.

For *anything* that has a chat-like turn loop with a separate frontend
(chatbot, VTuber rig, game companion, voice assistant), the sidecar
pattern wins.

## Where to learn more

- **Full spec:** [TTS-PROVIDER-SPEC.md](https://github.com/Wty2003328/waifu-companion/blob/main/docs/TTS-PROVIDER-SPEC.md)
- **Working sidecars:** [qwen3_tts_sidecar.py](https://github.com/Wty2003328/waifu-companion/blob/main/tools/avatar/qwen3_tts_sidecar.py),
  [openai_proxy_sidecar.py](https://github.com/Wty2003328/waifu-companion/blob/main/tools/avatar/openai_proxy_sidecar.py)
- **Production-ready client (Rust):** [tts_server.rs](https://github.com/Wty2003328/waifu-companion/blob/main/crates/companion-avatar/src/tts_server.rs)
  — the companion's universal TTS client; ~150 LOC plus reqwest.
- **Survey of multilingual TTS models:** [TTS-MULTILINGUAL-GUIDE.md](https://github.com/Wty2003328/waifu-companion/blob/main/docs/TTS-MULTILINGUAL-GUIDE.md)

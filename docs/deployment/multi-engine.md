# Multi-Engine TTS Architecture — Design

**Status:** design draft, awaiting per-language model selection (task #141 research)

## Goal

Replace the single Qwen3-TTS engine with **N language-specialized
engines** routed by `x_companion.language`. Each engine is independently
deployed, configured, and version-pinned. The companion (Rust side)
sees ONE unchanged contract — `/v1/audio/speech` with the language hint
— and a router transparently directs the call.

## Why multi-engine

A single model that handles JA + ZH + EN + KO compromises on every
language. Per [13-inference-optimization](../gpt-sovits-voice-cloning-guide/docs/13-inference-optimization.md)
work and the per-language audit (task #139), Qwen3-TTS is mediocre on
JA (digit runaway, EOS misfires, pitch-accent errors). Native-JA models
like Style-Bert-VITS2 do much better — but only on JA. Trying to make
one model cover everything is a losing battle when single-language
specialists exist with permissive licenses.

## Architecture options (decision matrix)

### Option A — One process per engine, router in front

```
companion-server (Rust)
       │  POST /v1/audio/speech  {language: "ja", ...}
       ▼
tts-router:9890 (Python, lightweight FastAPI)
       │  routes by language
       ├──→ tts-ja:9891   (Style-Bert-VITS2 process)
       ├──→ tts-zh:9892   (CosyVoice 3 process)
       └──→ tts-en:9893   (Kokoro process)
```

**Pros**
- Each engine has its own Python env / deps — Style-Bert-VITS2's
  pyopenjtalk version doesn't fight Kokoro's misaki, etc.
- Independent failure: ZH engine crashing doesn't kill JA
- Can hot-swap a single engine without restarting the whole stack
- Lazy loading: only start engines for languages actually used
- Matches the existing TTS-Provider-Spec v1 (router serves the contract;
  each engine also serves it internally)

**Cons**
- 3-4 Python processes alive: ~6-12 GB VRAM if all loaded eagerly
- More moving parts to monitor
- Cross-process IPC overhead per call (~5-20ms)
- Router becomes a SPoF

### Option B — One fat process loading all engines

```
companion-server (Rust)
       │  POST /v1/audio/speech
       ▼
tts-sidecar:9890 (Python, all engines in one process)
       │  if-language-switch dispatcher
       ├── jp_engine.synthesize(text)
       ├── zh_engine.synthesize(text)
       └── en_engine.synthesize(text)
```

**Pros**
- Single Python env, single process, single GPU context
- Shared model loading code (faster-whisper for ASR, fade-in helpers)
- Lower IPC overhead

**Cons**
- Dependency conflicts: Style-Bert-VITS2 needs Pydantic v1.x, CosyVoice
  needs v2.x, Kokoro is fine with either — packing them in one env is
  fragile
- All-or-nothing failure: one engine's bad import kills the sidecar
- Hot-swap requires full sidecar restart (~30-60s warmup loss)
- Total VRAM = sum of all models loaded simultaneously

### Option C — Lazy single-engine with hot-swap

```
tts-sidecar:9890
  active_engine: JA | ZH | EN
  on language change → unload current → load requested
```

**Pros**
- Minimum VRAM: only one model resident
- Single process

**Cons**
- 30-60s cold-swap on every language change (unacceptable for chat)
- Voice clone consistency lost — each cold-load re-initializes voice prompts
- Only viable if language changes are rare

**Recommendation: Option A.** The deployment complexity is real but
the isolation and per-engine versioning benefits are decisive. VRAM
budget on RTX 5080 (16 GB) easily accommodates 3 small-medium models
loaded eagerly; if tight, use lazy-start (router spawns engine on
first request, keeps warm).

## Wire contract (unchanged)

The router and every engine speak the existing **TTS Provider Spec v1**
(see `zeroclaw-companion/docs/TTS-PROVIDER-SPEC.md`). Companion-server
talks ONLY to the router; engines are an internal implementation
detail. The contract is:

```
POST /v1/audio/speech         Content-Type: application/json
    {
      "input":           "<utterance>",
      "voice":           "<voice_id>",
      "speed":           1.0,
      "response_format": "wav",
      "stream_format":   "audio",
      "x_companion": {
        "language": "ja",           // ← router uses this to dispatch
        "quality":  "balanced",
        "advanced": { ... }         // ← engine-specific
      }
    }
```

The router:
1. Reads `x_companion.language` (defaults to "ja" if absent)
2. Looks up the engine for that language
3. Forwards the request to the engine's port, returns its response
4. On engine failure: returns 502 with engine identity in the response body

## Router implementation sketch

```python
# tts_router.py — ~150 LOC
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import Response
import httpx
import os

ENGINE_REGISTRY = {
    "ja": int(os.environ.get("TTS_JA_PORT", "9891")),
    "zh": int(os.environ.get("TTS_ZH_PORT", "9892")),
    "en": int(os.environ.get("TTS_EN_PORT", "9893")),
}
DEFAULT_LANG = "ja"

app = FastAPI(title="TTS Router")
client = httpx.AsyncClient(timeout=180.0)

@app.get("/healthz")
async def healthz():
    # Aggregate per-engine health
    statuses = {}
    for lang, port in ENGINE_REGISTRY.items():
        try:
            r = await client.get(f"http://127.0.0.1:{port}/healthz", timeout=2.0)
            statuses[lang] = {"port": port, "ok": r.status_code == 200,
                              "info": r.json()}
        except Exception as e:
            statuses[lang] = {"port": port, "ok": False, "error": str(e)}
    any_ok = any(s["ok"] for s in statuses.values())
    return {"status": "ok" if any_ok else "error",
            "engines": statuses}

@app.post("/v1/audio/speech")
async def speech(request: Request):
    body = await request.json()
    lang = (body.get("x_companion", {}) or {}).get("language", DEFAULT_LANG)
    port = ENGINE_REGISTRY.get(lang)
    if port is None:
        raise HTTPException(400, f"no engine for language {lang!r}")
    try:
        r = await client.post(
            f"http://127.0.0.1:{port}/v1/audio/speech",
            json=body, timeout=180.0,
        )
        return Response(content=r.content, media_type=r.headers.get("content-type"),
                        headers={k: v for k, v in r.headers.items()
                                  if k.lower() in ("x-sample-rate", "x-channels", "x-format")})
    except httpx.ConnectError:
        raise HTTPException(502, f"engine ja-{port} unreachable")
```

The router is intentionally dumb — pure dispatch. All per-engine logic
(text normalization, fade-in, ASR validation, sampling params) lives
inside the engine processes.

## Per-engine sidecar interface

Every engine implements the same `/v1/audio/speech` contract. The
existing `qwen3_tts_sidecar.py` is already this shape — keep it as the
JA fallback / generic-multi-language fallback. New engines (e.g.
`style_bert_vits2_sidecar.py`) follow the same pattern.

### Engine selection logic (in router config)

The user picks which engine serves each language. Config:

```toml
# tts_router.toml
[engines.ja]
binary = "style_bert_vits2_sidecar"
port = 9891
voice = "target"

[engines.zh]
binary = "cosyvoice3_sidecar"
port = 9892
voice = "target"

[engines.en]
binary = "kokoro_sidecar"
port = 9893
voice = "target"
```

A `tts-router-supervisor` starts the engines defined here, monitors
them, restarts on crash.

## Voice consistency across languages

**The hard problem:** the user wants the same character voice in JA AND ZH AND
EN. Different engines have different speaker encoders — the SAME
reference audio produces different embeddings in each model.

Options:

1. **One reference clip per (engine, language)** — register e.g. `mychar_ja`
   in the JA engine, `mychar_zh` in the ZH engine, etc. Each engine
   independently produces its best approximation of the character. Voice
   consistency is imperfect but acceptable; the user already accepts
   some drift in cross-lingual cloning.

2. **Train per-engine voice models from the SAME clip pool** — for
   per-voice-trained engines (Style-Bert-VITS2), train a custom
   from the diverse5 clips in each engine separately. Higher consistency
   within each engine but engines still differ from each other.

3. **Speaker-embedding bridge** — train a tiny adapter that maps one
   engine's speaker embedding to another's. Out of scope; too complex
   for the value.

**Recommendation: (1) one reference clip per (engine, language).** This
is the path of least resistance and matches what the user already
implicitly accepts (the diverse5 ref was selected via per-language A/B
listening anyway).

## Migration plan

1. **Phase 1 — research per-language SOTA** (task #141, in progress)
2. **Phase 2 — guide chapter** ("Specialized models per language")
3. **Phase 3 — prototype JA engine** (Style-Bert-VITS2 likely)
   - Stand up `style_bert_vits2_sidecar.py` speaking the v1 spec
   - Compare quality to Qwen3-TTS baseline via the `test_tts_audio_quality`
     rig
   - Decision gate: if JA quality clearly better, proceed; else iterate
4. **Phase 4 — router process** (`tts_router.py` + supervisor)
   - Companion-server's TTS URL points at router (default 9890)
   - Router dispatches JA→Style-Bert-VITS2, ZH/EN→Qwen3-TTS as fallback
5. **Phase 5 — add ZH engine** (CosyVoice 3 likely, possibly via WSL2)
6. **Phase 6 — add EN engine** (Kokoro or CosyVoice 3)

Each phase is independently shippable. Companion-server doesn't need to
change — it talks to the same v1 spec.

## VRAM budget

RTX 5080 = 16 GB total. Need to fit (in approximate order of weight):
- Style-Bert-VITS2 JA: ~1 GB
- CosyVoice 3 ZH: ~2 GB
- Kokoro EN: ~1 GB
- ASR validator (faster-whisper small): ~600 MB
- Working memory + KV caches: ~3-4 GB
- **Total budget**: ~8-9 GB ← comfortably fits

If we need to add the qwen3-tts fallback engine alongside, that's ~5 GB
on top → 13-14 GB total. Still fits but no room for expansion.

## Failure modes + recovery

- Router unreachable: companion-server's existing `/healthz` polling
  reports `tts_up=false`, frontend shows degraded state
- Engine for requested language down: router returns 502; companion
  can fall back to its own degraded path (text-only reply, no audio)
- All engines down: same as today's qwen3-tts sidecar being down
- Engine returns bad audio (runaway): the per-engine debug-capture
  (already shipped in qwen3_engine.py) catches it; we have the
  failing input for replay

## Why this is the right shape

- **Match the right model to each language** — no compromise
- **Independent versioning** — upgrade CosyVoice without breaking JA
- **License flexibility per engine** — if a great Apache JA model
  exists but only CC-BY-NC ZH model, we can run JA but not ZH
- **Standard contract** — companion-server and the existing test rigs
  see one v1-spec endpoint, no per-engine knowledge
- **Future-proof** — adding a 4th language is "spawn 4th engine + add
  one router entry"

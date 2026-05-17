# Qwen3-TTS-12Hz-1.7B-Base

| Field | Value |
|---|---|
| **Family** | Autoregressive multi-codebook LM (RQ-Transformer) |
| **License (code)** | Apache-2.0 |
| **License (weights)** | Apache-2.0 |
| **Languages** | JA, EN, ZH, KO, DE, FR, RU, PT, ES, IT |
| **Voice cloning** | Zero-shot (3-30s reference audio) |
| **Phonemization** | Built-in (model handles raw text) |
| **Params** | 1.7B |
| **Weights size** | ~4 GB |
| **VRAM** | ~4.8 GB peak (incl. KV cache) |
| **RTF (RTX 5080, optimized)** | 0.40 lab / ~0.55 production |
| **Best-in-class for** | Multi-language single-model deployments where Apache + zero-shot are mandatory |
| **Weakest in** | Native pitch-accent JA (worse than Style-Bert-VITS2); robustness on raw digit input (needs pre-normalization) |
| **Status in this repo** | ✅ Production-validated as multilingual baseline |

## When to pick Qwen3-TTS

- You need ONE model handling multiple languages with the same speaker
  embedding (voice consistency across JA → EN → ZH)
- Apache-2.0 commercial use is required
- True zero-shot voice cloning (no per-voice training pipeline)
- You can tolerate mediocre JA pitch-accent if you don't have a
  specialized JA engine alongside
- Sub-real-time inference on consumer GPU after the kernel-opt recipe
  ([13-inference-optimization.md](../13-inference-optimization.md))

## When NOT to pick it

- You only need Japanese — Style-Bert-VITS2 has better JA prosody
- You only need Chinese at peak quality — CosyVoice 3 via TensorRT-LLM
  is faster + higher quality
- You only need fast English — Kokoro-82M is smaller + lower latency
- You need per-voice fidelity that justifies fine-tuning — GPT-SoVITS
  v4 with LoRA gets higher per-character quality

## Architecture

Qwen3-TTS is an **RQ-Transformer**: each audio token is decoded via a
nested loop over a 15-codebook residual quantizer. The wrapper
`Qwen3TTSModel` runs roughly:

```text
for outer_step in range(audio_token_count):           # ~60 for 5s @ 12 Hz
    main_token = talker.generate(...)                  # 28-layer LM, 1 step
    for inner_step in range(15):                       # ← inner RQ loop
        code_predictor.generate(...)                  # 5-layer LM, 1 step
```

That's ~60 outer forwards + 900 inner forwards per ~5s of audio. Each
`.generate()` call traverses HuggingFace's full `GenerationMixin`
pipeline. The CPU dispatch overhead is the main perf bottleneck —
addressed in the optimization recipe.

## Quickstart

Install + download + synthesize. See [10-zero-shot-cloning.md](../10-zero-shot-cloning.md)
for the full recipe (reference clip selection, multi-clip concat,
quality presets, troubleshooting).

```bash
pip install -U qwen-tts huggingface_hub soundfile
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-Base --local-dir ./qwen3-tts-1.7b-base
```

```python
from qwen_tts import Qwen3TTSModel
import torch, soundfile as sf

model = Qwen3TTSModel.from_pretrained(
    "./qwen3-tts-1.7b-base",
    device_map="cuda:0", dtype=torch.bfloat16,
    attn_implementation="sdpa",
)
prompt = model.create_voice_clone_prompt(
    ref_audio="my_reference.wav",
    ref_text="My reference transcription.",
    x_vector_only_mode=False,
)
wavs, sr = model.generate_voice_clone(
    text="今日はとてもいい天気ですね。",
    language="Japanese",
    voice_clone_prompt=prompt,
    temperature=0.4, top_p=0.85, max_new_tokens=240,
)
sf.write("out.wav", wavs[0], sr)
```

## Pros

- **Apache-2.0 weights** — commercial use, no restrictions
- **True zero-shot** — 3-30s reference, no training
- **Sub-real-time** after optimization recipe (RTF 0.40 lab)
- **Multi-language with consistent voice** — same speaker embedding
  generalizes across JA / EN / ZH
- **Active maintenance** by Alibaba's Qwen team
- **Permissive deployment**: pip-installable, pure PyTorch, runs on
  Windows + RTX 5080 native (no WSL2 needed)

## Cons

- **Mediocre Japanese pitch accent** — multilingual training doesn't
  optimize for JA's prosodic distinctness (橋/箸/端 ambiguity)
- **Digit runaway** — raw "2024年12月31日" can make the model
  generate 30× expected audio (verified, audited). Fix: pre-normalize
  digit runs to kana via pyopenjtalk before TTS.
- **Stochastic EOS misfires** — model occasionally over-generates on
  certain inputs (long-tail). Fix: codec-repetition guard (force EOS
  when same codec_id emitted N times in a row) + max_new_tokens
  runaway cap.
- **Voice fidelity ceiling** — zero-shot averages over training
  speakers; can't match a fine-tuned LoRA on a specific voice's
  prosody patterns.
- **First-sample DC click** — minor: 5ms cosine fade-in fixes it.

## Production safety nets (validated in this repo)

In our companion-tauri deployment (see
[../12-integration.md](../12-integration.md)), the production engine
wrapper adds:

1. **Digit normalization** (pyopenjtalk g2p on `\d+` spans) — fixes
   runaway on dates/numbers/phone numbers
2. **Codec-repetition guard** — if last 8 outer codebook-0 tokens are
   identical → force EOS
3. **max_new_tokens=2000** runaway cap (was 200/240/320 — caused
   truncation on long stories)
4. **5ms raised-cosine fade-in** on output (eliminates start click)
5. **Production debug-capture** — if audio_s > 3× expected from text
   length, save WAV + text to `~/.cache/qwen3-tts-debug/` for replay
6. **ASR-validate-and-retry** (opt-in via env var; cross-thread
   cudagraph issue keeps it off by default)

Together: ~95% of real-world chat inputs produce clean audio.
Remaining edge cases (1-2 char utterances, unusual emoji sequences)
require model upgrade, not parameter tuning.

## Optimization recipe (5.98× speedup over baseline)

See [../13-inference-optimization.md](../13-inference-optimization.md)
for the full walkthrough. Summary:

| Tactic | Speedup | What it does |
|---|---|---|
| T1: tight inner predictor loop | 1.80× | Bypass HF GenerationMixin for the inner 15-codebook decode |
| T2: torch.compile inner forward | 2.81× cumulative | CUDA graph capture on the inner predictor |
| T3: tight outer talker loop | 5.98× cumulative | Same trick for the 28-layer outer talker |
| T4-prealloc: DynamicCache w/ config | 1.1× standalone | Pre-allocate cache layers so Dynamo doesn't recompile |

All four together: RTF 2.41 baseline → 0.40 lab / 0.55 production.

## Deployment difficulty

**1/5 (easiest of all open-source TTS).**

```
pip install -U qwen-tts huggingface_hub soundfile
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-Base --local-dir ./qwen3-tts-1.7b-base
```

Runs on Windows + RTX 5080 native. No WSL2, no Docker, no custom CUDA
builds, no specialized inference framework. The kernel-opt recipe is
~150 LOC of monkey-patches to apply at engine init.

## Validated configurations

| | Hardware | OS | Torch | qwen-tts | RTF |
|---|---|---|---|---|---|
| Production | RTX 5080 (Blackwell, sm_120, 16 GB) | Win 11 | 2.11.0+cu128 | 0.1.x | 0.55 (with T1+T2+T3+T4-prealloc) |

## Failure modes seen in production

| Symptom | Cause | Fix |
|---|---|---|
| 160s of garbage on "2024年12月31日..." | Digit runaway | Pre-normalize digits → kana via pyopenjtalk |
| Repeating first 2 words | Sampling stuck in attractor (low temp) | Codec-repetition guard + temperature retry |
| Truncated long stories | max_new_tokens cap (was 320) | Cap raised to 2000 (runaway safety only; rely on EOS) |
| Bad voice timbre on first few words | Cold-start in cudagraph capture (~30-60s first call) | Warmup at boot with throwaway synth |
| First-sample audible click | DC offset at sample 0 | 5ms raised-cosine fade-in |
| Cross-thread cudagraph assertion | faster-whisper init breaks cudagraph TLS | Don't load ASR in engine init (ASR-validate path opt-in) |

## See also

- [../10-zero-shot-cloning.md](../10-zero-shot-cloning.md) — practical zero-shot recipe (reference clip selection, presets, troubleshooting)
- [../11-multilingual.md](../11-multilingual.md) — cross-lingual cloning details
- [../13-inference-optimization.md](../13-inference-optimization.md) — full optimization walkthrough
- [../12-integration.md](../12-integration.md) — wiring Qwen3-TTS into a production app
- [../per-language/japanese.md](../per-language/japanese.md) — when to prefer Style-Bert-VITS2 over Qwen3-TTS for JA

# CosyVoice 3 (Fun-CosyVoice3-0.5B-2512)

| Field | Value |
|---|---|
| **Family** | Autoregressive LM + flow-matching decoder |
| **License (code)** | Apache-2.0 |
| **License (weights)** | Apache-2.0 (Fun-CosyVoice3-0.5B-2512 release) |
| **Best language** | Mandarin Chinese (Alibaba-native training) |
| **Voice cloning** | Zero-shot 3-30s reference |
| **Phonemization** | External: `ttsfrd` (Alibaba's text frontend) |
| **Params** | 0.5B (Fun-0.5B-2512); also 1B variants |
| **VRAM** | ~3-5 GB |
| **RTF (RTX 5080, PyTorch)** | ~0.5 |
| **RTF (RTX 5080, TensorRT-LLM)** | **~0.10** (5× real-time) |
| **Best-in-class for** | Chinese chat / character voices; emerging as strong multilingual option |
| **Status in this repo** | 🚧 Prototype planned via PyTorch path first, TRT-LLM second |

## When to pick CosyVoice 3

- You need the BEST Chinese voice quality from open-source
- Apache-2.0 commercial use required
- You can deploy via WSL2 or Docker (PyTorch path also works on Windows
  native but slower)
- True zero-shot voice cloning (no per-voice training)

## When NOT to pick it

- You only need Japanese (use Style-Bert-VITS2 instead — better JA
  pitch accent)
- You need Windows-native deployment with no WSL2 (PyTorch path works
  but you lose the TRT-LLM 4× speedup)
- The `ttsfrd` text-frontend dep is a blocker (see below)

## Why it wins for Chinese

- **Alibaba-native** — the training data is real Chinese language depth,
  not multilingual generalist
- **CER 0.81%** on Chinese test sets (state-of-the-art among
  open-source commercial-OK models)
- **Speaker SIM 78%** — *higher than the human reference at 75.5%*.
  Meaning: the model's clone is more consistent with the speaker's
  identity than a different recording of the same speaker. Cross-clip
  speaker drift in human recordings is normal; CosyVoice's
  embedding-driven clone eliminates that drift.
- **4× speedup with TensorRT-LLM** — RTF 0.10 on RTX 5080 means 5×
  faster than real-time. Sub-second TTS even on long replies.

## Architecture in brief

```
text → ttsfrd (text frontend) → phoneme + prosody tokens
     → AR LLM (0.5B) → discrete audio tokens
     → flow-matching decoder → mel
     → vocoder → waveform
```

The flow-matching decoder is non-AR, which avoids the runaway/loop
class of failures seen in fully-AR codec models. The AR part is just
the text→audio-token step, which is much shorter than full-waveform AR.

## Deployment paths

### Path A: PyTorch native on Windows

```bash
git clone https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice
pip install -r requirements.txt
huggingface-cli download FunAudioLLM/CosyVoice3-0.5B-2512 --local-dir ./cosyvoice3-0.5b
```

RTF ~0.5 on RTX 5080. Adequate for batch processing; for interactive
chat the 0.10 path is much better.

### Path B: TensorRT-LLM via WSL2 (recommended for production)

1. Install WSL2 Ubuntu 22.04
2. Install nvidia-container-toolkit + Docker
3. Pull CosyVoice's official TRT-LLM container
4. Convert weights to TRT-LLM engine (one-time, ~30 min)
5. Run inference

RTF ~0.10 on RTX 5080. 5× faster than PyTorch path. The setup is more
work but pays back on every synth call.

## Voice cloning

Zero-shot. Same reference clip pattern as Qwen3-TTS:

```python
from cosyvoice.cli.cosyvoice import CosyVoice

cv = CosyVoice("./cosyvoice3-0.5b")
audio = cv.inference_zero_shot(
    tts_text="你好，今天天气很好。",
    prompt_text="My reference transcription.",
    prompt_speech_16k="my_reference.wav",
)
```

Same the target speaker reference (`target_concat_diverse5.wav`) is reusable. The
speaker encoder is different from Qwen3-TTS / SBV2 / Higgs, so voice
timbre will differ slightly across engines. Cross-engine voice
consistency: ~85-90% (acceptable for typical use, blind A/B test
recommended).

## Pros

- **Apache-2.0** (clean commercial use)
- **SOTA Chinese quality** among permissive-license models
- **TRT-LLM speedup** available (5× faster than PyTorch baseline)
- **Zero-shot voice clone** (no training)
- **Flow-matching decoder** avoids AR runaway/loop class
- **Active maintenance** by Alibaba's FunAudioLLM team

## Cons

- **`ttsfrd` text-frontend dep** caused a prior trial to fail in this
  workspace. Needs 1-day spike to re-verify before committing.
- **TRT-LLM path is Linux-only** — Windows users need WSL2 or Docker
  for the 5× speedup. PyTorch native works but at RTF ~0.5.
- **Chinese-dominant training** — strong ZH but multi-language ability
  is decent rather than great. English / Japanese available but not
  best-in-class for those.
- **Larger model than alternatives** — 3-5 GB VRAM vs SBV2's 2 GB.

## Deployment difficulty

| Path | Difficulty | Notes |
|---|---|---|
| PyTorch native (Windows) | 3/5 | `pip install` + ttsfrd setup; ttsfrd has historical install issues |
| Docker (CosyVoice's official) | 2/5 | Container handles deps; needs nvidia-container-toolkit on Windows host |
| TensorRT-LLM via WSL2 | 5/5 | Requires WSL2, custom plugin compile, engine conversion. 5-7 days first-time setup. |

For our companion's first integration: **Docker path** (medium effort,
good speed, isolated from host env). Migrate to TRT-LLM later if perf
matters more.

## Production integration

In the multi-engine architecture:

```
tts-router:9890
       └─→ tts-zh:9892 (CosyVoice 3, Docker)
```

ZH-only — Qwen3-TTS handles other languages. Lazy-load (router spawns
the container on first ZH request; keeps warm for subsequent calls).

## See also

- [../per-language/chinese.md](../per-language/chinese.md) — why CosyVoice 3 over alternatives
- [../deployment/multi-engine.md](../deployment/multi-engine.md) — sidecar router
- [Upstream repo](https://github.com/FunAudioLLM/CosyVoice)
- [Hugging Face: Fun-CosyVoice3-0.5B-2512](https://huggingface.co/FunAudioLLM/CosyVoice3-0.5B-2512)
- Paper: arXiv:2505.17589

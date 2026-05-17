# Higgs Audio v2 / v2.5

| Field | Value |
|---|---|
| **Family** | Multimodal audio LM (text + audio input/output) |
| **License (code)** | Apache-2.0 |
| **License (weights)** | Apache-2.0 |
| **Best language** | English (also handles JA/ZH/multilingual with **explicit cross-lingual voice clone**) |
| **Voice cloning** | Zero-shot 3-30s reference, **with cross-lingual support** |
| **Phonemization** | Built-in (model handles raw text) |
| **Params** | v2.5 = 1B (condensed from v2's 3B) |
| **VRAM** | ~3 GB (v2.5) |
| **RTF (RTX 5080)** | ~0.5 (TBD — only RTX 4090 numbers published; need on-hardware measurement) |
| **Best-in-class for** | English chat / character voices needing cross-lingual voice consistency |
| **Status in this repo** | 🚧 Prototype planned |

## Why it wins for English (+ cross-lingual)

**The unique selling point:** Higgs is the only candidate in our
shortlist with **explicit cross-lingual voice clone support** —
designed-in capability for "use the same speaker reference, generate
speech in a different language."

For our multi-engine architecture this matters because:
- Asuna speaks JA via Style-Bert-VITS2 (fine-tuned per voice)
- Asuna speaks ZH via CosyVoice 3 (zero-shot clone)
- Asuna speaks EN via Higgs v2.5 (zero-shot clone)

Each engine has a different speaker encoder; voice consistency across
languages would normally drift heavily. Higgs's cross-lingual design
keeps the EN clone tight to the reference voice, even though the
reference is JA audio.

Beyond that:
- **75.7% win vs gpt-4o-mini-tts on EmergentTTS-Eval Emotions** —
  Higgs handles emotional intonation explicitly, outperforming
  commercial baselines
- Apache-2.0 weights (clean commercial use)
- 1B params (condensed from 3B) fits comfortably in our VRAM budget
  alongside JA + ZH engines

## When to pick Higgs Audio

- You need **English** TTS with good voice cloning quality
- You also need **cross-lingual voice consistency** (same speaker
  across JA/EN/ZH)
- Apache commercial OK
- Emotional / expressive output matters (TTS for storytelling, drama,
  affective companions)

## When NOT to pick it

- You only need fast English with limited voice variety →
  Kokoro-82M (RTF ~0.05) is much faster
- You only need Chinese → CosyVoice 3 has better ZH
- You only need Japanese → Style-Bert-VITS2 has much better JA pitch
  accent

## Architecture in brief

Multimodal audio LM — accepts text + audio as input, generates audio
as output. The "audio in" capability is what makes cross-lingual voice
clone clean: the model sees the reference voice as audio tokens, not
just a speaker embedding, so it can preserve voice character beyond
what a fixed-size embedding can capture.

```
text + reference_audio → audio-LM (1B params) → audio codec tokens
                      → codec decoder → waveform
```

## Quickstart

```bash
# Install Higgs Audio
pip install higgs-audio-v2  # or whatever the package name resolves to
huggingface-cli download bosonai/higgs-audio-v2.5 --local-dir ./higgs-audio-v2.5
```

```python
from higgs_audio import HiggsAudioModel
import soundfile as sf

model = HiggsAudioModel.from_pretrained("./higgs-audio-v2.5", device="cuda")
audio = model.generate(
    text="Hello, this is a test.",
    speaker_reference="asuna_reference.wav",  # JA reference works for EN output
    language="en",
)
sf.write("out.wav", audio.cpu().numpy(), 24000)
```

(Exact API may differ — verify with upstream docs.)

## Pros

- **Apache-2.0** (clean commercial use)
- **Explicit cross-lingual voice clone** (unique in our shortlist)
- **Strong emotion modeling** (75.7% win on EmergentTTS-Eval Emotions
  vs gpt-4o-mini-tts)
- **Reasonable size** (1B condensed) fits with other engines in 16 GB
  VRAM
- **Boson AI active maintenance** + v2.5 released recently

## Cons

- **RTF on Blackwell unverified** — only RTX 4090 numbers published
  (~0.5); expect similar or slightly faster on RTX 5080 but needs
  measurement
- **Newer / smaller community** vs Qwen3-TTS or XTTS-v2 → fewer
  community recipes, less battle-tested deployment paths
- **Not best-in-class for any single language** — wins on
  cross-lingual consistency, not on per-language quality
- **Larger than specialized models** (1B vs SBV2's 165M)

## Deployment difficulty

**2-3/5.** Pip-installable, but verify Blackwell + Windows compatibility.

## Production integration

In the multi-engine architecture:

```
tts-router:9890
       └─→ tts-en:9893 (Higgs Audio v2.5)
```

EN-only — lazy-load (router spawns Higgs sidecar on first EN
request; ~2-3s cold-load time first call, then warm).

Alternative: Higgs could serve as the **cross-lingual fallback**
engine when the specialized JA / ZH engines are unavailable or when
voice consistency across languages is the top priority (e.g. a user
who switches languages mid-conversation and wants the voice to feel
continuous).

## Open questions

1. **Actual RTF on Blackwell (RTX 5080)** — measure
2. **Voice fidelity vs SBV2 / CosyVoice on same Asuna reference** —
   blind A/B test required before shipping
3. **Latency profile** — is it AR like Qwen3-TTS (subject to
   per-token costs) or non-AR like F5-TTS? Determines optimization
   strategy.

## See also

- [../per-language/english.md](../per-language/english.md) — why Higgs over alternatives
- [../deployment/multi-engine.md](../deployment/multi-engine.md) — sidecar router
- [Upstream: Boson AI](https://www.boson.ai/blog/higgs-audio-v2)
- [Hugging Face: bosonai/higgs-audio-v2.5](https://huggingface.co/bosonai)

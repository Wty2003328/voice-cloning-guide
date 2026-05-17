# Per-model deep dives

Each model gets its own page covering:
- Architecture (AR / non-AR, codec used, ~param count)
- License (code + weights, commercial use)
- Voice cloning capability (zero-shot 3s, zero-shot 30s, per-voice
  training)
- Best languages + benchmarks
- Phonemization (built-in or external)
- Deployment (Windows native? WSL2? Docker?)
- Inference perf (RTF, VRAM)
- Known failure modes + safety nets
- Optimization insights (kernel-level tactics that worked)
- Status: hands-on validated by this repo, or research-only

## Index

| Model | Family | License (weights) | Status |
|---|---|---|---|
| [Qwen3-TTS-12Hz-1.7B-Base](qwen3-tts.md) | AR multi-codebook | Apache-2.0 | ✅ Validated |
| [GPT-SoVITS v4 (LoRA)](gpt-sovits-v4.md) | AR (GPT + SoVITS) | MIT | ✅ Validated (fine-tune track) |
| [CosyVoice 3 (Fun-CosyVoice3-0.5B)](cosyvoice-3.md) | AR + flow-matching | Apache-2.0 | 🚧 Research, prototype planned |
| [Style-Bert-VITS2](style-bert-vits2.md) | VITS (VAE-based) | MIT | 🚧 Research, prototype planned |
| [Kokoro-82M](kokoro.md) | Compact AR | Apache-2.0 | 🚧 Research |
| [F5-TTS / OpenF5](f5-tts.md) | Flow matching | F5: CC-BY-NC / OpenF5: Apache (alpha) | 🚧 Research |
| [XTTS-v2 (Coqui)](xtts-v2.md) | AR multilingual | MPL-2.0 (code) / CPML (weights) | 🚧 Research |
| [Higgs Audio v2 / v2.5](higgs-audio.md) | AR (Boson AI) | Apache-2.0 | 🚧 Research |
| [Sesame CSM-1B](sesame-csm.md) | AR (~1B) | Apache-2.0 | 🚧 Research |
| [VOICEVOX](voicevox.md) | Pre-trained canonical voices | LGPL / per-voice (most non-commercial) | 🚧 Research (license caveats) |

## What to look at first

If you're picking ONE model to start with for a commercial-OK
multilingual chat app: read **qwen3-tts.md** first — it's the
production-validated zero-shot baseline this repo is built around.

If you're picking for a specific language: go to
[../per-language/](../per-language/) first, then drill into the
recommended model's page.

## Contributing a new model page

Use [_template.md](_template.md) (coming) as the template. Fill in
your hands-on observations — license cliff-notes, what worked, what
didn't, real RTF on your hardware. Cite sources. Open a PR.

# 02 — Picking a TTS Model in 2026: A Decision Tree

The 2026 open-source TTS landscape has converged on a clear default
pattern (**vLLM-Omni in Docker** serving an Apache/MIT-licensed
multi-codebook LM) with a small set of well-understood escape hatches
when the default doesn't fit. This page is the decision tree.

## TL;DR

1. **On the vLLM-Omni Docker path?** Pick from the vLLM-Omni-native
   models — see [ch. 15 — Picking a model](15-vllm-omni-model-selection.md).
2. **Need the absolute Japanese quality ceiling and can run a sidecar?**
   Style-Bert-VITS2 JP-Extra is unbeaten on character-style JA MOS.
3. **Already-have-GPT-SoVITS-v4 user?** The legacy LoRA fine-tune path
   still works — see [`scripts/sovits-finetune/`](../scripts/sovits-finetune/).

## Decision flow

```text
                Need open-source TTS?
                        │
            Want OpenAI-compatible URL +
            one-container deploy?
                        │
              ┌─────────┴─────────┐
              │                   │
             Yes                  No
              │                   │
       vLLM-Omni in Docker        │
       (recommended path)         │
              │                   │
       Pick model by language ────│──────────────┐
              │                   │              │
        ┌─────┼─────┐             │              │
        ▼     ▼     ▼             ▼              ▼
       JA    ZH    EN        Need JA            Already
        │     │     │        MOS ceiling?      have a
        ▼     ▼     ▼              │           GPT-SoVITS
       OmniVoice CosyVoice3 Open  ▼            v4 weights
        +SFT      (Apache,  (eval Style-Bert-   set?
       (ch.16)    CER 0.81)  in    VITS2          │
                            flux)  JP-Extra       ▼
                                    │            Legacy LoRA
                                    │            path:
                                    │            scripts/
                                    │            sovits-finetune/
                                    ▼
                              Non-AR flow
                              architecture;
                              cannot ride vLLM-Omni
                              cleanly — needs
                              its own sidecar.
```

## Per-language picks (2026-05)

| Language | Top pick | Profile / path |
|---|---|---|
| **Japanese** | OmniVoice + per-character SFT | `docker compose up` (default) + ch. 16 SFT recipe |
| **Chinese** | CosyVoice3 | `docker compose --profile cosy3 up` |
| **English** | Open — see [per-language/english.md](per-language/english.md) | `--profile qwen` (Qwen3-TTS baseline) or default OmniVoice |
| **Multilingual (1 model, all langs)** | VoxCPM2 (30 langs) or Qwen3-TTS (~10 langs) | `vllm serve openbmb/VoxCPM2` or `--profile qwen` |

No model wins everywhere. The tree above is the production answer in
2026-05; revisit the per-language pages for the model-by-model trade-offs.

## Model summary

| Model | License | Voice clone | Strength | Deploy |
|---|---|---|---|---|
| **OmniVoice** (`k2-fsa/OmniVoice`) | Apache-2.0 | Zero-shot 3–10 s + SFT recipe | Japanese (36k+ hr JA pretrain, char-level Qwen3 tokenizer) | vLLM-Omni Docker (default) |
| **CosyVoice3** (`FunAudioLLM/Fun-CosyVoice3-0.5B-2512`) | Apache-2.0 | Zero-shot 3–30 s | Chinese (CER 0.81%, SIM 78% > human 75.5%); 18 Chinese dialects incl. Cantonese | vLLM-Omni Docker `--profile cosy3` |
| **Qwen3-TTS-12Hz-1.7B-Base** | Apache-2.0 | Zero-shot 3–30 s | Multilingual baseline (JA / ZH / EN / KO + 6 EU langs) | vLLM-Omni Docker `--profile qwen` |
| **VoxCPM2** (`openbmb/VoxCPM2`) | Apache-2.0 | Zero-shot ("Ultimate Cloning" mode) | 30 languages | `vllm serve openbmb/VoxCPM2` |
| **Voxtral-TTS-4B** (Mistral) | Apache-2.0 | Zero-shot | Multilingual EU-centric; supports EN; explicitly does NOT support JA | vLLM-Omni Docker |
| **Style-Bert-VITS2 JP-Extra** | AGPL-3.0 code | Per-voice fine-tune (~1–2 days train) | JA character-style MOS 4.37 (vs human 4.38) | Non-AR flow; needs its own Python sidecar |
| **GPT-SoVITS v4 (LoRA)** | MIT | Per-voice fine-tune (~1 hour) | Legacy character voice path; ZH / JA / EN / KO / YUE | [`scripts/sovits-finetune/`](../scripts/sovits-finetune/) |
| **Fish-Speech S2 Pro** (`fishaudio/s2-pro`) | Open weights | Zero-shot | Strongest JA paper numbers (CV3-Eval JA CER 3.96%) | Blocked — vLLM-Omni v0.20 has `ModuleNotFoundError: fish_speech` |
| **IndexTTS-2** | bilibili proprietary | Zero-shot | 42k hr JA training; SS 0.833 / WER 9.95% Common Voice JA | Parked — multi-week port to vLLM-Omni |
| **Higgs Audio v2.5** | Apache-2.0 | Zero-shot | Cross-lingual identity preservation | Rejected for 16 GB envelope (~10.75 GB weights alone) |
| **RVC v2** | MIT | Voice conversion (not TTS) | Real-time live mic, singing | Separate process; chain after any TTS |

## When to pick each path

### vLLM-Omni Docker — the recommended path

Pick this when:

- You want **one OpenAI-compatible URL** any client can talk to.
- You want to **swap models with a profile flag**, not reinstall a
  Python env per engine.
- You have **Docker + nvidia-container-toolkit + a ≥ 8 GB GPU** (16 GB
  for headroom if Windows desktop shares the device).

Skip when:

- The model you need isn't in the vLLM-Omni-native set (e.g. Style-Bert-VITS2,
  IndexTTS-2 — both require their own runtimes).
- You're on a CPU-only host. vLLM-Omni requires CUDA.

### Style-Bert-VITS2 JP-Extra — the JA quality ceiling

Pick this when:

- The MOS ceiling on character-style Japanese matters more than deploy
  simplicity.
- You can afford a separate Python sidecar process (its own
  conda env, its own healthz route, its own request format).

The architecture is non-AR (VITS / flow-matching VAE), which means it
can't share vLLM-Omni's PagedAttention or its OpenAI-compatible
adapter cleanly. It needs to run as its own service.

Deep dive: [`models/style-bert-vits2.md`](models/style-bert-vits2.md).

### Legacy GPT-SoVITS v4 LoRA — already-have-v4 users

Pick this when:

- You already have a GPT-SoVITS v4 weights set and don't want to redo
  the dataset.
- You specifically want the SoVITS-style training pipeline (Demucs +
  slicer + ASR + features + semantic + SoVITS + GPT training).

Scripts: [`scripts/sovits-finetune/`](../scripts/sovits-finetune/).
Deep dive: [`models/gpt-sovits-v4.md`](models/gpt-sovits-v4.md).

For **new** projects, prefer the OmniVoice SFT recipe — same data
requirement (~20 min), shorter wallclock, drops straight into the
production Docker compose. See
[ch. 16 — OmniVoice SFT](16-omnivoice-sft-recipe.md).

### Voice conversion (RVC v2)

Pick this when:

- Live-mic real-time conversion is the use case (streaming, singing).
- You want to chain into an existing TTS just for timbre.

RVC swaps timbre only — prosody comes from the input audio. For
single-character TTS, the modern recipe (OmniVoice or CosyVoice3 +
SFT) replaces the historical "TTS + RVC" chain at lower latency and
higher quality.

## What changed since the 2024 / early-2026 era

Pre-2026 production patterns (per-engine Python sidecars, one process
per model) are deprecated for new projects. The vLLM-Omni Docker pattern
consolidates ~14 TTS architectures into one container image, one URL,
one wire contract. Existing sidecar deploys still work; new builds
should start from [ch. 15 — vLLM-Omni Docker](15-vllm-omni-docker.md).

## See also

- [ch. 00 — Landscape 2026](00-landscape-2026.md) — architecture taxonomy
  + license matrix.
- [ch. 15 — vLLM-Omni Docker](15-vllm-omni-docker.md) — production
  deploy walkthrough.
- [ch. 15 — Picking a model](15-vllm-omni-model-selection.md) — empirical
  per-model eval.
- [ch. 16 — OmniVoice SFT](16-omnivoice-sft-recipe.md) — fine-tuning a
  character voice on the production stack.
- [per-language/](per-language/) — language-specific picks.
- [models/](models/) — per-model deep dives.

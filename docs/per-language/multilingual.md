# Multilingual TTS — when one model serves many languages

> **Status:** Research in progress (task #141). This page complements
> the per-language pages: it covers the case where you want ONE model
> handling all languages instead of N specialized engines.

## When you want a single multilingual model

- VRAM tight (one model ~1-3 GB vs N×1-3 GB for specialized)
- Voice clone consistency across languages is the priority (same
  speaker_embedding produces "the same voice" regardless of language —
  with specialized engines, each language's speaker encoder differs)
- Deployment simplicity (one process, one set of weights, one update
  path)
- Mixed-language content within a single utterance (English loanwords
  in Japanese chat, code-mixed reply) — multilingual models handle this
  natively; routing to specialized engines requires sentence-level
  language detection

## When you want specialized per-language (multi-engine)

- Per-language quality matters more than cross-language consistency
- You can afford the VRAM + deployment complexity
- Most replies are single-language (chat in JA, then ZH later — not
  mid-sentence switching)

See [../deployment/multi-engine.md](../deployment/multi-engine.md) for
the router architecture that lets you keep both options open: route by
language to specialized engines for JA/ZH/EN, fall back to a
multilingual generalist for everything else.

## Candidate multilingual models

1. **Qwen3-TTS-12Hz-1.7B-Base** — Apache-2.0, true zero-shot,
   sub-real-time. JA/EN/ZH/KO/DE/FR/RU/PT/ES/IT supported. Quality
   uneven across languages (JA mediocre vs native-JA models). Our
   production baseline. [Deep dive](../models/qwen3-tts.md).

2. **CosyVoice 3** — Apache-2.0, multilingual. Strongest in Chinese
   but good across all supported languages. Heavier deployment for
   peak performance (TensorRT-LLM via WSL2).

3. **XTTS-v2 (Coqui)** — MPL/CPML, multilingual zero-shot. Older
   (2023) but battle-tested. Good speaker fidelity.

4. **MeloTTS** — MIT, fast inference, multilingual. Lower quality
   ceiling than the above; useful when latency matters most.

## Recommendation (preliminary)

For most users: **Qwen3-TTS-12Hz-1.7B-Base** as the multilingual
generalist + multi-engine router for per-language quality where it
matters most (typically JA).

Once research lands, this page will include:
- Cross-language quality matrix (each model × each language)
- Voice consistency scores (cosine similarity of speaker embeddings
  across languages, same reference)
- Code-mixing test cases (JA-embedded EN, etc.)

## See also

- [../11-multilingual.md](../11-multilingual.md) — cross-lingual zero-shot cloning recipe (using Qwen3-TTS specifically)
- [japanese.md](japanese.md), [chinese.md](chinese.md), [english.md](english.md) — per-language alternatives
- [../deployment/multi-engine.md](../deployment/multi-engine.md) — sidecar router that combines multilingual + specialized engines

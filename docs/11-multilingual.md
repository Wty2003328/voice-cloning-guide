# 11 — Multilingual & Cross-Lingual Voice Cloning

Modern zero-shot models can take a voice reference in one language and
synthesize speech in *other* languages — same voice, different
language. This doc covers how it works, the practical limits, and how
to get good results.

## The problem

You have a reference clip of a Japanese voice actor. You want them to
say "Hello, how are you?" in English with their Japanese-voice timbre.
Or read a Chinese sentence. Or a Korean greeting. Without retraining.

This is **cross-lingual zero-shot voice cloning**. It's hard because:

- The model needs to disentangle **speaker identity** (timbre,
  pitch range, vocal tract) from **language phonotactics** (which
  phonemes the speaker is actually producing).
- The reference clip is full of Japanese phonemes — the model has to
  borrow the speaker's *acoustic style* without dragging those
  phonemes into the target.

## How Qwen3-TTS handles it

Qwen3-TTS uses a **discrete multi-codebook LM** architecture. The
speaker embedding is computed by a separate encoder and is mostly
language-agnostic — it captures the acoustic properties of the voice
without locking to the specific phonemes in the reference.

In `create_voice_clone_prompt(...)`:

- **`x_vector_only_mode=True`** uses *only* the speaker embedding.
  Prompt-text-paired features (which carry the reference language's
  phonotactics) are dropped. This is the right mode for cross-lingual.
- **`x_vector_only_mode=False`** uses both speaker embedding AND
  paired features. Better voice fidelity for same-language synthesis;
  causes phoneme bleed for cross-lingual.

We call this **hybrid speaker conditioning** in our wrapper code —
pick the prompt mode based on `target_language == reference_language`.

## The 2026 quality bar (measured)

With our 33-case eval ([a single JA voice reference](10-zero-shot-cloning.md)):

| Direction | Pass rate | Notes |
|-----------|-----------|-------|
| **JA → JA** | 13/13 (100%) | Same language, fully natural — production ready |
| **JA → EN** | 10/10 (100%) | Cross-lingual works — voice identity holds, mild accent expected |
| **JA → ZH** | 9/10 (90%) | The 1 fail is Whisper homophone artifact, not TTS — effectively 100% |
| **JA → KO** | ~80% (informal) | Works but de-prioritized — phonotactic gap is wider |
| **JA → YUE** (Cantonese) | Not supported | Qwen3-TTS-Base doesn't cover Cantonese; CosyVoice 3 does |

## Languages supported

Qwen3-TTS-12Hz-1.7B-Base natively covers 10 languages:

| Code | Name | Cross-lingual quality vs JA ref |
|------|------|-------------------------------|
| `ja` | Japanese | Native (same as ref) |
| `en` | English | Excellent |
| `zh` | Chinese (Mandarin) | Excellent |
| `ko` | Korean | Good |
| `de` | German | Untested |
| `fr` | French | Untested |
| `ru` | Russian | Untested |
| `pt` | Portuguese | Untested |
| `es` | Spanish | Untested |
| `it` | Italian | Untested |

Pass language names (not codes) to the API: `"Japanese"`, `"English"`,
etc. — see `LANG_HINT` in `qwen3_engine.py` for the map.

**Not in Qwen3-TTS:** Cantonese, Vietnamese, Thai, Arabic, Hindi.
For Cantonese specifically, [Fun-CosyVoice3-0.5B-2512](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512)
has first-class support (and a `<|yue|>` dialect token) but its
proprietary `ttsfrd` text frontend is needed and not in the HF
distribution.

## Cross-lingual gotchas

### Latin loanwords in JA targets

In English context, you'd say "iPhone" — but in Japanese context, the
canonical pronunciation is "アイフォン" (ai-fon). Qwen3-TTS handles this
correctly *if you give it the raw Latin "iPhone" in JA text*. Do
**not** pre-convert to katakana manually — the LLM's text encoder
knows the right katakana already and any manual conversion will lose
accuracy.

### Chinese reading of English letters

In Chinese context, native speakers code-switch into English for
acronyms. "请设置API密钥" → "請設置 A-P-I 密鑰" (reading API as English
letters, not as 艾-皮-埃). Qwen3-TTS reproduces this correctly — again,
don't pre-normalize.

### Japanese kanji bleed in cross-lingual

Older models (CosyVoice 2, GPT-SoVITS) had a documented "kanji bleed"
bug: when given a JA reference and a ZH target containing kanji that
exist in both writing systems, the model would emit JA-tinged
Mandarin. CosyVoice 3 fixed this by pre-converting JA text to
katakana. Qwen3-TTS sidesteps it via the LLM text encoder — no manual
fix needed.

### Tonal accuracy for ZH

Chinese is tonal — wrong tone changes the meaning. Cross-lingual
Mandarin from a JA-voice reference is usually tonally correct in
Qwen3-TTS but expect mild pitch-contour drift on heavily emphasized
syllables. For broadcast-quality Mandarin you'd want a Mandarin-native
reference, but for chatbot-grade output the JA-reference path is fine.

## Practical recipe

```python
# Pseudocode for a multi-language voice
prompt_paired   = model.create_voice_clone_prompt(ref_audio=..., ref_text=...,
                                                   x_vector_only_mode=False)
prompt_xvec     = model.create_voice_clone_prompt(ref_audio=..., ref_text=...,
                                                   x_vector_only_mode=True)

REFERENCE_LANG = "ja"  # the language of your reference clip

def synthesize(text, target_lang):
    prompt = prompt_paired if target_lang == REFERENCE_LANG else prompt_xvec
    return model.generate_voice_clone(
        text=text, language=LANG_MAP[target_lang],
        voice_clone_prompt=prompt,
        temperature=0.4, top_p=0.85, max_new_tokens=240,
    )
```

This is the production code in our companion wrapper
(`qwen3_engine.py`). Same speaker, four languages, one model.

## When to NOT use cross-lingual

- **Broadcast / professional audiobook quality** in a non-reference
  language. Train on native speakers of that language instead.
- **Languages outside Qwen3-TTS's 10** — Cantonese, Vietnamese, etc.
  Either accept English-mode fallback or pick a model that natively
  supports your target.
- **Tonally-strict use cases** like teaching Mandarin pronunciation —
  use a native Mandarin reference, not a cross-lingual JA-reference.

For most chat / companion / VTuber use cases, cross-lingual is good
enough that listeners won't notice unless they're specifically
listening for accent.

## What the field looked like before Qwen3-TTS (Jan 2026)

If you're reading this and a newer model has come out, the comparison
table in [02-comparison.md](02-comparison.md) tracks the landscape.
Pre-Qwen3-TTS, the best cross-lingual options were:

- **CosyVoice 3** (Dec 2025) — fixed JA→ZH kanji bleed, added
  Cantonese dialect token. Local install is finicky due to proprietary
  text frontend.
- **IndexTTS-2** (Sep 2025) — best anime expressiveness, but
  ZH/EN/JA only (no KO).
- **Higgs Audio v2.5** — strong cross-lingual identity preservation;
  no Cantonese.

We picked Qwen3-TTS for its install-cleanness, all-3-priorities
(ja/en/zh) at production quality, and native LLM text handling.

## See also

- [10-zero-shot-cloning.md](10-zero-shot-cloning.md) — base zero-shot tutorial
- [12-integration.md](12-integration.md) — wiring multilingual cloning into an app
- [02-comparison.md](02-comparison.md) — comparison of all 2026 multilingual TTS models

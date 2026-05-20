# 14 — Cross-Lingual Voice Cloning: What Actually Fails in 2026

You have a clean Japanese voice reference. You want the same character
saying English and Chinese. How well does zero-shot cross-lingual cloning
work in 2026?

This page documents the empirical findings from
`tts_lab/sbv2_lab/xlingual_experiment.py` (run 2026-05-18). It updates
the 2024-era folklore that "cross-lingual cloning produces unintelligible
output" — modern (2025+) multilingual codebook architectures fixed the
content-drift problem. **The remaining failure mode is timbre / accent
leak, not content.**

## TL;DR

- **Content fidelity is good.** Modern multilingual zero-shot models
  (Chatterbox-MTL v2 verified here; CosyVoice2 and F5-TTS v2 in
  principle) preserve the text faithfully when cloning across languages.
  Char-jaccard 0.86-1.00 against the prompt for the target speaker-JA → EN/ZH cases.
- **Accent leak is the real cost.** Short utterances especially carry
  source-language phonotactics into the target — a Japanese reference
  cloning English produces Japanese-accented English, more pronounced
  on short utterances.
- **Recommendation:** use a **native target-language reference** when
  naturalness matters. Use cross-lingual cloning only when consistent
  timbre across languages outweighs accent naturalness (e.g., one
  fictional character speaking many languages, where slight accent
  is a feature not a bug).
- **The old "content drift" / "kanji-bleed" failure mode IS dead** for
  the SOTA 2025+ architectures — but is still alive in older ones
  (GPT-SoVITS-v4, Qwen3-TTS in the 2024-2025 range) and in engines
  whose text-frontend dependencies aren't available in your env (e.g.
  CosyVoice without ttsfrd; see "Engine-specific footguns" below).

## The experiment

- **Voice reference (cross-lingual case):** `target_voice/audio/0027.wav`
  — 6.76s of clean JA female speech. Same speaker across all
  cross-lingual cases.
- **Engines tested:**
  - SBV2 (target speaker fine-tune) (JA same-language baseline; production path)
  - Chatterbox-Multilingual v2 (ZH native, ZH cross-lingual, EN native,
    EN cross-lingual)
- **Per case:** 2 sentences (short greeting + longer descriptive),
  6 cases × 2 = 12 synth calls + 1 ASR pass per call.
- **Two metrics:**
  - **`char_jaccard`** — char-set jaccard between Whisper transcript
    (forced target language) and the input prompt. Higher = the
    synth produced the right words.
  - **`auto_lang @ prob`** — Whisper's language auto-detection on the
    synthesized audio with NO language hint. Mismatch from target =
    accent leak; lower probability on the target = degraded
    naturalness.

## Raw numbers (2026-05-18)

| Engine | Mode | Tag | jaccard | auto_lang | Verdict |
|--------|------|-----|---------|-----------|---------|
| SBV2 (target) | native_ja | ja_short | 0.69 | ja @ 0.89 | ✅ baseline |
| SBV2 (target) | native_ja | ja_long | 0.95 | ja @ 0.99 | ✅ baseline |
| Chatterbox | native_zh | zh_short | 0.39 | zh @ 1.00 | ⚠ content shaky on short ZH |
| Chatterbox | native_zh | zh_long | 0.67 | zh @ 1.00 | ✅ accent native |
| Chatterbox | **xlingual_zh** | zh_short | **1.00** | zh @ 0.99 | ✅ content + accent both clean |
| Chatterbox | **xlingual_zh** | zh_long | **0.88** | zh @ 0.98 | ✅ content + accent both clean |
| Chatterbox | native_en | en_short | 1.00 | en @ 0.72 | ✅ |
| Chatterbox | native_en | en_long | 0.86 | en @ 0.99 | ✅ |
| Chatterbox | **xlingual_en** | en_short | **1.00** | **ja @ 0.64** | ❌ **accent leak** |
| Chatterbox | **xlingual_en** | en_long | 0.86 | en @ 0.96 | ✅ accent recovers on longer text |

(Char-jaccard < 1 is partly due to Whisper's punctuation normalization,
not a real content miss — the transcripts read as the prompt minus
commas/periods.)

## What this tells us

### 1. Content survives cross-lingual cloning in modern models

Every cross-lingual case scored ≥ 0.86 char-jaccard — same as native.
The model is producing the prompted text. The 2024-era "kanji bleed"
failure (where a Chinese prompt would synthesize a phonologically
plausible but completely unrelated sentence) does **not** show up in
Chatterbox-MTL v2.

This contradicts the framing in `11-multilingual.md` that implied
content drift was an open problem — at least with this generation of
architectures (T3 codebook LM + flow vocoder + dedicated speaker
encoder), it's solved.

### 2. The accent leak is small but real

The clearest signal is `xlingual_en/en_short`: char-jaccard 1.00
(content perfect) but Whisper auto-detected the audio as Japanese at
0.64 probability instead of English. Listening: it's "Hello, nice to
meet you" delivered with a Japanese-accented English vowel space.

The longer English utterance (`xlingual_en/en_long`) recovered —
Whisper detected EN at 0.96. The model has more audio frames to
commit to the target language's phonotactics; short utterances don't
give it enough rope.

ZH cross-lingual from JA reference did NOT show this accent leak
(both detected as ZH @ ≥ 0.98). The phonological distance between JA
and ZH is greater than JA and EN, so the model commits harder to the
target language pattern; meanwhile, JA→EN's shared vowel inventory
makes accent leak more likely.

### 3. The cost is naturalness, not intelligibility

If your downstream consumer is humans, "Japanese-accented English" is
charming for a JA character speaking EN in fiction but jarring for a
production assistant. If your downstream is ASR (e.g. you're piping
the synthesis into another model that needs to recognize the words),
content fidelity is what matters — and that's intact.

## Recommendation: per-language native references

For applications where the avatar speaks multiple languages and
naturalness in each is non-negotiable:

| Language | Engine | Reference |
|----------|--------|-----------|
| JA | Style-Bert-VITS2 fine-tuned | your target voice (137-clip dataset, native JA) |
| ZH | Chatterbox-Multilingual v2 | native ZH reference clip (TBD per character) |
| EN | Chatterbox-Multilingual v2 | native EN reference clip (TBD per character) |

For "the SAME character speaking many languages with consistent timbre":

- Same Chatterbox-MTL with one JA reference, accept the accent leak
  as character flavor.
- OR maintain three separate reference clips of the same character
  recorded in each language (best-of-both-worlds; expensive to source
  recordings).

The companion's `tts_lab/launch_tts.py` registry supports both
patterns — each voice in the per-engine `voices.json` is a separate
entry with its own reference clip + language tag.

## Engine-specific footguns observed during the experiment

### CosyVoice2-0.5B has a content-hallucination bug in this env

CosyVoice2 was the original ZH engine pick — Apache 2.0, ZH-specialist,
ttsfrd-free per its WeTextProcessing fallback. **In practice it
generated fluent Mandarin audio that has NOTHING to do with the input
text.** Synth call for `"你好,很高兴认识你。"` returned audio of
~"因為我放了個三次的水所以我現在不太能夠跟她們說話" — completely
unrelated content. Same failure mode as the 2025 CosyVoice 3 rejection
(see [[project-tts-multilang-robustness]]); the v2 fallback path is
nominally there but doesn't fix it under torch nightly cu128 + Blackwell.

Workaround applied: switched ZH to Chatterbox-Multilingual. The
production launcher still has the `cosyvoice2-zh` entry registered
because protocol-conformance tests pass (audio bytes are produced) —
but it's a known-broken endpoint for actual content. See
[[project-cosyvoice-install-gotchas]].

### Higgs Audio v2.5 doesn't fit the 8GB VRAM cap

Higgs was the original EN pick per the SOTA audit; in practice the
only published TTS weights (`bosonai/higgs-audio-v2-generation-3B-base`)
are 10.75 GB at bfloat16 — over budget before KV cache + activations.
Plus the v2.5 HF config schema is incompatible with the GitHub code
without a multi-step rollback. Rejected; Chatterbox-MTL took its slot.
See [[project-chatterbox-install-gotchas]].

### Older engines DID have content drift

The 2024-era `gpt-sovits-v4` and `qwen3-tts-1.7b-base` had real
content-drift on cross-lingual paths — see archived
[02-comparison.md](02-comparison.md) and [10-zero-shot-cloning.md].
The framing in those chapters predates the 2025 architectural shift;
treat their pessimism about cross-lingual as accurate for those
engines, not as a general law.

## Reproducing this

```powershell
# From workspace root with the tts conda env on PATH for the harness:
python -m tts_lab.sbv2_lab.xlingual_experiment

# Outputs go to tts_lab/sbv2_lab/xlingual_out/:
#   <engine>_<mode>_<tag>.wav  — the synth output per case
#   report.json                — machine-readable summary
```

The script auto-launches each registered engine via `launch_tts.py`,
runs the synth + ASR battery, and writes per-WAV results. Edit the
`TESTS` list at the top to add a new engine, language, or reference
clip.

## Related

- [10 — Zero-Shot Cloning](10-zero-shot-cloning.md) — the same-language
  case; production path for most setups.
- [11 — Multilingual](11-multilingual.md) — when ONE model serves many
  languages; complement to the per-language pattern.
- [per-language/japanese.md](per-language/japanese.md),
  [per-language/chinese.md](per-language/chinese.md),
  [per-language/english.md](per-language/english.md) — per-language
  engine picks.

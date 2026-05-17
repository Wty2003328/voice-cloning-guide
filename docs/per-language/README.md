# Per-language model recommendations

Pick a TTS model based on your **target language**. Each page below
covers:
- Top 3 candidate models with rationale
- Quality benchmarks (MOS / UTMOS / WER when available)
- Voice cloning support per candidate
- Phonemization story
- Deployment difficulty
- Recommended pick + why

| Language | Page | Status |
|---|---|---|
| Japanese | [japanese.md](japanese.md) | 🚧 In progress (task #141) |
| Mandarin Chinese | [chinese.md](chinese.md) | 🚧 In progress (task #141) |
| English | [english.md](english.md) | 🚧 In progress (task #141) |
| Multilingual (one model, multiple langs) | [multilingual.md](multilingual.md) | 🚧 In progress (task #141) |

If you have a language not yet covered (Korean, Spanish, Arabic, etc.)
and have validated a model for it, open a PR with a new page following
the format of the existing ones.

## How the recommendations are made

Each page is informed by:
1. Apache/MIT/BSD license (commercial use OK) — non-negotiable
2. Voice cloning support — zero-shot preferred, per-voice fine-tune
   acceptable if quality justifies
3. Native phonemization for the language (proper pitch accent, tone
   marks, stress prediction)
4. Sub-real-time inference on consumer GPU (RTX 4060+ baseline)
5. Active community / maintenance signal

See [../00-landscape-2026.md](../00-landscape-2026.md) for the
underlying taxonomy.

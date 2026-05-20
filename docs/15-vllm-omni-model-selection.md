# 15 — Picking a vLLM-Omni model

vLLM-Omni's registry lists ~14 TTS architectures. Most are mediocre for
Japanese, most don't fit a 16 GB GPU, and most don't actually deliver
clean voice-cloning even when they claim to. This page documents the
2026-05 empirical eval that landed us on **OmniVoice** as the production
pick.

The eval rig is at
[`vllm-omni-tests/`](https://github.com/<your-user>/vllm-omni-deploy):
21 JA prompts × N reference clips, content-checked via faster-whisper
large-v3 (char-jaccard against input), language auto-detect.

## TL;DR

For Japanese on a 16 GB Blackwell GPU with Windows desktop running:

1. **OmniVoice** (`k2-fsa/OmniVoice`) — 36,914 hrs JA training,
   char-level Qwen3 tokenizer (no kanji-byte-fallback trap), FLEURS JA
   CER 5.96. **Production pick. ~7 GB system VRAM.**
2. **VoxCPM2** (`openbmb/VoxCPM2`) — 30 langs incl. JA, voice cloning,
   ~8 GB VRAM. Untested in our env. Decent backup.
3. (Everything else either failed our content-fidelity gate, exceeded
   VRAM, or is blocked by an upstream bug. Details below.)

## Eval rig

A production-realistic battery: 21 prompts covering greetings, casual
chat, affection, concern, joy, sadness, questions, narrative,
instruction, numbers/dates. Mean jaccard ≥ 0.95 on all 21 = ship-ready.
Mean ≤ 0.6 with one common failure mode = reject.

Reference clip: a 3–8 s clean clip of the target voice (any clean JA
voice works — distinctive timbre is fine).

```bash
cd vllm-omni-tests/
python run_eval.py --endpoint http://127.0.0.1:8000 --whisper-model large-v3
```

Outputs `out/report.md` with a summary table + per-prompt ASR
transcripts + the WAV files for subjective listening.

## Results by model

### ✅ OmniVoice (`k2-fsa/OmniVoice`) — PICK

| Test | Mean jaccard | Notes |
|---|---|---|
| 21-prompt eval, base zero-shot | **0.95 – 0.97** across 3 ref clips | Char-level tokenizer handles raw kanji cleanly. Long-form 6-sentence input handled in one synth call. |
| Same eval, post-SFT (400 steps on ~20 min target voice) | **0.96 – 0.97** | Content preserved + timbre clearly improved per A/B listen. |
| System VRAM | **7.2 GB** | Container alone ~2.9 GB; Windows desktop adds ~4.3 GB. Under "around 8 GB" target. |

Why this works: OmniVoice is a Qwen3-0.6B backbone + Higgs-codec
decoder, trained explicitly on 36k+ hours of JA. The char-level
tokenizer means kanji are first-class input — none of the
byte-fallback gibberish that broke CosyVoice3.

Deploy: `docker compose up -d` (default service) in
[`vllm-omni-deploy`](https://github.com/<your-user>/vllm-omni-deploy).

### ❌ CosyVoice3 (`FunAudioLLM/Fun-CosyVoice3-0.5B-2512`) — REJECTED

| Test | Mean jaccard | Notes |
|---|---|---|
| 21-prompt eval, base | **0.37** | Severe content drift on most prompts. Only katakana-only prompts passed. |
| Same eval + kanji→kana adapter patch | **0.65 (helped) — 0.10 (worse)** | Short sentences improved; long-form / numbers got *worse*. |

Root cause (paper arxiv:2505.17589): the model was trained on
**kana-converted text**, but neither the upstream pipeline nor the
vLLM-Omni adapter applies kanji→kana at inference. We wrote the kana
patch (fugashi + num2words → hiragana before BPE) — it helps short
cleanly-phrased inputs but cannot lift the model's ~9% base JA CER
ceiling on long-form or numeric content.

Keep around as a Chinese specialist (Apache, CER 0.81% native ZH);
**do not use for JA**.

Patch lives at [`Dockerfile.cosy-ja`](https://github.com/<your-user>/vllm-omni-deploy/blob/main/Dockerfile.cosy-ja)
+ a serving_speech.py edit in
[`vllm-omni-fork`](https://github.com/<your-user>/vllm-omni-fork).

### ❌ Qwen3-TTS-1.7B-Base — REJECTED earlier

Generic multi-lang. Mediocre JA per our 2026-Q1 eval (recorded in
[`docs/per-language/japanese.md`](per-language/japanese.md) — "digit
runaway, EOS misfires"). Not retested in the new rig because the
qualitative failure mode hasn't changed.

### ❌ Qwen3-TTS-1.7B-CustomVoice — REJECTED architecturally

The `CustomVoice` variant is NOT a zero-shot model. The
`task_type=CustomVoice` code path looks up a `speaker` name in a
pre-baked `spk_id` dict — it does not take `ref_audio`. You'd have to
fine-tune speaker tokens INTO the model weights to add a new voice.
Skip.

### ❌ Fish-Speech S2 Pro (`fishaudio/s2-pro`) — BLOCKED

On paper: best published JA quality of any candidate (100k hrs JA
training, Bradley-Terry preference winner in their 2026 blind test,
CV3-Eval JA CER 3.96%).

In practice: **vLLM-Omni v0.20.0 is missing the `fish_speech` pip
package** the adapter imports from
(`ModuleNotFoundError: No module named 'fish_speech'`, upstream issue
#2404, closed without resolution). The candidate fix `v0.21.0rc1` has
a separate `split_routed_experts` import error from vllm upstream PR
#42434. No usable tag exists as of 2026-05.

Watch the [upstream issue](https://github.com/vllm-project/vllm-omni/issues/2404).

### ⚠ VoxCPM2 (`openbmb/VoxCPM2`) — UNTESTED but plausible

- 2B params, ~8 GB VRAM (borderline on our 5080 with Windows).
- 30 languages including JA, voice cloning ("Ultimate Cloning" mode
  takes prompt_wav + prompt_text).
- Internal paper JA CER 2.40%; CV3-eval JA CER 5.96% (Fish wins at
  2.76% but we can't use Fish).
- JA proper-noun / mixed-number / long-form is "needs separate
  listening" per lilting.ch hands-on review.

We did not test VoxCPM2 in our rig because OmniVoice worked first.
Worth a second look if OmniVoice falls down for a use case not
covered by OmniVoice (e.g. you want a different language entirely).

### ⚠ Qwen2.5-Omni-7B / -3B — UNTESTED

- 7B variant: ~14 GB bf16 weights alone — exceeds 16 GB budget after
  KV cache.
- 3B variant: ~6 GB — borderline. JA listed in Qwen docs, has dedicated
  Talker for voice cloning.
- Not retested because OmniVoice landed first. If you want a single
  model that also does ASR + chat + audio understanding (not just TTS),
  the 3B variant is worth a shot.

### Filtered out

- **MammothModa2-Preview** (Bytedance) — image generation, not TTS.
- **Ming-flash-omni** / **BailingMM2** (Ant) — 100B MoE; JA not
  documented; way over budget.
- **MiMo-Audio-7B** (Xiaomi) — 16 GB bf16; over budget. No JA share
  documented.
- **Voxtral-TTS-4B** (Mistral) — explicitly does NOT support JA.
- **Dynin-Omni** (KAIST) — Korean-focused, no JA mention.
- **VoxCPM-0.5B** (v1) — model card says "Chinese and English only".
- **OmniVoice** is the only vLLM-Omni-native model that survives
  all gates for Japanese on our hardware.

### Considered outside vLLM-Omni (and rejected)

- **Style-Bert-VITS2 JP-Extra** — formerly our JA production winner;
  human-parity MOS on anime-character benchmark. **Non-AR flow
  architecture; can't run cleanly on vLLM-Omni** (no PagedAttention
  benefit, would need a single-stage `is_comprehension` adapter that
  bypasses the whole point of vLLM-Omni). Kept as a fallback option
  outside Docker; not the default.
- **IndexTTS-2** — 42k hrs JA, strongest paper numbers among
  cloning-capable specialists. **Port to vLLM-Omni is multi-week**
  engineering (GPT-2 backbone + embed-prefix inputs + global ODE solve
  in Stage 1 + monkey-patch position handling) — see PARKED design
  notes in `project_indextts_vllm_omni_port`. Reactivate only if
  OmniVoice falls down on a future use case.

## Decision flowchart

```text
                  Need TTS in vLLM-Omni Docker?
                              │
                       Pick a language ─────┬───────────┬──────────┐
                              │             │           │          │
                         Japanese        Chinese     English  Multilingual
                              │             │           │          │
                         OmniVoice    CosyVoice3      ???       OmniVoice
                              │       (--profile      │      (also handles
                              │        cosy3)         │       ZH well)
                              │             │      Not yet
                              │             │      decided.
                       Need character        │      OmniVoice EN
                       voice fidelity?       │      under-tested;
                              │              │      Higgs Audio
                       ┌──────┴───────┐      │      doesn't fit
                       │              │      │      16 GB.
                     Yes              No     │
                       │              │      │
                  16-omnivoice-sft   ship    │
                  -recipe.md         base    │
                  (~20 min character model   │
                   data, 8 min train)
```

## Re-running the eval

If you have a new model candidate or want to verify our results on your
hardware:

```bash
# 1. Bring up the candidate service in vllm-omni-deploy.
cd vllm-omni-deploy
docker compose --profile <name> up -d

# 2. Run the rig.
cd ../vllm-omni-tests
python run_eval.py --endpoint http://127.0.0.1:8000 --whisper-model large-v3

# 3. Inspect.
cat out/report.md
```

The bar to clear: **mean jaccard ≥ 0.95 across the 21-prompt battery**
with no single prompt below 0.60. Below that and you'll ship a TTS that
hallucinates words on inputs the LLM will plausibly produce.

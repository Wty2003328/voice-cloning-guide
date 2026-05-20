# Mandarin Chinese TTS — model recommendations

**Top pick (2026-05):** [**CosyVoice 3 (Fun-CosyVoice3-0.5B-2512)**](../models/cosyvoice-3.md), served via [vLLM-Omni in Docker](../15-vllm-omni-docker.md) with `--profile cosy3`. Apache-2.0, CER 0.81% native ZH, SIM 78% beats human reference 75.5%.

## What makes Chinese TTS hard

- **Tones are lexically distinctive.** 妈/麻/马/骂 (mā/má/mǎ/mà) — same
  syllable, four tones, four meanings. A model that misses tones
  produces fluent gibberish.
- **Polyphonic characters.** 行 = xíng (walk) or háng (row);
  disambiguation needs context.
- **No spaces.** Word segmentation depends on the model's understanding
  of Mandarin morphology.
- **Code-mixing.** Modern Chinese text mixes English loanwords (APP,
  OK, WiFi, API); the model has to switch phoneme systems mid-sentence
  and pronounce acronyms letter-by-letter.

## Candidate comparison

| Model | License | Voice clone | ZH quality | Deploy |
|---|---|---|---|---|
| **CosyVoice 3 (Fun-0.5B-2512)** ✅ | Apache-2.0 | Zero-shot 3–30 s | **Best** — CER 0.81%, SIM 78% (beats human ref 75.5%) | `docker compose --profile cosy3 up` (vLLM-Omni) |
| Qwen3-TTS-12Hz-1.7B-Base | Apache-2.0 | Zero-shot | Decent multilingual baseline | `docker compose --profile qwen up` |
| VoxCPM2 | Apache-2.0 | Zero-shot | 30 langs including ZH; not yet evaluated at depth here | `vllm serve openbmb/VoxCPM2` |
| GPT-SoVITS v4 | MIT | Per-voice fine-tune | Strong ZH (Chinese-community origin) | Legacy LoRA path |
| Spark-TTS | CC-BY-NC-SA | Zero-shot | Strong | Blocked — license flipped from Apache to non-commercial |
| ChatTTS (weights) | CC-BY-NC | Limited | Conversational style | Blocked — non-commercial weights |

## Recommendation: CosyVoice 3

**Why it wins:**

- Apache-2.0 weights (clean commercial use).
- Trained on Alibaba-native ZH data — real Chinese language depth, not
  a multilingual generalist's compromise.
- CER 0.81% on Chinese test sets (industry-leading).
- Speaker SIM 78% — better than the human reference at 75.5%, meaning
  the model's clones are more consistent with the speaker than a
  different recording of the same speaker would be.
- First-class **Cantonese** support via a `<|yue|>` dialect token, plus
  17 other Chinese dialects.

**Known caveat for non-Chinese:** the model is trained on
kana-converted text for Japanese, so raw kanji input degrades JA
content fidelity (mean jaccard 0.37 on a 21-prompt JA battery without a
kanji→kana adapter). Keep it for Chinese; use OmniVoice for Japanese.
See [ch. 15 — Picking a model](../15-vllm-omni-model-selection.md) for
the full eval.

## Deploy via vLLM-Omni Docker

The recommended path is the same one used for every other language: a
single Docker compose with a profile per model.

```yaml
services:
  cosyvoice3:
    image: vllm-omni-cosy-ja:v0.20.0
    profiles: ["cosy3"]
    ports: ["8000:8000"]
    volumes:
      - hf-cache:/root/.cache/huggingface
      - ./refs:/refs:ro
    runtime: nvidia
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    ipc: host
    shm_size: '8gb'
    command: >
      vllm serve FunAudioLLM/Fun-CosyVoice3-0.5B-2512
      --omni --host 0.0.0.0 --port 8000
      --gpu-memory-utilization 0.40
      --enforce-eager
      --max-model-len 2048 --max-num-seqs 1
      --allowed-local-media-path /refs

volumes:
  hf-cache:
```

The `vllm-omni-cosy-ja:v0.20.0` image extends `vllm/vllm-omni:v0.20.0`
with the CosyVoice3 text-frontend dependencies. Build it the same way
the training image is built — see
[ch. 16 — OmniVoice SFT, step 0](../16-omnivoice-sft-recipe.md#0-prereqs)
for the pattern.

Bring up and verify:

```bash
docker compose --profile cosy3 up -d
curl -s http://127.0.0.1:8000/v1/models
# → { "data": [ { "id": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512", ... } ] }
```

## Sending a synth request

Same OpenAI-compatible body as every other model on vLLM-Omni:

```bash
curl -X POST http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d "$(cat <<'JSON'
{
  "model":     "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
  "input":     "你好,世界。请帮我设置一下API。",
  "language":  "Chinese",
  "ref_audio": "data:audio/wav;base64,<base64 of your reference clip>",
  "ref_text":  "<exact transcript of the reference clip>"
}
JSON
)" \
  -o hello.wav
```

For Cantonese: switch `language` to `"Chinese"` and use a Cantonese
reference clip; CosyVoice3 picks up the dialect from the reference.

## Voice cloning

CosyVoice3 is zero-shot. A 5–10 s clean reference clip plus an exact
transcript produces the cloned voice — no GPU training. See
[ch. 10 — Zero-shot cloning](../10-zero-shot-cloning.md) for the
ref-clip recipe (length, cleanliness, prosody-diversity tips).

For character-level timbre fidelity beyond what zero-shot achieves,
fine-tune on ~10–20 minutes of target audio. The
[OmniVoice SFT recipe](../16-omnivoice-sft-recipe.md) is JA-focused
but the dataset-prep and training-config pattern transfers to CosyVoice3
with minor changes (different config schema; same Higgs codec / WebDataset
pipeline).

## Appendix: lowest-latency path (TensorRT-LLM)

If you need lower latency than the default vLLM-Omni path (RTF ~0.5 on
RTX 5080), CosyVoice3 also has a TensorRT-LLM build that runs in WSL2
with a custom plugin. Reported numbers: **RTF ~0.10 on RTX 5080** —
roughly 5× faster than vLLM-Omni eager mode for the same model.

Trade-off: 5–7 days of integration work (custom plugin build, WSL2 GPU
passthrough, separate from the Docker compose). The vLLM-Omni Docker
path is the recommended starting point; reach for TensorRT-LLM only if
the latency floor matters for your specific use case.

## See also

- [../15-vllm-omni-docker.md](../15-vllm-omni-docker.md) — production
  deploy walkthrough.
- [../15-vllm-omni-model-selection.md](../15-vllm-omni-model-selection.md)
  — full per-model eval.
- [../10-zero-shot-cloning.md](../10-zero-shot-cloning.md) —
  reference-clip tuning.
- [../models/cosyvoice-3.md](../models/cosyvoice-3.md) — CosyVoice 3
  deep dive.
- [multilingual.md](multilingual.md) — when one model should serve ZH
  + EN + JA.

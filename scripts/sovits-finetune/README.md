# Legacy GPT-SoVITS v4 fine-tune pipeline

Moved here from `scripts/` in the 2026-05 voice-cloning-guide reorg.

This is the **pre-vLLM-Omni** path: an end-to-end LoRA fine-tune of
GPT-SoVITS v4 (Apache-licensed, MIT-licensed v4 modifications) using
~10-20 minutes of target-voice audio. Wallclock ~30-60 min on an
RTX 5080.

For new projects we recommend the
[OmniVoice SFT recipe](../../docs/16-omnivoice-sft-recipe.md) instead
— same data requirement, ~8 min training wallclock, and the output
drops into our production `vllm-omni-deploy` docker compose
unchanged.

This pipeline is kept for:
- Users who already have a GPT-SoVITS v4 weights set and want to
  iterate on it.
- Use cases where GPT-SoVITS v4's specific prosody-learning behaviour
  beats OmniVoice (rare; not seen empirically by us in 2026).
- Reference for the dataset-prep stages (slicing + ASR + feature
  extraction) which are reusable for any TTS fine-tune.

## Pipeline overview

```
raw video / podcast audio
       │
       │  demucs_isolate.py        (vocal isolation)
       ▼
   speaker_vocals.wav
       │
       │  01_slice_audio.py        (sentence-level slicing → audio/0001.wav, …)
       ▼
   experiments/<name>/0_sliced/
       │
       │  02_asr_transcribe.py     (faster-whisper transcribe → transcripts.list)
       ▼
   transcripts.list
       │
       │  03_extract_features.py   (HuBERT semantic + voice features)
       │  04_extract_semantic.py
       ▼
   experiments/<name>/2_feats/
       │
       │  05_train_sovits_v4.py    (LoRA fine-tune of SoVITS v4)
       │  06_train_gpt.py          (LoRA fine-tune of GPT layer)
       ▼
   weights/<name>/
       │
       │  07_inference_v4.py       (synth)
       ▼
   output.wav
```

## Quickstart

```bash
# 1. Prereqs: clone GPT-SoVITS upstream + install deps.
git clone https://github.com/RVC-Boss/GPT-SoVITS
cd GPT-SoVITS && pip install -r requirements.txt && cd ..

# 2. Install this script set's deps.
pip install -r scripts/sovits-finetune/requirements.txt

# 3. Run the pipeline.
cd scripts/sovits-finetune
python demucs_isolate.py     --input raw_video_audio.wav --output speaker_vocals.wav
python 01_slice_audio.py     --vocals ../../speaker_vocals.wav --exp my_speaker
python 02_asr_transcribe.py  --exp my_speaker --lang ja
python 03_extract_features.py --exp my_speaker
python 04_extract_semantic.py --exp my_speaker
python 05_train_sovits_v4.py --exp my_speaker --epochs 20 --lora-rank 32
python 06_train_gpt.py       --exp my_speaker --epochs 15 --pretrained-version v4
python 07_inference_v4.py    --exp my_speaker --lang ja --text "..." \
    --ref-wav ../../GPT-SoVITS/logs/my_speaker/0_sliced/0003.wav \
    --ref-text "ここは私に任せて私を選んでくれる" --ref-lang ja \
    --out hello.wav
```

Full walkthrough: [`docs/models/gpt-sovits-v4.md`](../../docs/models/gpt-sovits-v4.md).

## Files

| Script | Purpose |
|---|---|
| `_common.py` | Shared helpers (paths, logging) — imported by the other scripts. |
| `demucs_isolate.py` | Vocal isolation from mixed audio via Demucs. |
| `01_slice_audio.py` | Sentence-level slicing of vocal-only audio. |
| `02_asr_transcribe.py` | Faster-whisper ASR over slices → transcripts.list. |
| `03_extract_features.py` | HuBERT/wav2vec semantic + voice features. |
| `04_extract_semantic.py` | Semantic-token extraction for SoVITS conditioning. |
| `05_train_sovits_v4.py` | LoRA fine-tune of the SoVITS v4 stage. |
| `06_train_gpt.py` | LoRA fine-tune of the GPT (semantic-token AR) layer. |
| `07_inference_v4.py` | Synthesize using the fine-tuned weights. |
| `build_reference.py` | Pick "best" reference clip from a slice pool for inference. |
| `requirements.txt` | Pip deps for this script set. |

## Migrating to OmniVoice SFT

If you've got a v4 fine-tune working and want to compare against the
new pipeline:

1. Reuse the **sliced audio + transcripts** from
   `experiments/<name>/0_sliced/` — they're the same input shape
   OmniVoice SFT needs.
2. Build the JSONL manifest as described in
   [`docs/16-omnivoice-sft-recipe.md` step 2](../../docs/16-omnivoice-sft-recipe.md#2-build-jsonl-manifests).
3. Run the 8-minute SFT.
4. A/B both models on the same prompts. If GPT-SoVITS wins
   subjectively, keep it; if OmniVoice wins or ties, switch (OmniVoice
   has the docker deploy advantage).

Try both and ship the one that sounds closer to your target.

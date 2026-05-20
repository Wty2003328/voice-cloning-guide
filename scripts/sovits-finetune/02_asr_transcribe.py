"""Step 2: Transcribe each slice with faster-whisper, then phonemize the text.

Produces two files:
- logs/<exp>/asr.list — pipe-separated raw ASR (wav|speaker|lang|text)
- logs/<exp>/2-name2text.txt — TAB-separated phonemized form (wav\\tphonemes\\tspeaker\\tlang)
  This file is the canonical training input downstream.

Note: faster-whisper's CTranslate2 backend often clashes with newer CUDA. We force CPU
(int8) by default — slightly slower but reliable across Windows/Linux.

Example:
    python 02_asr_transcribe.py --exp my_speaker --lang ja
"""
import argparse
from pathlib import Path

from _common import setup, pretrained_paths


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--exp", required=True)
    p.add_argument("--lang", default="ja", choices=["ja", "en", "zh"],
                   help="Source language for ASR + phonemizer")
    p.add_argument("--speaker", default="speaker", help="Speaker tag in output files")
    p.add_argument("--gs-dir", default="./GPT-SoVITS")
    p.add_argument("--whisper-model", default="large-v3")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                   help="CTranslate2 device (CPU is most compatible)")
    args = p.parse_args()

    gs_dir = setup(Path(args.gs_dir))
    exp_dir = gs_dir / "logs" / args.exp
    slices = sorted((exp_dir / "0_sliced").glob("*.wav"))
    if not slices:
        raise SystemExit(f"No slices in {exp_dir}/0_sliced — run 01_slice_audio.py first")

    from faster_whisper import WhisperModel
    model_path = gs_dir / "tools" / "asr" / "models" / f"faster-whisper-{args.whisper_model}"
    if not model_path.exists():
        from huggingface_hub import snapshot_download
        snapshot_download(f"Systran/faster-whisper-{args.whisper_model}",
                          local_dir=str(model_path))

    model = WhisperModel(str(model_path), device=args.device,
                         compute_type="int8" if args.device == "cpu" else "float16")

    asr_results = []
    for i, wav in enumerate(slices):
        segments, _ = model.transcribe(
            str(wav), language=args.lang, beam_size=5, vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=200),
        )
        text = "".join(s.text for s in segments).strip()
        if text:
            asr_results.append((wav.name, text))
        if (i + 1) % 10 == 0:
            print(f"  Transcribed {i+1}/{len(slices)}")

    list_path = exp_dir / "asr.list"
    with open(list_path, "w", encoding="utf-8") as f:
        for wav_name, text in asr_results:
            f.write(f"{wav_name}|{args.speaker}|{args.lang}|{text}\n")

    from GPT_SoVITS.text.cleaner import clean_text
    name2text = exp_dir / "2-name2text.txt"
    skipped = 0
    with open(name2text, "w", encoding="utf-8") as f:
        for wav_name, text in asr_results:
            try:
                phones, _, _ = clean_text(text, args.lang, "v2")
                f.write(f"{wav_name}\t{' '.join(phones)}\t{args.speaker}\t{args.lang}\n")
            except Exception as e:
                print(f"  Phonemize failed for {wav_name}: {e}")
                skipped += 1

    print(f"Saved {len(asr_results) - skipped} entries to {name2text} (skipped {skipped})")


if __name__ == "__main__":
    main()

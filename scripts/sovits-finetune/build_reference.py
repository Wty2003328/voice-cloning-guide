"""Build a prosody-diverse multi-clip reference for zero-shot cloning.

Concatenates N short clips (typically 5) of the same speaker into one
~20-32s reference. Each clip is loudness-normalized and silence-trimmed
before concat. Output: <out_dir>/concat.wav + <out_dir>/concat.txt
(the joined transcript) ready to pass to zero_shot_clone.py.

The selection strategy: you provide the clips you want (and their
transcripts). For automatic clip scoring from a pool, see the more
elaborate `build_best_reference.py` example in the waifu-companion
tts_lab/.

Usage:
    python build_reference.py \\
        --clip clip1.wav "transcript of clip 1" \\
        --clip clip2.wav "transcript of clip 2" \\
        --clip clip3.wav "transcript of clip 3" \\
        --out-dir ./reference_clips/my_voice/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

TARGET_PEAK = 0.8
SILENCE_GAP_S = 0.3
SILENCE_DB = -40.0
SILENCE_THRESHOLD = 10 ** (SILENCE_DB / 20.0)


def trim_silence(arr: np.ndarray, sr: int) -> np.ndarray:
    """Trim leading/trailing silence from a mono signal."""
    win = max(80, sr // 100)  # 10ms windows
    starts = np.arange(0, len(arr) - win, win)
    if len(starts) == 0:
        return arr
    win_rms = np.array([np.sqrt(np.mean(arr[i:i+win] ** 2) + 1e-12)
                        for i in starts])
    voiced = np.where(win_rms > SILENCE_THRESHOLD)[0]
    if len(voiced) == 0:
        return arr
    first = max(0, voiced[0] * win - sr // 50)   # 20ms padding
    last = min(len(arr), (voiced[-1] + 1) * win + sr // 50)
    return arr[first:last]


def normalize_peak(arr: np.ndarray, target: float = TARGET_PEAK) -> np.ndarray:
    pk = float(np.abs(arr).max())
    return arr * (target / pk) if pk > 1e-6 else arr


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clip", action="append", nargs=2,
                   metavar=("WAV", "TRANSCRIPT"),
                   help="Pass once per clip: --clip path.wav 'transcript'. "
                        "Repeat for each. 5 clips around 4-7s each is the sweet spot.")
    p.add_argument("--out-dir", required=True,
                   help="Output directory (concat.wav + concat.txt written here)")
    p.add_argument("--gap", type=float, default=SILENCE_GAP_S,
                   help=f"Silence between clips, seconds (default {SILENCE_GAP_S})")
    p.add_argument("--max-duration", type=float, default=32.0,
                   help="Soft ceiling; the model has issues above ~32s")
    args = p.parse_args()

    if not args.clip or len(args.clip) < 1:
        print("ERROR: at least one --clip needed", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Building reference from {len(args.clip)} clips ===")
    waves, transcripts = [], []
    sr_ref = None

    for wav_path_s, transcript in args.clip:
        wav_path = Path(wav_path_s)
        if not wav_path.exists():
            print(f"ERROR: clip not found: {wav_path}", file=sys.stderr)
            return 2
        arr, sr = sf.read(wav_path)
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        arr = arr.astype(np.float32)
        if sr_ref is None:
            sr_ref = sr
        elif sr != sr_ref:
            print(f"WARN: {wav_path.name} sr={sr} ≠ first clip's {sr_ref}, "
                  f"resampling needed (skipping)", file=sys.stderr)
            continue
        trimmed = trim_silence(arr, sr)
        normed = normalize_peak(trimmed)
        if waves:
            waves.append(np.zeros(int(args.gap * sr_ref), dtype=np.float32))
        waves.append(normed)
        transcripts.append(transcript)
        print(f"  + {wav_path.name}: {len(normed)/sr:.1f}s "
              f"(orig {len(arr)/sr:.1f}s)  '{transcript[:50]}'")

    combined = np.concatenate(waves)
    duration = len(combined) / sr_ref
    if duration > args.max_duration:
        print(f"WARN: reference duration {duration:.1f}s exceeds "
              f"--max-duration={args.max_duration}s; AR hallucinations "
              f"are common above this. Drop a clip or lower --gap.",
              file=sys.stderr)

    out_wav = out_dir / "concat.wav"
    out_txt = out_dir / "concat.txt"
    sf.write(out_wav, combined, sr_ref)
    joined_tx = " ".join(transcripts)
    out_txt.write_text(joined_tx, encoding="utf-8")
    print(f"\nwrote {out_wav} ({duration:.1f}s @ {sr_ref} Hz)")
    print(f"wrote {out_txt} (joined transcript)")
    print(f"\nUse with zero_shot_clone.py:")
    print(f"  --reference {out_wav}")
    print(f"  --reference-text \"$(cat {out_txt})\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())

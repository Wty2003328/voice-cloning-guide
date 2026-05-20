"""Step 1: Slice a long vocal recording into 3-15s segments using VAD-aware silence detection.

The slicer (from RVC-Boss/GPT-SoVITS tools/) detects silences below a -40 dB threshold and
splits at those boundaries, producing chunks the downstream models can train on. Output is
32 kHz mono PCM at logs/<exp>/0_sliced/0001.wav, 0002.wav, ...

Why this matters: GPT-SoVITS expects ~3-15 second utterances at 32 kHz. Files outside that
range either get padded (wasting capacity) or truncated (losing prosody). The slicer
respects sentence boundaries via silence, which keeps phoneme-aligned later.

Example:
    python 01_slice_audio.py --vocals my_speaker_vocals.wav --exp my_speaker
"""
import argparse
import glob
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from _common import setup


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--vocals", required=True, help="Input vocals .wav (Demucs-isolated, any sr)")
    p.add_argument("--exp", required=True, help="Experiment name (subdir under logs/)")
    p.add_argument("--gs-dir", default="./GPT-SoVITS", help="Path to GPT-SoVITS repo")
    p.add_argument("--threshold", type=float, default=-40, help="Silence threshold (dB)")
    p.add_argument("--min-length-ms", type=int, default=4000, help="Min slice length (ms)")
    p.add_argument("--min-interval-ms", type=int, default=300, help="Min silence interval (ms)")
    args = p.parse_args()

    gs_dir = setup(Path(args.gs_dir))
    exp_dir = gs_dir / "logs" / args.exp
    out_dir = exp_dir / "0_sliced"
    out_dir.mkdir(parents=True, exist_ok=True)

    import sys
    sys.path.insert(0, str(gs_dir / "tools"))
    from slicer2 import Slicer
    from tools.my_utils import load_audio

    slicer = Slicer(
        sr=32000,
        threshold=args.threshold,
        min_length=args.min_length_ms,
        min_interval=args.min_interval_ms,
        hop_size=10,
        max_sil_kept=500,
    )
    audio = load_audio(args.vocals, 32000)

    # Soft normalization preserving relative dynamics — copied from GPT-SoVITS pipeline
    _max, alpha = 0.9, 0.25
    idx = 0
    for chunk, _, _ in slicer.slice(audio):
        m = np.abs(chunk).max()
        if m > 1:
            chunk /= m
        chunk = (chunk / m * (_max * alpha)) + (1 - alpha) * chunk
        wavfile.write(str(out_dir / f"{idx:04d}.wav"), 32000,
                      (chunk * 32767).astype("int16"))
        idx += 1

    n = len(glob.glob(str(out_dir / "*.wav")))
    print(f"Generated {n} slices in {out_dir}")


if __name__ == "__main__":
    main()

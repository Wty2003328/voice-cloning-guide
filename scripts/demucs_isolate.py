"""Step 0 (optional): Isolate vocals from a mixed-source audio file using Demucs.

Bypasses Demucs's built-in audio loader (which depends on torchcodec — frequently broken
on Windows) by feeding a pre-loaded tensor through `apply_model` directly. Reads any
sound-file-supported format, writes mono 44.1 kHz vocals.

Example:
    python demucs_isolate.py --input my_video_audio.wav --output my_speaker_vocals.wav
"""
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--input", required=True, help="Mixed audio (.wav, decoded from your video)")
    p.add_argument("--output", required=True, help="Output vocals .wav")
    p.add_argument("--model", default="htdemucs", choices=["htdemucs", "htdemucs_ft", "mdx_extra"])
    p.add_argument("--device", default=None, help="cuda or cpu (auto-detect default)")
    args = p.parse_args()

    from demucs.pretrained import get_model
    from demucs.apply import apply_model

    data, sr = sf.read(args.input)
    if data.ndim == 1:
        data = np.stack([data, data], axis=-1)
    wav = torch.from_numpy(data.T).float()

    model = get_model(args.model)
    model.cpu().eval()
    if sr != model.samplerate:
        import torchaudio
        wav = torchaudio.functional.resample(wav, sr, model.samplerate)
        sr = model.samplerate

    ref = wav.mean(0)
    wav_norm = (wav - ref.mean()) / ref.std()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    sources = apply_model(model, wav_norm[None], device=device,
                          shifts=1, split=True, overlap=0.25, progress=True)[0]
    sources = sources * ref.std() + ref.mean()

    vocals = dict(zip(model.sources, sources))["vocals"]
    # Mix to mono for downstream training (single-channel = simpler memory + fewer artifacts)
    vocals_mono = vocals.mean(0).cpu().numpy()
    sf.write(args.output, vocals_mono, sr, subtype="PCM_16")
    print(f"Wrote {args.output} ({len(vocals_mono)/sr:.1f}s mono @ {sr}Hz)")


if __name__ == "__main__":
    main()

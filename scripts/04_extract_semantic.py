"""Step 4: Quantize HuBERT features into discrete semantic tokens (vocab=1024 @ 25 Hz).

The pretrained SoVITS generator already contains a vector-quantizer that maps the
continuous 768-dim HuBERT space into 1024 codebook entries. We use it without modification
— the GPT model later predicts these codes autoregressively from text.

Output: logs/<exp>/6-name2semantic.tsv (TAB-separated: <wav>\\t<token1 token2 ...>)
"""
import argparse
from pathlib import Path

import torch

from _common import setup, pretrained_paths


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--exp", required=True)
    p.add_argument("--gs-dir", default="./GPT-SoVITS")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--no-fp16", action="store_true")
    args = p.parse_args()

    gs_dir = setup(Path(args.gs_dir))
    paths = pretrained_paths(gs_dir)
    exp_dir = gs_dir / "logs" / args.exp
    DEVICE = torch.device(args.device)
    is_half = not args.no_fp16

    import GPT_SoVITS.utils as utils
    from GPT_SoVITS.module.models import SynthesizerTrn

    hps = utils.get_hparams_from_file(paths["s2_config"])
    vq = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        version="v2",
        **hps.model,
    )
    vq = (vq.half() if is_half else vq).to(DEVICE).eval()
    vq.load_state_dict(
        torch.load(paths["s2g"], map_location="cpu", weights_only=False)["weight"],
        strict=False,
    )

    hubert_dir = exp_dir / "4-cnhubert"
    list_path = exp_dir / "2-name2text.txt"
    if not list_path.exists():
        raise SystemExit(f"{list_path} missing — run prior steps first")
    lines = list_path.read_text(encoding="utf-8").strip().split("\n")

    out_lines = []
    for line in lines:
        parts = line.split("\t")
        if not parts:
            continue
        wav_name = parts[0]
        ssl_path = hubert_dir / f"{wav_name}.pt"
        if not ssl_path.exists():
            continue
        ssl = torch.load(str(ssl_path), map_location="cpu")
        ssl = (ssl.half() if is_half else ssl).to(DEVICE)
        with torch.no_grad():
            codes = vq.extract_latent(ssl)
        out_lines.append(f"{wav_name}\t{' '.join(str(i) for i in codes[0, 0, :].tolist())}")

    out_tsv = exp_dir / "6-name2semantic.tsv"
    out_tsv.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"Wrote {len(out_lines)} entries to {out_tsv}")


if __name__ == "__main__":
    main()

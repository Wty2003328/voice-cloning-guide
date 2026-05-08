"""Step 3: Extract HuBERT (audio) and BERT (text) features.

HuBERT (CNHuBERT, 768-dim @ 20ms): self-supervised speech representation. Each slice
becomes a feature tensor saved to 4-cnhubert/<wav>.pt.

BERT (Chinese-RoBERTa-wwm-ext-large, 1024-dim, last-3 hidden state): used only for
Chinese text — provides contextual cues that help the GPT predict prosody. For ja/en,
zero tensors are written at inference time, so we skip BERT extraction here for non-zh.

Side effect: writes a normalized 32 kHz copy of each slice to 5-wav32k/<wav> — required
because the SoVITS data loader reads from this dir, not 0_sliced.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile

from _common import setup, pretrained_paths


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--exp", required=True)
    p.add_argument("--gs-dir", default="./GPT-SoVITS")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--no-fp16", action="store_true", help="Disable half precision")
    args = p.parse_args()

    gs_dir = setup(Path(args.gs_dir))
    exp_dir = gs_dir / "logs" / args.exp
    paths = pretrained_paths(gs_dir)
    DEVICE = torch.device(args.device)
    is_half = not args.no_fp16

    # ---- HuBERT ----
    from GPT_SoVITS.feature_extractor import cnhubert
    cnhubert.cnhubert_base_path = paths["cnhubert"]
    hubert = cnhubert.get_model()
    hubert = (hubert.half() if is_half else hubert).to(DEVICE).eval()

    hubert_dir = exp_dir / "4-cnhubert"
    wav32k_dir = exp_dir / "5-wav32k"
    hubert_dir.mkdir(parents=True, exist_ok=True)
    wav32k_dir.mkdir(parents=True, exist_ok=True)

    list_path = exp_dir / "2-name2text.txt"
    if not list_path.exists():
        raise SystemExit(f"{list_path} missing — run 02_asr_transcribe.py first")
    lines = list_path.read_text(encoding="utf-8").strip().split("\n")
    slices_dir = exp_dir / "0_sliced"

    import librosa
    from tools.my_utils import load_audio

    maxx, alpha = 0.95, 0.5
    done = 0
    for line in lines:
        parts = line.split("\t")
        if not parts:
            continue
        wav_name = parts[0]
        out_pt = hubert_dir / f"{wav_name}.pt"
        if out_pt.exists():
            done += 1
            continue
        wav_path = slices_dir / wav_name
        if not wav_path.exists():
            continue
        audio = load_audio(str(wav_path), 32000)
        m = float(np.abs(audio).max())
        if m > 2.2 or m < 0.01:
            continue
        # Normalize for SoVITS training (saved to 5-wav32k)
        audio32 = (audio / m * (maxx * alpha * 32768)) + (1 - alpha) * 32768 * audio
        # Same scaling at 16k for HuBERT input
        audio16 = librosa.resample(
            (audio / m * (maxx * alpha * 1145.14)) + (1 - alpha) * 1145.14 * audio,
            orig_sr=32000, target_sr=16000,
        )
        t16 = torch.from_numpy(audio16)
        t16 = (t16.half() if is_half else t16).to(DEVICE)
        with torch.no_grad():
            ssl = hubert.model(t16.unsqueeze(0))["last_hidden_state"].transpose(1, 2).cpu()
        if np.isnan(ssl.numpy()).any():
            print(f"  NaN in {wav_name}, skipping")
            continue
        wavfile.write(str(wav32k_dir / wav_name), 32000, audio32.astype("int16"))
        torch.save(ssl, str(out_pt))
        done += 1
        if done % 20 == 0:
            print(f"  HuBERT {done}/{len(lines)}")
    print(f"HuBERT done: {done} files")

    # ---- BERT (zh only) ----
    bert_dir = exp_dir / "3-bert"
    bert_dir.mkdir(parents=True, exist_ok=True)
    has_zh = any(line.split("\t")[3] == "zh" for line in lines if len(line.split("\t")) >= 4)
    if not has_zh:
        print("No zh entries — skipping BERT extraction (zeros are produced at inference)")
        return

    from transformers import AutoModelForMaskedLM, AutoTokenizer
    from GPT_SoVITS.text.cleaner import clean_text
    tok = AutoTokenizer.from_pretrained(paths["bert"])
    bert = AutoModelForMaskedLM.from_pretrained(paths["bert"]).half().to(DEVICE).eval()

    def bert_feature(text, word2ph):
        with torch.no_grad():
            inp = {k: v.to(DEVICE) for k, v in tok(text, return_tensors="pt").items()}
            out = bert(**inp, output_hidden_states=True)
            res = torch.cat(out["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
        feats = [res[i].repeat(word2ph[i], 1) for i in range(len(word2ph))]
        return torch.cat(feats, dim=0).T

    n = 0
    for line in lines:
        parts = line.split("\t")
        if len(parts) < 4 or parts[3] != "zh":
            continue
        wav_name, raw_text = parts[0], parts[2]
        out_pt = bert_dir / f"{wav_name}.pt"
        if out_pt.exists():
            n += 1
            continue
        try:
            phones, w2p, norm = clean_text(
                raw_text.replace("%", "-").replace("￥", ","), "zh", "v2")
            torch.save(bert_feature(norm, w2p), str(out_pt))
            n += 1
        except Exception as e:
            print(f"  BERT fail {wav_name}: {e}")
    print(f"BERT done: {n} zh files")


if __name__ == "__main__":
    main()

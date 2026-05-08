"""Step 7: Generate speech from text using fine-tuned SoVITS + GPT.

Pipeline (per request): text → phonemes → BERT (zh only) → GPT (autoregressive,
seeded with reference semantic prefix) → semantic tokens → SoVITS decode → 32 kHz wav.

The reference clip provides two things: the *semantic prefix* (first 50 tokens) which
biases GPT into the speaker's prosody distribution, and the *mel spec* which the
SoVITS posterior encoder uses to pin down speaker identity (timbre).
"""
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from _common import setup, pretrained_paths


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--exp", required=True, help="Experiment name (used to find checkpoints)")
    p.add_argument("--text", required=True, help="Text to synthesize")
    p.add_argument("--lang", required=True, choices=["ja", "en", "zh"])
    p.add_argument("--ref-wav", required=True, help="Reference audio for the speaker")
    p.add_argument("--ref-text", required=True, help="Transcript of the reference audio")
    p.add_argument("--ref-lang", required=True, choices=["ja", "en", "zh"])
    p.add_argument("--out", required=True, help="Output .wav path")
    p.add_argument("--gs-dir", default="./GPT-SoVITS")
    p.add_argument("--top-k", type=int, default=15)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--repetition-penalty", type=float, default=1.35)
    p.add_argument("--sovits-ckpt", default=None,
                   help="Override SoVITS checkpoint path (default: latest <exp>_e*.pth)")
    p.add_argument("--gpt-ckpt", default=None,
                   help="Override GPT checkpoint path (default: latest <exp>-e*.ckpt)")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    gs_dir = setup(Path(args.gs_dir))
    paths = pretrained_paths(gs_dir)
    DEVICE = torch.device(args.device)

    import nltk
    for pkg in ["averaged_perceptron_tagger_eng", "cmudict", "averaged_perceptron_tagger"]:
        try:
            nltk.data.find(f"taggers/{pkg}" if "tagger" in pkg else f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)

    # ---- Models ----
    from GPT_SoVITS.feature_extractor import cnhubert
    cnhubert.cnhubert_base_path = paths["cnhubert"]
    hubert = cnhubert.get_model().half().to(DEVICE).eval()

    from transformers import AutoModelForMaskedLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(paths["bert"])
    bert = AutoModelForMaskedLM.from_pretrained(paths["bert"]).half().to(DEVICE).eval()

    import GPT_SoVITS.utils as utils
    from GPT_SoVITS.module.models import SynthesizerTrn
    from GPT_SoVITS.AR.models.t2s_lightning_module import Text2SemanticLightningModule
    hps = utils.get_hparams_from_file(paths["s2_config"])

    sovits_ckpt = args.sovits_ckpt or _latest(
        gs_dir / "SoVITS_weights_v2", f"{args.exp}_e*.pth",
        lambda p: int(p.stem.split("_e")[1].split("_")[0]),
    )
    gpt_ckpt = args.gpt_ckpt or _latest(
        gs_dir / "GPT_weights_v2", f"{args.exp}-e*.ckpt",
        lambda p: int(p.stem.split("-e")[1]),
    )
    print(f"SoVITS: {sovits_ckpt}\nGPT:    {gpt_ckpt}")

    vits = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        version="v2",
        **hps.model,
    ).half().to(DEVICE).eval()
    vits.load_state_dict(
        torch.load(sovits_ckpt, map_location="cpu", weights_only=False)["weight"],
        strict=False,
    )

    s1config = {
        "data": {"max_sec": 54, "pad_val": 1024},
        "model": {
            "vocab_size": 1025, "phoneme_vocab_size": 732,
            "embedding_dim": 512, "hidden_dim": 512, "head": 16,
            "linear_units": 2048, "n_layer": 24, "dropout": 0,
            "EOS": 1024, "random_bert": 0,
        },
    }
    gpt = Text2SemanticLightningModule(s1config, Path("."), is_train=False)
    gpt.load_state_dict(
        torch.load(gpt_ckpt, map_location="cpu", weights_only=False)["weight"],
        strict=False,
    )
    gpt = gpt.half().to(DEVICE).eval()
    gpt.model.infer_panel = gpt.model.infer_panel_naive

    # ---- Reference audio ----
    from GPT_SoVITS.text.cleaner import clean_text
    from GPT_SoVITS.text import cleaned_text_to_sequence
    from GPT_SoVITS.module.mel_processing import spectrogram_torch
    from tools.my_utils import load_audio
    import librosa

    def get_ssl(wav_path):
        audio = load_audio(wav_path, 32000)
        audio16 = librosa.resample(audio, orig_sr=32000, target_sr=16000).astype(np.float32)
        t = torch.from_numpy(audio16).half().to(DEVICE)
        with torch.no_grad():
            out = hubert.model(t.unsqueeze(0))["last_hidden_state"].transpose(1, 2)
        return out

    def get_phoneme_ids(text, lang):
        phones, w2p, norm = clean_text(text, lang, "v2")
        return cleaned_text_to_sequence(phones, "v2"), w2p, norm

    def get_bert(phone_ids, w2p, norm, lang):
        if lang != "zh":
            return torch.zeros((1024, len(phone_ids)), dtype=torch.float32)
        with torch.no_grad():
            inp = {k: v.to(DEVICE) for k, v in tok(norm, return_tensors="pt").items()}
            out = bert(**inp, output_hidden_states=True)
            res = torch.cat(out["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
        feats = [res[i].repeat(w2p[i], 1) for i in range(len(w2p))]
        return torch.cat(feats, dim=0).T

    ref_ssl = get_ssl(args.ref_wav)
    with torch.no_grad():
        ref_codes = vits.extract_latent(ref_ssl)
    ref_semantic = ref_codes[0, 0, :]

    audio = load_audio(args.ref_wav, hps.data.sampling_rate)
    ref_spec = spectrogram_torch(
        torch.FloatTensor(audio).unsqueeze(0),
        hps.data.filter_length, hps.data.sampling_rate,
        hps.data.hop_length, hps.data.win_length, center=False,
    )

    # ---- Inference ----
    phone_ids, w2p, norm = get_phoneme_ids(args.text, args.lang)
    bert_feat = get_bert(phone_ids, w2p, norm, args.lang)

    all_phone_ids = torch.LongTensor(phone_ids).unsqueeze(0).to(DEVICE)
    all_phone_lens = torch.LongTensor([len(phone_ids)]).to(DEVICE)
    all_bert = bert_feat.half().unsqueeze(0).to(DEVICE)
    prompt_sem = ref_semantic[: min(50, ref_semantic.shape[0])].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        gen = gpt.model.infer_panel(
            all_phone_ids, all_phone_lens, prompt_sem, all_bert,
            top_k=args.top_k, top_p=1, temperature=args.temperature,
            early_stop_num=hps.data.sampling_rate // hps.data.hop_length * 54,
        )
        y, idx = next(gen)
    pred_sem = y[0, -idx:].unsqueeze(0).unsqueeze(0).to(DEVICE)
    print(f"Generated {idx} semantic tokens")

    with torch.no_grad():
        wav = vits.decode(
            pred_sem, all_phone_ids, [ref_spec.half().to(DEVICE)], speed=1.0,
        ).detach().cpu().float()[0, 0].numpy()
    sf.write(args.out, wav, hps.data.sampling_rate)
    print(f"Wrote {args.out} ({len(wav) / hps.data.sampling_rate:.2f}s)")


def _latest(d, pattern, key):
    files = sorted(Path(d).glob(pattern), key=key)
    if not files:
        raise SystemExit(f"No checkpoints matching {pattern} in {d}")
    return str(files[-1])


if __name__ == "__main__":
    main()

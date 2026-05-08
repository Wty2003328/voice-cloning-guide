"""Step 7 (v4): GPT (s1v3-base) → semantic → SoVITS DiT → mel → vocoder.pth → 48kHz audio.

The v4 inference path has three stages instead of v2's two:
  1. GPT predicts semantic tokens (same as v2).
  2. SoVITS DiT runs CFM inference (32 steps default), producing a 100-channel mel.
  3. Separate vocoder.pth (48kHz HiFi-GAN-style) decodes the mel into a waveform.

The CFM step is iterative — sample_steps=32 means 32 denoising steps. Lower steps
(4, 8) work for zero-shot but reduce quality on small fine-tunes.
"""
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from _common import setup, pretrained_paths


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--exp", required=True)
    p.add_argument("--text", required=True)
    p.add_argument("--lang", required=True, choices=["ja", "en", "zh"])
    p.add_argument("--ref-wav", required=True)
    p.add_argument("--ref-text", required=True)
    p.add_argument("--ref-lang", required=True, choices=["ja", "en", "zh"])
    p.add_argument("--out", required=True)
    p.add_argument("--gs-dir", default="./GPT-SoVITS")
    p.add_argument("--top-k", type=int, default=15)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--sample-steps", type=int, default=32,
                   help="CFM denoising steps. 32=best quality, 8=fast, 4=zero-shot")
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--sovits-ckpt", default=None)
    p.add_argument("--gpt-ckpt", default=None)
    args = p.parse_args()

    gs_dir = setup(Path(args.gs_dir), version="v4")
    paths = pretrained_paths(gs_dir, version="v4")
    DEVICE = torch.device("cuda:0")

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
    from GPT_SoVITS.module.models import SynthesizerTrnV3, Generator
    from GPT_SoVITS.AR.models.t2s_lightning_module import Text2SemanticLightningModule
    from GPT_SoVITS.module.mel_processing import mel_spectrogram_torch, spectrogram_torch
    from peft import LoraConfig, get_peft_model

    hps = utils.get_hparams_from_file(paths["s2_config"])
    hps.model.version = "v4"

    sovits_ckpt = args.sovits_ckpt or _latest(
        gs_dir / "SoVITS_weights_v4", f"{args.exp}_e*.pth",
        lambda p: int(p.stem.split("_e")[1].split("_")[0]),
    )
    gpt_ckpt = args.gpt_ckpt or _latest(
        gs_dir / "GPT_weights_v3", f"{args.exp}-e*.ckpt",
        lambda p: int(p.stem.split("-e")[1]),
    )
    print(f"SoVITS v4: {sovits_ckpt}\nGPT (v3-base): {gpt_ckpt}")

    # SoVITS: load v4 base, wrap with LoRA, load fine-tuned, merge for inference
    vits = SynthesizerTrnV3(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    )
    vits.load_state_dict(
        torch.load(paths["s2g"], map_location="cpu", weights_only=False)["weight"],
        strict=False,
    )
    lora_config = LoraConfig(
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        r=args.lora_rank, lora_alpha=args.lora_rank, init_lora_weights=True,
    )
    vits.cfm = get_peft_model(vits.cfm, lora_config)
    vits.load_state_dict(
        torch.load(sovits_ckpt, map_location="cpu", weights_only=False)["weight"],
        strict=False,
    )
    vits.cfm = vits.cfm.merge_and_unload()
    vits = vits.half().to(DEVICE).eval()

    # 48kHz vocoder
    vocoder = Generator(
        initial_channel=100, resblock="1",
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_rates=[10, 6, 2, 2, 2],
        upsample_initial_channel=512,
        upsample_kernel_sizes=[20, 12, 4, 4, 4],
        gin_channels=0, is_bias=True,
    )
    vocoder.remove_weight_norm()
    vocoder.load_state_dict(torch.load(paths["vocoder"], map_location="cpu", weights_only=False))
    vocoder = vocoder.half().to(DEVICE).eval()

    # GPT (architecture identical to v2; weights from v4-style fine-tune)
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

    # ---- Helpers ----
    from GPT_SoVITS.text.cleaner import clean_text
    from GPT_SoVITS.text import cleaned_text_to_sequence
    from tools.my_utils import load_audio
    import librosa

    SPEC_MIN, SPEC_MAX = -12, 2

    def norm_spec(x):
        return (x - SPEC_MIN) / (SPEC_MAX - SPEC_MIN) * 2 - 1

    def denorm_spec(x):
        return (x + 1) / 2 * (SPEC_MAX - SPEC_MIN) + SPEC_MIN

    def mel_fn_v4(x):
        return mel_spectrogram_torch(
            x, n_fft=1280, win_size=1280, hop_size=320,
            num_mels=100, sampling_rate=32000, fmin=0, fmax=None, center=False,
        )

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

    def get_ssl(wav_path):
        audio = load_audio(wav_path, 32000)
        audio16 = librosa.resample(audio, orig_sr=32000, target_sr=16000).astype(np.float32)
        t = torch.from_numpy(audio16).half().to(DEVICE)
        with torch.no_grad():
            return hubert.model(t.unsqueeze(0))["last_hidden_state"].transpose(1, 2)

    # ---- Reference cache ----
    ref_ssl = get_ssl(args.ref_wav)
    with torch.no_grad():
        ref_codes = vits.extract_latent(ref_ssl)
    ref_semantic = ref_codes[0, 0, :]
    ref_phone_ids, _, _ = get_phoneme_ids(args.ref_text, args.ref_lang)

    audio = load_audio(args.ref_wav, hps.data.sampling_rate)
    ref_spec = spectrogram_torch(
        torch.FloatTensor(audio).unsqueeze(0),
        hps.data.filter_length, hps.data.sampling_rate,
        hps.data.hop_length, hps.data.win_length, center=False,
    ).half().to(DEVICE)

    audio32 = load_audio(args.ref_wav, 32000)
    audio_t = torch.FloatTensor(audio32).unsqueeze(0).to(DEVICE)
    ref_mel = norm_spec(mel_fn_v4(audio_t)).half()

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

    prompt_sem_full = ref_semantic.unsqueeze(0).unsqueeze(0).to(DEVICE)
    ref_phones_t = torch.LongTensor(ref_phone_ids).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        fea_ref, ge = vits.decode_encp(prompt_sem_full, ref_phones_t, ref_spec)
        fea_todo, ge = vits.decode_encp(pred_sem, all_phone_ids, ref_spec, ge, 1.0)

        T_min = min(ref_mel.shape[2], fea_ref.shape[2])
        mel2 = ref_mel[:, :, :T_min]
        fea_ref = fea_ref[:, :, :T_min]
        T_ref, T_chunk = 500, 1000  # v4 vocoder chunk sizes
        if T_min > T_ref:
            mel2 = mel2[:, :, -T_ref:]
            fea_ref = fea_ref[:, :, -T_ref:]
            T_min = T_ref
        chunk_len = T_chunk - T_min

        cfm_results = []
        idx_pos = 0
        while True:
            chunk = fea_todo[:, :, idx_pos: idx_pos + chunk_len]
            if chunk.shape[-1] == 0:
                break
            idx_pos += chunk_len
            fea = torch.cat([fea_ref, chunk], 2).transpose(2, 1)
            cfm_res = vits.cfm.inference(
                fea, torch.LongTensor([fea.size(1)]).to(fea.device),
                mel2, args.sample_steps, inference_cfg_rate=0,
            )
            cfm_res = cfm_res[:, :, mel2.shape[2]:]
            mel2 = cfm_res[:, :, -T_min:]
            fea_ref = chunk[:, :, -T_min:]
            cfm_results.append(cfm_res)

        full_mel = denorm_spec(torch.cat(cfm_results, 2))
        wav_gen = vocoder(full_mel)
        wav = wav_gen[0, 0].cpu().float().numpy()

    sf.write(args.out, wav, 48000)
    print(f"Wrote {args.out} ({len(wav)/48000:.2f}s @ 48kHz)")


def _latest(d, pattern, key):
    files = sorted(Path(d).glob(pattern), key=key)
    if not files:
        raise SystemExit(f"No checkpoints matching {pattern} in {d}")
    return str(files[-1])


if __name__ == "__main__":
    main()

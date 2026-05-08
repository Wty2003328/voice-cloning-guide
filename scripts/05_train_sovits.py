"""Step 5: Fine-tune SoVITS (s2) — single GPU, no DDP, no Lightning.

The full GPT-SoVITS repo trains via DDP through a Lightning module. That's overkill for
a single consumer GPU and breaks on Windows (NCCL is Linux-only). This script bypasses
both layers and trains in a plain torch loop, while keeping the original loss
formulation: mel + KL + adversarial + feature-matching + KL_ssl.

Saves G checkpoints every --save-every epochs to SoVITS_weights_v2/<exp>_e<N>_s<step>.pth.
"""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from _common import setup, pretrained_paths


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--exp", required=True)
    p.add_argument("--gs-dir", default="./GPT-SoVITS")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--save-every", type=int, default=5)
    args = p.parse_args()

    gs_dir = setup(Path(args.gs_dir))
    paths = pretrained_paths(gs_dir)
    exp_dir = gs_dir / "logs" / args.exp

    import GPT_SoVITS.utils as utils
    from GPT_SoVITS.module import commons
    from GPT_SoVITS.module.data_utils import (
        DistributedBucketSampler,
        TextAudioSpeakerCollate,
        TextAudioSpeakerLoader,
    )
    from GPT_SoVITS.module.losses import (
        discriminator_loss, feature_loss, generator_loss, kl_loss,
    )
    from GPT_SoVITS.module.mel_processing import (
        mel_spectrogram_torch, spec_to_mel_torch,
    )
    from GPT_SoVITS.module.models import (
        MultiPeriodDiscriminator, SynthesizerTrn,
    )

    hps = utils.get_hparams_from_file(paths["s2_config"])
    hps.data.training_files = str(exp_dir / "2-name2text.txt")
    hps.data.exp_dir = str(exp_dir)

    torch.manual_seed(hps.train.seed)

    train_dataset = TextAudioSpeakerLoader(hps.data, version="v2")
    sampler = DistributedBucketSampler(
        train_dataset, args.batch_size,
        [32, 300, 400, 500, 600, 700, 800, 900],
        num_replicas=1, rank=0, shuffle=True,
    )
    loader = DataLoader(
        train_dataset, num_workers=0, shuffle=False, pin_memory=True,
        collate_fn=TextAudioSpeakerCollate(version="v2"),
        batch_sampler=sampler,
    )
    print(f"Dataset: {len(train_dataset)} samples, {len(loader)} batches")

    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        version="v2",
        **hps.model,
    ).cuda(0)
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm, version="v2").cuda(0)

    net_g.load_state_dict(
        torch.load(paths["s2g"], map_location="cpu", weights_only=False)["weight"],
        strict=False,
    )
    net_d.load_state_dict(
        torch.load(paths["s2d"], map_location="cpu", weights_only=False)["weight"],
        strict=False,
    )

    # Lower LR for the text/speaker encoder layers — they generalize across speakers,
    # so aggressive fine-tuning destroys multi-lingual capability.
    te_p = list(map(id, net_g.enc_p.text_embedding.parameters()))
    et_p = list(map(id, net_g.enc_p.encoder_text.parameters()))
    mrte_p = list(map(id, net_g.enc_p.mrte.parameters()))
    base_params = filter(
        lambda p: id(p) not in te_p + et_p + mrte_p and p.requires_grad,
        net_g.parameters(),
    )
    optim_g = torch.optim.AdamW([
        {"params": base_params, "lr": args.lr},
        {"params": net_g.enc_p.text_embedding.parameters(),
         "lr": args.lr * hps.train.text_low_lr_rate},
        {"params": net_g.enc_p.encoder_text.parameters(),
         "lr": args.lr * hps.train.text_low_lr_rate},
        {"params": net_g.enc_p.mrte.parameters(),
         "lr": args.lr * hps.train.text_low_lr_rate},
    ], args.lr, betas=hps.train.betas, eps=hps.train.eps)
    optim_d = torch.optim.AdamW(net_d.parameters(), args.lr,
                                betas=hps.train.betas, eps=hps.train.eps)

    sched_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay)
    sched_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay)
    scaler = GradScaler(enabled=hps.train.fp16_run)

    out_dir = gs_dir / "SoVITS_weights_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        net_g.train(); net_d.train()
        for batch_idx, data in enumerate(loader):
            ssl, ssl_lens, spec, spec_lens, y, y_lens, text, text_lens = data
            spec = spec.cuda(0, non_blocking=True)
            spec_lens = spec_lens.cuda(0, non_blocking=True)
            y = y.cuda(0, non_blocking=True); y_lens = y_lens.cuda(0, non_blocking=True)
            ssl = ssl.cuda(0, non_blocking=True); ssl.requires_grad = False
            text = text.cuda(0, non_blocking=True); text_lens = text_lens.cuda(0, non_blocking=True)

            with autocast(enabled=hps.train.fp16_run):
                (y_hat, kl_ssl, ids_slice, x_mask, z_mask,
                 (z, z_p, m_p, logs_p, m_q, logs_q), stats_ssl) = net_g(
                    ssl, spec, spec_lens, text, text_lens)

                mel = spec_to_mel_torch(
                    spec, hps.data.filter_length, hps.data.n_mel_channels,
                    hps.data.sampling_rate, hps.data.mel_fmin, hps.data.mel_fmax)
                y_mel = commons.slice_segments(
                    mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
                y_hat_mel = mel_spectrogram_torch(
                    y_hat.squeeze(1), hps.data.filter_length, hps.data.n_mel_channels,
                    hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
                    hps.data.mel_fmin, hps.data.mel_fmax)
                y = commons.slice_segments(y, ids_slice * hps.data.hop_length,
                                           hps.train.segment_size)

                y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
                with autocast(enabled=False):
                    loss_disc, _, _ = discriminator_loss(y_d_hat_r, y_d_hat_g)
                    loss_disc_all = loss_disc

            optim_d.zero_grad()
            scaler.scale(loss_disc_all).backward()
            scaler.unscale_(optim_d)
            commons.clip_grad_value_(net_d.parameters(), None)
            scaler.step(optim_d)

            with autocast(enabled=hps.train.fp16_run):
                y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
                with autocast(enabled=False):
                    loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                    loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                    loss_fm = feature_loss(fmap_r, fmap_g)
                    loss_gen, _ = generator_loss(y_d_hat_g)
                    loss_g_all = loss_gen + loss_fm + loss_mel + kl_ssl + loss_kl

            optim_g.zero_grad()
            scaler.scale(loss_g_all).backward()
            scaler.unscale_(optim_g)
            commons.clip_grad_value_(net_g.parameters(), None)
            scaler.step(optim_g)
            scaler.update()
            global_step += 1

            if batch_idx % 5 == 0:
                print(f"E{epoch} [{batch_idx}/{len(loader)}] "
                      f"d={float(loss_disc):.3f} g={float(loss_gen):.3f} "
                      f"mel={min(float(loss_mel),75):.1f} "
                      f"kl_ssl={float(kl_ssl):.3f} kl={float(loss_kl):.3f}")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt = net_g.state_dict()
            save_path = out_dir / f"{args.exp}_e{epoch}_s{global_step}.pth"
            torch.save({
                "weight": {k: v.half() for k, v in ckpt.items()},
                "config": hps,
                "info": f"SoVITS-{args.exp}-e{epoch}",
            }, str(save_path))
            print(f"Saved: {save_path}")
        sched_g.step(); sched_d.step()

    print("SoVITS training complete")


if __name__ == "__main__":
    main()

"""Step 5 (v4): LoRA fine-tune SoVITS v4 — single GPU, no DDP, no Lightning.

The v4 SoVITS is a flow-matching DiT (~700 MB) — too large for full fine-tune on
consumer GPUs. Upstream applies LoRA to the CFM's attention layers (to_k, to_q,
to_v, to_out.0) via PEFT. Single CFM loss, no discriminator.

Reuses the same data pipeline as v2 (logs/<exp>/2-name2text.txt, 4-cnhubert/,
5-wav32k/, 6-name2semantic.tsv) — no separate data prep.

Output: SoVITS_weights_v4/<exp>_e<N>_s<step>_l<rank>.pth (LoRA adapters + non-frozen
params merged with the base for direct loading at inference time).
"""
import argparse
from pathlib import Path
from time import time as ttime

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from _common import setup, pretrained_paths


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--exp", required=True)
    p.add_argument("--gs-dir", default="./GPT-SoVITS")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=2,
                   help="v4 DiT is heavy — keep small (2 fits on 16GB with grad ckpt)")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument("--no-grad-ckpt", action="store_true",
                   help="Disable gradient checkpointing (faster but more VRAM)")
    args = p.parse_args()

    gs_dir = setup(Path(args.gs_dir), version="v4")
    paths = pretrained_paths(gs_dir, version="v4")
    exp_dir = gs_dir / "logs" / args.exp

    import GPT_SoVITS.utils as utils
    from GPT_SoVITS.module import commons
    from GPT_SoVITS.module.data_utils import (
        DistributedBucketSampler,
        TextAudioSpeakerCollateV4,
        TextAudioSpeakerLoaderV4,
    )
    from GPT_SoVITS.module.models import SynthesizerTrnV3
    from peft import LoraConfig, get_peft_model

    hps = utils.get_hparams_from_file(paths["s2_config"])
    hps.data.training_files = str(exp_dir / "2-name2text.txt")
    hps.data.exp_dir = str(exp_dir)
    hps.model.version = "v4"
    hps.train.batch_size = args.batch_size
    hps.train.fp16_run = True
    hps.train.grad_ckpt = not args.no_grad_ckpt
    hps.train.lora_rank = args.lora_rank

    torch.manual_seed(hps.train.seed)

    train_dataset = TextAudioSpeakerLoaderV4(hps.data)
    sampler = DistributedBucketSampler(
        train_dataset, args.batch_size,
        [32, 300, 400, 500, 600, 700, 800, 900, 1000],
        num_replicas=1, rank=0, shuffle=True,
    )
    loader = DataLoader(
        train_dataset, num_workers=0, shuffle=False, pin_memory=True,
        collate_fn=TextAudioSpeakerCollateV4(),
        batch_sampler=sampler,
    )
    print(f"Dataset: {len(train_dataset)} samples, {len(loader)} batches/epoch")

    print(f"Building SynthesizerTrnV3 with v4 weights...")
    net_g = SynthesizerTrnV3(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    )
    print(f"Loading {paths['s2g']}")
    net_g.load_state_dict(
        torch.load(paths["s2g"], map_location="cpu", weights_only=False)["weight"],
        strict=False,
    )

    print(f"Applying LoRA (rank={args.lora_rank}) to CFM attention...")
    lora_config = LoraConfig(
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        r=args.lora_rank, lora_alpha=args.lora_rank,
        init_lora_weights=True,
    )
    net_g.cfm = get_peft_model(net_g.cfm, lora_config)

    no_grad_names = {n for n, p in net_g.named_parameters() if not p.requires_grad}
    trainable = sum(p.numel() for p in net_g.parameters() if p.requires_grad)
    total = sum(p.numel() for p in net_g.parameters())
    print(f"  trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    net_g = net_g.cuda(0)

    optim_g = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, net_g.parameters()),
        lr=args.lr, betas=hps.train.betas, eps=hps.train.eps,
    )
    sched_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay)
    scaler = GradScaler(enabled=hps.train.fp16_run)

    out_dir = gs_dir / "SoVITS_weights_v4"
    out_dir.mkdir(parents=True, exist_ok=True)
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        net_g.train()
        sampler.set_epoch(epoch)
        t0 = ttime()
        running = []

        for batch_idx, (ssl, spec, mel, ssl_lens, spec_lens, text, text_lens, mel_lens) in enumerate(loader):
            spec = spec.cuda(0, non_blocking=True); spec_lens = spec_lens.cuda(0, non_blocking=True)
            mel = mel.cuda(0, non_blocking=True);   mel_lens = mel_lens.cuda(0, non_blocking=True)
            ssl = ssl.cuda(0, non_blocking=True);   ssl.requires_grad = False
            text = text.cuda(0, non_blocking=True); text_lens = text_lens.cuda(0, non_blocking=True)

            with autocast(enabled=hps.train.fp16_run):
                cfm_loss = net_g(
                    ssl, spec, mel, ssl_lens, spec_lens,
                    text, text_lens, mel_lens,
                    use_grad_ckpt=hps.train.grad_ckpt,
                )

            optim_g.zero_grad()
            scaler.scale(cfm_loss).backward()
            scaler.unscale_(optim_g)
            commons.clip_grad_value_(net_g.parameters(), None)
            scaler.step(optim_g)
            scaler.update()

            global_step += 1
            running.append(float(cfm_loss))

            if batch_idx % 5 == 0:
                print(f"E{epoch} [{batch_idx}/{len(loader)}] cfm_loss={float(cfm_loss):.4f}")

        avg = sum(running) / max(1, len(running))
        print(f"=== Epoch {epoch}: avg_cfm_loss={avg:.4f} ({ttime()-t0:.0f}s) ===")
        sched_g.step()

        if epoch % args.save_every == 0 or epoch == args.epochs:
            full_state = net_g.state_dict()
            sim_ckpt = {k: v.half().cpu() for k, v in full_state.items()
                        if k not in no_grad_names}
            save_path = out_dir / f"{args.exp}_e{epoch}_s{global_step}_l{args.lora_rank}.pth"
            torch.save({
                "weight": sim_ckpt,
                "config": hps,
                "info": f"SoVITS-v4-{args.exp}-e{epoch}-l{args.lora_rank}",
                "lora_rank": args.lora_rank,
                "model_version": "v4",
            }, str(save_path))
            print(f"Saved: {save_path}")

    print("v4 SoVITS training complete!")


if __name__ == "__main__":
    main()

"""Step 6: Fine-tune the GPT (s1, Text2SemanticDecoder) — single GPU, plain torch loop.

The GPT predicts semantic tokens autoregressively given phoneme IDs and BERT features.
We train with cross-entropy (sum reduction) on each (phoneme + BERT → semantic) pair.

Saves a half-precision checkpoint per epoch to GPT_weights_v2/<exp>-e<N>.ckpt.
"""
import argparse
from pathlib import Path
from time import time as ttime

import torch
from torch.utils.data import DataLoader, RandomSampler

from _common import setup, pretrained_paths


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--exp", required=True)
    p.add_argument("--gs-dir", default="./GPT-SoVITS")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    gs_dir = setup(Path(args.gs_dir))
    paths = pretrained_paths(gs_dir)
    exp_dir = gs_dir / "logs" / args.exp
    DEVICE = torch.device(args.device)

    from AR.data.dataset import Text2SemanticDataset
    from GPT_SoVITS.AR.models.t2s_lightning_module import Text2SemanticLightningModule

    dataset = Text2SemanticDataset(
        phoneme_path=str(exp_dir / "2-name2text.txt"),
        semantic_path=str(exp_dir / "6-name2semantic.tsv"),
        max_sec=54, pad_val=1024,
    )
    print(f"Dataset size: {len(dataset)}")

    loader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=RandomSampler(dataset),
        collate_fn=dataset.collate, num_workers=0, drop_last=False,
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

    model = Text2SemanticLightningModule(s1config, Path("."), is_train=True)
    ckpt = torch.load(paths["s1"], map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["weight"], strict=False)
    model = model.to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        betas=(0.9, 0.95), weight_decay=0.01,
    )

    out_dir = gs_dir / "GPT_weights_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.model.train()
        t0 = ttime()
        total_loss, total_acc, n = 0.0, 0.0, 0

        for batch_idx, batch in enumerate(loader):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(DEVICE)

            global_step += 1
            if global_step < args.warmup_steps:
                lr = args.lr * global_step / args.warmup_steps
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

            loss, acc = model.model.forward_old(
                batch["phoneme_ids"], batch["phoneme_ids_len"],
                batch["semantic_ids"], batch["semantic_ids_len"],
                batch["bert_feature"],
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_acc += acc
            n += 1
            if batch_idx % 5 == 0:
                print(f"E{epoch} [{batch_idx}/{len(loader)}] "
                      f"loss={loss.item():.1f} acc={acc:.3f} "
                      f"lr={optimizer.param_groups[0]['lr']:.2e}")

        avg_loss = total_loss / max(n, 1)
        avg_acc = total_acc / max(n, 1)
        print(f"=== Epoch {epoch}: avg_loss={avg_loss:.1f} avg_acc={avg_acc:.3f} "
              f"({ttime()-t0:.0f}s) ===")

        save_path = out_dir / f"{args.exp}-e{epoch}.ckpt"
        torch.save({
            "weight": {k: v.half() for k, v in model.state_dict().items()},
            "config": s1config,
            "info": f"{args.exp}-e{epoch}",
        }, str(save_path))
        print(f"Saved: {save_path}")

    print("GPT training complete")


if __name__ == "__main__":
    main()

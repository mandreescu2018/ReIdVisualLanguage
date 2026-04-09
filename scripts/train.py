"""
scripts/train.py — training entry point.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml --resume checkpoints/default/reid_last.pth
"""

import argparse, random, sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets        import build_dataset, PKSampler
from engine          import Trainer, Evaluator, LoggerCallback, EvalCheckpointCallback
from losses          import build_losses
from models          import build_model
from utils           import get_logger, load_checkpoint


# ── helpers ───────────────────────────────────────────────────────────────────

def load_cfg(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if "_base_" in cfg:                         # simple single-level inheritance
        with open(Path(path).parent / cfg.pop("_base_")) as f:
            base = yaml.safe_load(f)
        base.update(cfg)
        cfg = base
    return cfg


def build_optimizer_and_scheduler(cfg, model, criterion, total_epochs):
    lr    = cfg["optimizer"]["lr"]
    scale = cfg["optimizer"].get("backbone_lr_scale", 0.1)
    wd    = cfg["optimizer"].get("weight_decay", 5e-4)
    wu    = cfg["scheduler"].get("warmup_epochs", 10)

    opt = Adam([
        {"params": model.backbone.parameters(), "lr": lr * scale},
        {"params": list(model.neck.parameters()) + list(model.head.parameters()), "lr": lr},
    ], weight_decay=wd)

    center_opt = None
    if hasattr(criterion, "aux_loss") and criterion.aux_loss is not None:
        center_opt = Adam(criterion.aux_loss.parameters(), lr=0.5)

    sched = SequentialLR(opt, [
        LinearLR(opt, start_factor=0.01, end_factor=1.0, total_iters=wu),
        CosineAnnealingLR(opt, T_max=max(total_epochs - wu, 1),
                          eta_min=cfg["scheduler"].get("min_lr", 1e-7)),
    ], milestones=[wu])

    return opt, center_opt, sched


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",  required=True)
    ap.add_argument("--resume",  default="")
    args = ap.parse_args()

    cfg    = load_cfg(args.config)
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    seed   = cfg.get("seed", 42)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    out_dir = cfg.get("output_dir", "checkpoints/default")
    logger  = get_logger(log_file=Path(out_dir) / "train.log")
    logger.info(f"config={args.config}  device={device}")

    # ── data ──
    nw  = cfg["data"].get("num_workers", 4)
    smp = cfg["sampler"]
    train_set     = build_dataset(cfg, "train")
    query_set     = build_dataset(cfg, "query")
    gallery_set   = build_dataset(cfg, "gallery")
    logger.info(train_set)

    train_loader   = DataLoader(train_set,
                                batch_sampler=PKSampler(train_set, smp["P"], smp["K"], smp["num_iter"]),
                                num_workers=nw, pin_memory=True)
    query_loader   = DataLoader(query_set,   batch_size=128, shuffle=False, num_workers=nw)
    gallery_loader = DataLoader(gallery_set, batch_size=128, shuffle=False, num_workers=nw)

    # ── model + losses ──
    feat_dim  = cfg["model"]["neck"]["feat_dim"]
    model     = build_model(cfg, num_classes=train_set.num_pids).to(device)
    criterion = build_losses(cfg, num_classes=train_set.num_pids, feat_dim=feat_dim).to(device)
    logger.info(f"model ready  |  {train_set.num_pids} identities")

    # ── optimiser + scheduler ──
    total_ep = cfg["scheduler"]["num_epochs"]
    opt, center_opt, sched = build_optimizer_and_scheduler(cfg, model, criterion, total_ep)

    # ── resume ──
    start_epoch = 0
    if args.resume:
        ckpt        = load_checkpoint(args.resume, model, opt, sched, device=device)
        start_epoch = ckpt.get("epoch", 0) + 1
        logger.info(f"resumed from epoch {start_epoch}")

    # ── callbacks ──
    evaluator = Evaluator(model, device,
                          metric=cfg["eval"].get("metric", "cosine"),
                          ranks=cfg["eval"].get("ranks", [1, 5, 10, 20]))
    callbacks = [
        LoggerCallback(logger),
        EvalCheckpointCallback(evaluator, query_loader, gallery_loader,
                               out_dir, eval_every=cfg["eval"]["eval_every"],
                               logger=logger),
    ]

    # ── train ──
    trainer = Trainer(model, criterion, opt, sched, train_loader, device, center_opt)
    trainer.run(num_epochs=total_ep - start_epoch, start_epoch=start_epoch, callbacks=callbacks)
    logger.info("done.")


if __name__ == "__main__":
    main()

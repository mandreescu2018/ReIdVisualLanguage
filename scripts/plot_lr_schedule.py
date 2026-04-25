"""Visualise the LR schedule used by train.py (via LearningRateScheduler)."""
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from yacs.config import CfgNode as CN
import matplotlib.pyplot as plt

from solver import LearningRateScheduler


def build_cfg(args):
    cfg = CN()
    cfg.SOLVER = CN()
    cfg.SOLVER.BASE_LR = args.base_lr
    cfg.SOLVER.MAX_EPOCHS = args.epochs
    cfg.SOLVER.SCHEDULER = args.scheduler
    cfg.SOLVER. WARMUP_EPOCHS = args.warmup_epochs
    cfg.SOLVER.WARMUP_FACTOR = args.warmup_factor
    cfg.SOLVER.WARMUP_METHOD = "linear"
    cfg.SOLVER.GAMMA = args.gamma
    cfg.SOLVER.STEPS = args.steps
    cfg.SOLVER.BIAS_LR_FACTOR = 1
    cfg.SOLVER.LARGE_FC_LR = False
    cfg.SOLVER.WEIGHT_DECAY = 1e-4
    cfg.SOLVER.WEIGHT_DECAY_BIAS = 1e-4
    cfg.SOLVER.MOMENTUM = 0.9
    return cfg


def collect_lrs(optimizer, scheduler, total_epochs):
    lrs = []
    for epoch in range(total_epochs):
        lrs.append(optimizer.param_groups[0]["lr"])
        scheduler.step(epoch)
    return lrs


def main():
    parser = argparse.ArgumentParser(description="Plot LR schedule (train.py path)")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--base_lr", type=float, default=3e-4)
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["cosine", "cosine_annealing", "step", "exponential", "warm_up"])
    parser.add_argument("--warmup_epochs", type=int, default=10,
                        help="Warmup epochs (used by cosine and warm_up schedulers)")
    parser.add_argument("--warmup_factor", type=float, default=0.01,
                        help="Starting LR = base_lr * warmup_factor")
    parser.add_argument("--gamma", type=float, default=0.1,
                        help="Decay factor for step/exponential schedulers")
    parser.add_argument("--steps", type=int, nargs="+", default=[40, 80],
                        help="Milestone epochs for warm_up/step schedulers")
    args = parser.parse_args()

    cfg = build_cfg(args)

    dummy = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(dummy.parameters(), lr=args.base_lr)

    lr_scheduler = LearningRateScheduler(optimizer, cfg)
    lrs = collect_lrs(optimizer, lr_scheduler, args.epochs)

    epochs = list(range(args.epochs))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, lrs, label=f"LR — {args.scheduler}")
    if args.warmup_epochs > 0 and args.scheduler in ("cosine", "warm_up"):
        ax.axvline(args.warmup_epochs, color="gray", linestyle=":",
                   linewidth=1, label=f"Warmup end (epoch {args.warmup_epochs})")
    if args.scheduler in ("step", "warm_up"):
        for s in args.steps:
            ax.axvline(s, color="red", linestyle="--", linewidth=1, alpha=0.5, label=f"Step @ {s}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title(f"LR Schedule — {args.scheduler}, {args.epochs} epochs, base_lr={args.base_lr:.1e}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    # plt.savefig("lr_schedule.png", dpi=150)
    # print("Saved lr_schedule.png")
    plt.show()


if __name__ == "__main__":
    main()

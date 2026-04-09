"""utils/checkpoint.py"""
from pathlib import Path
import torch


def save_checkpoint(path: str, epoch: int, model, optimizer, scheduler, metrics: dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch":     epoch,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "metrics":   metrics,
    }, path)


def load_checkpoint(path: str, model, optimizer=None, scheduler=None, device="cpu") -> dict:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt
"""
engine/hooks.py

Lightweight hook system. Hooks fire at training lifecycle events.
Add new hooks here; register them in scripts/train.py.

Available events:
    before_train(trainer)
    after_train(trainer)
    before_epoch(trainer, epoch)
    after_epoch(trainer, epoch, metrics)
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .trainer import Trainer


class BaseHook(ABC):
    def before_train(self, trainer: "Trainer"):  pass
    def after_train(self, trainer: "Trainer"):   pass
    def before_epoch(self, trainer: "Trainer", epoch: int): pass
    def after_epoch(self, trainer: "Trainer", epoch: int, metrics: dict): pass


class HookList:
    """Container that broadcasts events to all registered hooks."""
    def __init__(self, hooks: list[BaseHook]):
        self.hooks = hooks

    def before_train(self, t):              
        [h.before_train(t) for h in self.hooks]
    def after_train(self, t):               
        [h.after_train(t) for h in self.hooks]
    def before_epoch(self, t, e):           
        [h.before_epoch(t, e) for h in self.hooks]
    def after_epoch(self, t, e, m):         
        [h.after_epoch(t, e, m) for h in self.hooks]


class LoggingHook(BaseHook):
    """Prints loss metrics after every epoch."""

    def __init__(self, logger):
        self.logger = logger

    def after_epoch(self, trainer, epoch, metrics):
        self.logger.info(
            f"Epoch {epoch+1:>4} | "
            f"loss={metrics['total']:.4f} "
            f"(id={metrics['id']:.4f} "
            f"metric={metrics['metric']:.4f} "
            f"aux={metrics['aux']:.4f}) "
            f"[{metrics['elapsed']:.1f}s] "
            f"lr={trainer.scheduler.get_last_lr()[0]:.2e}"
        )


class CheckpointHook(BaseHook):
    """Saves model checkpoint every N epochs, and always the best rank-1."""

    def __init__(
        self,
        output_dir:  str,
        evaluator,
        query_loader,
        gallery_loader,
        eval_every:  int = 10,
        logger=None,
    ):
        self.output_dir     = Path(output_dir)
        self.evaluator      = evaluator
        self.query_loader   = query_loader
        self.gallery_loader = gallery_loader
        self.eval_every     = eval_every
        self.logger         = logger
        self.best_rank1     = 0.0
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def after_epoch(self, trainer, epoch, metrics):
        if (epoch + 1) % self.eval_every != 0:
            return

        eval_metrics = self.evaluator.evaluate(self.query_loader, self.gallery_loader)
        msg = (
            f"  → Rank-1: {eval_metrics['rank1']*100:.2f}%  "
            f"Rank-5: {eval_metrics['rank5']*100:.2f}%  "
            f"mAP: {eval_metrics['mAP']*100:.2f}%"
        )
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

        self._save(trainer, epoch, eval_metrics, tag="last")
        if eval_metrics["rank1"] > self.best_rank1:
            self.best_rank1 = eval_metrics["rank1"]
            self._save(trainer, epoch, eval_metrics, tag="best")
            note = f"  ★ New best Rank-1: {self.best_rank1*100:.2f}%"
            if self.logger:
                self.logger.info(note)
            else:
                print(note)

    def _save(self, trainer, epoch, eval_metrics, tag):
        path = self.output_dir / f"reid_{tag}.pth"
        torch.save({
            "epoch":     epoch,
            "model":     trainer.model.state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
            "scheduler": trainer.scheduler.state_dict(),
            "metrics":   eval_metrics,
        }, path)
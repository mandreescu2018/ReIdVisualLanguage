"""
engine/callbacks.py

Callbacks fire at training lifecycle events.
Add new ones here; register them in scripts/train.py.

Events:  on_train_begin / on_train_end
         on_epoch_begin / on_epoch_end(metrics)
"""

from __future__ import annotations
from pathlib import Path
import torch


class BaseCallback:
    def on_train_begin(self, trainer): pass
    def on_train_end(self, trainer):   pass
    def on_epoch_begin(self, trainer, epoch: int): pass
    def on_epoch_end(self, trainer, epoch: int, metrics: dict): pass


class LoggerCallback(BaseCallback):
    """Prints per-epoch loss + LR to a logger."""

    def __init__(self, logger):
        self.log = logger

    def on_epoch_end(self, trainer, epoch, metrics):
        self.log.info(
            f"[{epoch+1:>4}] "
            f"loss={metrics['total']:.4f} "
            f"(id={metrics['id']:.4f} "
            f"metric={metrics['metric']:.4f} "
            f"aux={metrics['aux']:.4f})  "
            f"lr={metrics['lr']:.2e}  "
            f"[{metrics['elapsed']:.0f}s]"
        )


class EvalCheckpointCallback(BaseCallback):
    """
    Every eval_every epochs: evaluate, save 'last', save 'best' if improved.
    """

    def __init__(self, evaluator, query_loader, gallery_loader,
                 output_dir: str, eval_every: int = 10, logger=None):
        self.evaluator       = evaluator
        self.query_loader    = query_loader
        self.gallery_loader  = gallery_loader
        self.out             = Path(output_dir)
        self.eval_every      = eval_every
        self.log             = logger
        self.best_rank1      = 0.0
        self.out.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, trainer, epoch, metrics):
        if (epoch + 1) % self.eval_every:
            return

        m = self.evaluator.evaluate(self.query_loader, self.gallery_loader)
        msg = (f"  Rank-1 {m['rank1']*100:.2f}%  "
               f"Rank-5 {m['rank5']*100:.2f}%  "
               f"mAP {m['mAP']*100:.2f}%")
        (self.log.info if self.log else print)(msg)

        self._save(trainer, epoch, m, "last")
        if m["rank1"] > self.best_rank1:
            self.best_rank1 = m["rank1"]
            self._save(trainer, epoch, m, "best")
            note = f"  ★ best Rank-1 → {self.best_rank1*100:.2f}%"
            (self.log.info if self.log else print)(note)

    def _save(self, trainer, epoch, metrics, tag):
        torch.save({
            "epoch":     epoch,
            "model":     trainer.model.state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
            "scheduler": trainer.scheduler.state_dict(),
            "metrics":   metrics,
        }, self.out / f"reid_{tag}.pth")


class FreezeBBCallback(BaseCallback):
    """
    Freeze the backbone for the first `freeze_epochs` epochs, then unfreeze.
    Useful as a warmup strategy: train new layers first, then fine-tune all.
    """

    def __init__(self, freeze_epochs: int, logger=None):
        self.freeze_epochs = freeze_epochs
        self.log           = logger

    def on_epoch_begin(self, trainer, epoch):
        if epoch < self.freeze_epochs:
            trainer.freeze_backbone()
        elif epoch == self.freeze_epochs:
            trainer.unfreeze_backbone()
            msg = f"  Epoch {epoch+1}: backbone unfrozen"
            (self.log.info if self.log else print)(msg)
"""
engine/trainer.py

Training loop. Designed to be stage-aware and hook-compatible.

Stages let you change LR, freeze layers, swap losses, etc.
between phases of training — all driven from config.
"""

import time
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .hooks import HookList


class Trainer:
    """
    Manages the training loop for one or more stages.

    Args:
        model:            ReIDModel (backbone + neck + head).
        criterion:        CombinedLoss instance.
        optimizer:        Main parameter optimiser.
        scheduler:        LR scheduler.
        train_loader:     DataLoader using PKSampler.
        device:           torch.device.
        center_optimizer: Optional separate optimiser for CenterLoss centres.
        hooks:            Optional HookList for checkpointing, logging, etc.
    """

    def __init__(
        self,
        model:             nn.Module,
        criterion:         nn.Module,
        optimizer:         torch.optim.Optimizer,
        scheduler,
        train_loader:      DataLoader,
        device:            torch.device,
        center_optimizer:  Optional[torch.optim.Optimizer] = None,
        hooks:             Optional[HookList] = None,
    ):
        self.model            = model
        self.criterion        = criterion
        self.optimizer        = optimizer
        self.scheduler        = scheduler
        self.train_loader     = train_loader
        self.device           = device
        self.center_optimizer = center_optimizer
        self.hooks            = hooks or HookList([])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, num_epochs: int, start_epoch: int = 0):
        """Run training for num_epochs, calling hooks at each boundary."""
        self.hooks.before_train(self)
        for epoch in range(start_epoch, start_epoch + num_epochs):
            self.hooks.before_epoch(self, epoch)
            metrics = self._run_epoch(epoch)
            self.scheduler.step()
            self.hooks.after_epoch(self, epoch, metrics)
        self.hooks.after_train(self)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_epoch(self, epoch: int) -> dict:
        self.model.train()
        totals = {"total": 0.0, "id": 0.0, "metric": 0.0, "aux": 0.0}
        t0 = time.time()

        for batch in self.train_loader:
            imgs   = batch["image"].to(self.device)
            labels = batch["pid"].to(self.device)

            # Forward
            feat_map         = self.model.backbone(imgs)
            feat, feat_bn    = self.model.neck(feat_map)      # training mode → two tensors
            logits           = self.model.head(feat_bn)

            losses = self.criterion(feat, feat_bn, logits, labels)

            # Backward
            self.optimizer.zero_grad()
            if self.center_optimizer:
                self.center_optimizer.zero_grad()

            losses["total"].backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            if self.center_optimizer:
                self.center_optimizer.step()

            for k in totals:
                totals[k] += losses[k].item()

        n = max(len(self.train_loader), 1)
        elapsed = time.time() - t0
        avg = {k: v / n for k, v in totals.items()}
        avg["elapsed"] = elapsed
        avg["epoch"]   = epoch
        return avg
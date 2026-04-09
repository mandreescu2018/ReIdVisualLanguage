"""losses/center_loss.py — needs its own optimiser (lr ≈ 0.5), see engine/trainer.py."""
import torch
import torch.nn as nn


class CenterLoss(nn.Module):
    """Pulls intra-class features towards learnable class centroids (Wen et al. 2016)."""

    def __init__(self, num_classes, feat_dim=2048, **kw):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, feats: torch.Tensor, labels: torch.Tensor):
        return (feats - self.centers[labels]).pow(2).sum(1).mean() / 2.0

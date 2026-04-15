"""losses/triplet_loss.py — online hard-mining triplet loss."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """
    Batch-hard triplet loss (Hermans et al. 2017).
    Uses the hardest positive and hardest negative per anchor in the PK batch.

    Args:
        margin:      Hinge margin α (ignored when soft_margin=True).
        hard_mining: True = batch-hard, False = batch-mean (softer).
        normalize:   L2-normalise before computing distances.
        soft_margin: Use softplus instead of hinge.
    """

    def __init__(self, margin=0.3, hard_mining=True, normalize=True, soft_margin=False, **kw):
        super().__init__()
        self.margin      = margin
        self.hard_mining = hard_mining
        self.normalize   = normalize
        self.soft_margin = soft_margin

    def _dist(self, f):
        if self.normalize:
            f = F.normalize(f, p=2, dim=1)
        d = f @ f.T
        s = d.diag().unsqueeze(1)
        return (s + s.T - 2 * d).clamp(min=1e-12).sqrt()

    def forward(self, feats: torch.Tensor, labels: torch.Tensor):
        dist = self._dist(feats)
        same = labels.unsqueeze(1).eq(labels.unsqueeze(0))

        if self.hard_mining:
            pos = (dist * same.float()).max(1).values
            neg = (dist + (~same).float() * 1e9).min(1).values
        else:
            pos = (dist * same.float()).sum(1) / same.float().sum(1).clamp(min=1)
            neg = (dist * (~same).float()).sum(1) / (~same).float().sum(1).clamp(min=1)

        return (F.softplus(pos - neg) if self.soft_margin
                else F.relu(pos - neg + self.margin)).mean()

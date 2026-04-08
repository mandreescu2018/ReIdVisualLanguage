"""
models/necks/bnneck.py

BN-Neck: pools spatial features, applies BatchNorm.

  train → returns (feat, feat_bn)
          feat    goes to metric losses (triplet, center)
          feat_bn goes to the classifier head
  eval  → returns L2-normalised feat_bn for cosine retrieval
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeMPool(nn.Module):
    """Generalised Mean Pooling — learnable interpolation between avg and max."""
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p   = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p), (1, 1)).pow(1.0 / self.p)


class BNNeck(nn.Module):
    """
    Args:
        feat_dim: Channel count from backbone (2048 for ResNet-50).
    """

    def __init__(self, feat_dim=2048):
        super().__init__()
        self.feat_dim = feat_dim
        self.pool = GeMPool()
        self.bn   = nn.BatchNorm1d(feat_dim)
        self.bn.bias.requires_grad_(False)
        nn.init.constant_(self.bn.weight, 1.0)
        nn.init.constant_(self.bn.bias,   0.0)

    def forward(self, feat_map):
        feat    = self.pool(feat_map).flatten(1)   # (B, D)
        feat_bn = self.bn(feat)
        if self.training:
            return feat, feat_bn
        return F.normalize(feat_bn, p=2, dim=1)
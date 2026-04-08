"""
models/heads/classifier.py

Linear classifier for ID supervision.
Receives feat_bn from the neck; returns raw logits.
Swap this with ArcFaceHead, CircleHead, etc. as you experiment.
"""

import torch.nn as nn


class LinearClassifier(nn.Module):
    def __init__(self, feat_dim=2048, num_classes=751):
        super().__init__()
        self.fc = nn.Linear(feat_dim, num_classes, bias=False)
        nn.init.normal_(self.fc.weight, std=0.001)

    def forward(self, feat_bn):
        return self.fc(feat_bn)   # (B, num_classes)
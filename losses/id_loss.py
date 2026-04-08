"""losses/id_loss.py"""
import torch
import torch.nn as nn

class CrossEntropyLoss(nn.Module):
    def __init__(self, num_classes, **kw):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        return self.ce(logits, labels)

class LabelSmoothCELoss(nn.Module):
    """Label-smoothed cross-entropy. Typically +0.3–0.5% mAP over plain CE."""

    def __init__(self, num_classes, epsilon=0.1, **kw):
        super().__init__()
        self.K       = num_classes
        self.eps     = epsilon
        self.log_sm  = nn.LogSoftmax(dim=1)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        log_p = self.log_sm(logits)
        with torch.no_grad():
            smooth = torch.full_like(log_p, self.eps / (self.K - 1))
            smooth.scatter_(1, labels.unsqueeze(1), 1.0 - self.eps)
        return -(smooth * log_p).sum(1).mean()
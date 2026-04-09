"""
tests/test_losses.py  —  run with: pytest tests/ -v
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch, pytest
from losses.id_loss      import CrossEntropyLoss, LabelSmoothCELoss
from losses.triplet_loss import TripletLoss
from losses.center_loss  import CenterLoss

B, D, C = 16, 128, 50   # small dims for speed


@pytest.fixture
def batch():
    return (torch.randn(B, D),          # feats
            torch.randint(0, C, (B,)),  # labels
            torch.randn(B, C))          # logits


def test_cross_entropy(batch):
    _, labels, logits = batch
    loss = CrossEntropyLoss(C)(logits, labels)
    assert loss.item() > 0 and loss.shape == ()


def test_label_smooth_ce(batch):
    _, labels, logits = batch
    loss = LabelSmoothCELoss(C)(logits, labels)
    assert loss.item() > 0


def test_triplet_hard(batch):
    feats, labels, _ = batch
    loss = TripletLoss(margin=0.3, hard_mining=True)(feats, labels)
    assert loss.item() >= 0


def test_triplet_soft_margin(batch):
    feats, labels, _ = batch
    loss = TripletLoss(soft_margin=True)(feats, labels)
    assert loss.item() >= 0


def test_center_loss(batch):
    feats, labels, _ = batch
    loss = CenterLoss(num_classes=C, feat_dim=D)(feats, labels)
    assert loss.item() >= 0

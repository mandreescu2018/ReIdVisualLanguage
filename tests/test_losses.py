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
P, K = 4, 4             # P identities × K images each (required by hard_example_mining)


@pytest.fixture
def batch():
    labels = torch.repeat_interleave(torch.arange(P), K)  # [0,0,0,0, 1,1,1,1, ...]
    return (torch.randn(B, D),   # feats
            labels,              # balanced P×K labels
            torch.randn(B, C))  # logits


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
    loss, _, _ = TripletLoss(margin=0.3)(feats, labels)
    assert loss.item() >= 0


def test_triplet_soft_margin(batch):
    feats, labels, _ = batch
    loss, _, _ = TripletLoss(margin=None)(feats, labels)
    assert loss.item() >= 0


def test_center_loss(batch):
    feats, labels, _ = batch
    loss = CenterLoss(num_classes=C, feat_dim=D)(feats, labels)
    assert loss.item() >= 0

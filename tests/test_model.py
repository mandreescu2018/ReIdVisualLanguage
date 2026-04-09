"""
tests/test_model.py  —  run with: pytest tests/ -v
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch, pytest
from models.backbones.resnet import ResNet50Backbone
from models.necks.bnneck    import BNNeck
from models.heads.classifier import LinearClassifier
from models.builder          import ReIDModel

B = 4


@pytest.fixture
def imgs():
    return torch.randn(B, 3, 256, 128)


def make_model(num_classes=100):
    bb = ResNet50Backbone(pretrained=False, last_stride=1, ibn=False)
    nk = BNNeck(feat_dim=2048)
    hd = LinearClassifier(feat_dim=2048, num_classes=num_classes)
    return ReIDModel(bb, nk, hd)


def test_backbone_shape(imgs):
    bb  = ResNet50Backbone(pretrained=False)
    out = bb(imgs)
    assert out.shape == (B, 2048, 16, 8)


def test_train_forward(imgs):
    m = make_model(); m.train()
    feat, feat_bn, logits = m(imgs)
    assert feat.shape    == (B, 2048)
    assert feat_bn.shape == (B, 2048)
    assert logits.shape  == (B, 100)


def test_eval_forward_normalised(imgs):
    m = make_model(); m.eval()
    with torch.no_grad():
        emb = m(imgs)
    assert emb.shape == (B, 2048)
    assert torch.allclose(emb.norm(dim=1), torch.ones(B), atol=1e-5)

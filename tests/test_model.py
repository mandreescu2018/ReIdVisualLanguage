"""
tests/test_model.py  - run with: pytest tests/ -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch

from config import cfg
from models.vit_model import build_transformer

BATCH_SIZE = 2
NUM_CLASSES = 7
FEATURE_DIM = 384


@pytest.fixture
def vit_cfg():
    test_cfg = cfg.clone()
    test_cfg.defrost()
    test_cfg.MODEL.DEVICE = "cpu"
    test_cfg.MODEL.NAME = "vit_transformer"
    test_cfg.MODEL.PRETRAIN_CHOICE = "none"
    test_cfg.MODEL.TRANSFORMER.TYPE = "deit_small_patch16_224_TransReID"
    test_cfg.MODEL.TRANSFORMER.LAYERS = 1
    test_cfg.MODEL.TRANSFORMER.NUM_HEADS = 6
    test_cfg.INPUT.SIZE_TRAIN = [32, 16]
    test_cfg.INPUT.SIZE_TEST = [32, 16]
    test_cfg.MODEL.STRIDE_SIZE = [16, 16]
    test_cfg.MODEL.DROP_PATH = 0.0
    test_cfg.MODEL.DROP_OUT = 0.0
    test_cfg.MODEL.ATT_DROP_RATE = 0.0
    test_cfg.DATASETS.NUMBER_OF_CLASSES = NUM_CLASSES
    test_cfg.DATASETS.NUMBER_OF_CAMERAS = 0
    test_cfg.DATASETS.NUMBER_OF_TRACKS = 0
    test_cfg.TEST.NECK_FEAT = "before"
    test_cfg.freeze()
    return test_cfg


@pytest.fixture
def imgs(vit_cfg):
    height, width = vit_cfg.INPUT.SIZE_TRAIN
    return torch.randn(BATCH_SIZE, 3, height, width)


def test_vit_train_forward(vit_cfg, imgs):
    model = build_transformer(vit_cfg)
    model.train()

    cls_score, global_feat = model(imgs)

    assert cls_score.shape == (BATCH_SIZE, NUM_CLASSES)
    assert global_feat.shape == (BATCH_SIZE, FEATURE_DIM)


def test_vit_eval_forward_before_neck(vit_cfg, imgs):
    model = build_transformer(vit_cfg)
    model.eval()

    with torch.no_grad():
        embeddings = model(imgs)

    assert embeddings.shape == (BATCH_SIZE, FEATURE_DIM)


def test_vit_eval_forward_after_neck(vit_cfg, imgs):
    test_cfg = vit_cfg.clone()
    test_cfg.defrost()
    test_cfg.TEST.NECK_FEAT = "after"
    test_cfg.freeze()

    model = build_transformer(test_cfg)
    model.eval()

    with torch.no_grad():
        embeddings = model(imgs)

    assert embeddings.shape == (BATCH_SIZE, FEATURE_DIM)

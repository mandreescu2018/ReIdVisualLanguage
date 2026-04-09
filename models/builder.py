"""
models/builder.py

Assembles backbone + neck + head from config.
The ReIDModel class is the only object the training loop touches.
"""

import torch.nn as nn
from .backbones import BACKBONES
from .necks     import NECKS
from .heads     import HEADS


class ReIDModel(nn.Module):
    """
    backbone → neck → head  (training)
    backbone → neck          (inference, returns L2-normalised embedding)
    """

    def __init__(self, backbone, neck, head):
        super().__init__()
        self.backbone = backbone
        self.neck     = neck
        self.head     = head

    def forward(self, x):
        feat_map = self.backbone(x)
        if self.training:
            feat, feat_bn = self.neck(feat_map)
            logits        = self.head(feat_bn)
            return feat, feat_bn, logits
        return self.neck(feat_map)   # normalised embedding

    @property
    def feat_dim(self):
        return self.neck.feat_dim


def build_model(cfg: dict, num_classes: int) -> ReIDModel:
    bb_cfg   = cfg["model"]["backbone"]
    neck_cfg = cfg["model"]["neck"]
    head_cfg = cfg["model"]["head"]

    backbone = BACKBONES[bb_cfg["name"]](
        pretrained  = bb_cfg.get("pretrained", True),
        last_stride = bb_cfg.get("last_stride", 1),
        ibn         = bb_cfg.get("ibn", True),
    )
    neck = NECKS[neck_cfg["name"]](feat_dim=neck_cfg["feat_dim"])
    head = HEADS[head_cfg["name"]](feat_dim=neck_cfg["feat_dim"], num_classes=num_classes)

    return ReIDModel(backbone, neck, head)
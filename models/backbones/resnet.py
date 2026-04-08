"""
models/backbones/resnet.py

ResNet-50 backbone. Returns a spatial feature map (B, 2048, H, W).
Pooling lives in the neck — keep them separate so you can swap either freely.
"""

import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class IBN(nn.Module):
    """Instance-Batch Norm — mixes IN (style-invariant) + BN (discriminative)."""
    def __init__(self, planes: int, ratio: float = 0.5):
        super().__init__()
        half = int(planes * ratio)
        self.half = half
        self.IN = nn.InstanceNorm2d(half, affine=True)
        self.BN = nn.BatchNorm2d(planes - half)

    def forward(self, x):
        import torch
        a, b = torch.split(x, self.half, dim=1)
        return torch.cat([self.IN(a.contiguous()), self.BN(b.contiguous())], dim=1)


class ResNet50Backbone(nn.Module):
    """
    Args:
        pretrained:  Load ImageNet weights.
        last_stride: Set to 1 for denser feature maps (better retrieval).
        ibn:         Replace BN with IBN in layer1 & layer2.
    """

    def __init__(self, pretrained=True, last_stride=1, ibn=True):
        super().__init__()
        base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)

        if last_stride == 1:
            base.layer4[0].downsample[0].stride = (1, 1)
            base.layer4[0].conv2.stride = (1, 1)

        if ibn:
            for name in ("layer1", "layer2"):
                for block in getattr(base, name):
                    if hasattr(block, "bn1"):
                        block.bn1 = IBN(block.bn1.num_features)

        self.layer0 = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.out_channels = 2048

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x   # (B, 2048, H, W)
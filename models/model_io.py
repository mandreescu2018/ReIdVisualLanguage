from dataclasses import dataclass
import torch


@dataclass
class ModelOutput:
    logits: torch.Tensor | list    # classifier scores — ID loss + accuracy
    features: torch.Tensor | list  # metric features — triplet / center loss

@dataclass
class ModelInput:
    images: torch.Tensor
    labels: torch.Tensor | None = None
    cam_ids: torch.Tensor | None = None
    view_ids: torch.Tensor | None = None

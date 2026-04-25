from dataclasses import dataclass
import torch


@dataclass
class ModelOutput:
    logits: torch.Tensor | list    # classifier scores — ID loss + accuracy
    features: torch.Tensor | list  # metric features — triplet / center loss

"""
losses/builder.py

Instantiate and combine losses from config.
To add a new loss: implement it, drop it in this folder, register it below.
"""

import torch.nn as nn
from .id_loss      import CrossEntropyLoss, LabelSmoothCELoss
from .triplet_loss import TripletLoss
from .center_loss  import CenterLoss

ID_LOSSES = {
    "cross_entropy":   CrossEntropyLoss,
    "label_smooth_ce": LabelSmoothCELoss,
}
METRIC_LOSSES = {
    "triplet": TripletLoss,
    "none":    None,
}
AUX_LOSSES = {
    "center": CenterLoss,
    "none":   None,
}


class CombinedLoss(nn.Module):
    """
    Wraps all active losses.
    forward() returns a dict so the trainer can log each term separately.
    """

    def __init__(self, id_loss, metric_loss, aux_loss, lambda_metric=1.0, lambda_aux=0.0005):
        super().__init__()
        self.id_loss       = id_loss
        self.metric_loss   = metric_loss
        self.aux_loss      = aux_loss
        self.lam_metric    = lambda_metric
        self.lam_aux       = lambda_aux

    def forward(self, feat, feat_bn, logits, labels) -> dict:
        l_id     = self.id_loss(logits, labels)
        l_metric = self.metric_loss(feat,    labels) if self.metric_loss else logits.new_zeros(1)
        l_aux    = self.aux_loss(feat_bn,    labels) if self.aux_loss    else logits.new_zeros(1)
        total    = l_id + self.lam_metric * l_metric + self.lam_aux * l_aux
        return {"total": total, "id": l_id, "metric": l_metric, "aux": l_aux}


def build_losses(cfg: dict, num_classes: int, feat_dim: int) -> CombinedLoss:
    lc = cfg["loss"]

    id_loss = ID_LOSSES[lc["id"]["name"]](
        num_classes=num_classes,
        **{k: v for k, v in lc["id"].items() if k != "name"},
    )
    metric_cls  = METRIC_LOSSES.get(lc["metric"]["name"])
    metric_loss = metric_cls(**{k: v for k, v in lc["metric"].items() if k != "name"}) if metric_cls else None

    aux_cls  = AUX_LOSSES.get(lc["aux"]["name"])
    aux_loss = aux_cls(num_classes=num_classes, feat_dim=feat_dim) if aux_cls else None

    return CombinedLoss(
        id_loss      = id_loss,
        metric_loss  = metric_loss,
        aux_loss     = aux_loss,
        lambda_aux   = lc["aux"].get("weight", 0.0005),
    )
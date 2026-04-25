import torch.nn as nn
from .triplet_loss import TripletLoss
from .softmax_loss import CrossEntropyLabelSmooth
from .center_loss import CenterLoss

class BaseLoss:
    def __init__(self, cfg) -> None:
        self.weight = cfg["weight"]
        self.label_smooth = cfg.get("label_smooth", None)
        self.margin = cfg.get("margin", None)
        self.loss = None

    @property
    def is_center_loss(self):
        return False

    def __call__(self, outputs, target):
        return self.compute(outputs, target) * self.weight


class CenterLossWrap(BaseLoss):
    def __init__(self, cfg, num_classes, feature_dim) -> None:
        super().__init__(cfg)
        self.loss = CenterLoss(num_classes, feature_dim)

    @property
    def is_center_loss(self):
        return True

    def compute(self, outputs, target):
        feat = outputs.features
        if isinstance(feat, list):
            loss = [self.loss(f, target) for f in feat[1:]]
            loss = sum(loss) / len(loss)
            loss = 0.5 * loss + 0.5 * self.loss(feat[0], target)
        else:
            loss = self.loss(feat, target)
        return loss


class TripletLossWrap(BaseLoss):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.loss = TripletLoss(self.margin)

    def compute(self, outputs, target):
        feat = outputs.features
        if isinstance(feat, list):
            loss = [self.loss(f, target)[0] for f in feat[1:]]
            loss = sum(loss) / len(loss)
            loss = 0.5 * loss + 0.5 * self.loss(feat[0], target)[0]
        else:
            loss = self.loss(feat, target)[0]
        return loss


class CrossEntropyLossWrap(BaseLoss):
    def __init__(self, cfg, num_classes) -> None:
        super().__init__(cfg)
        if self.label_smooth == 'on':
            self.loss = CrossEntropyLabelSmooth(num_classes)
        else:
            self.loss = nn.CrossEntropyLoss()

    def compute(self, outputs, target):
        score = outputs.logits if hasattr(outputs, 'logits') else outputs
        if isinstance(score, list):
            loss = [self.loss(s, target) for s in score[1:]]
            loss = sum(loss) / len(loss)
            loss = 0.5 * loss + 0.5 * self.loss(score[0], target)
        else:
            loss = self.loss(score, target)
        return loss

class LossFactory:
    @staticmethod
    def create_loss(loss_conf, num_classes, feature_dim):
        loss_type = loss_conf["type"]
        if loss_type == "cross_entropy":
            return CrossEntropyLossWrap(loss_conf, num_classes)
        elif loss_type == "triplet":
            return TripletLossWrap(loss_conf)
        elif loss_type == "center":
            return CenterLossWrap(loss_conf, num_classes, feature_dim)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    

class ComposedLosses:
    def __init__(self, cfg, ds_info):
        """
        Initialize the ComposedLosses with configuration object.
        
        Args:
            cfg : Configuration yacs object.
            ds_info : DatasetInfo object containing dataset statistics.
        """
        self.config = cfg
        self._center_criterion = None
        self._center_loss_wrapper = None
        self.loss_functions = []

        loss_configs  = cfg.LOSS.COMPONENTS
        num_classes = ds_info.num_classes
        feature_dim = cfg.SOLVER.FEATURE_DIMENSION
        

        for loss_conf in loss_configs:
            loss_fn = LossFactory.create_loss(loss_conf, num_classes, feature_dim)
            if loss_fn.is_center_loss:
                self._center_loss_wrapper = loss_fn
                self._center_criterion = loss_fn.loss
            self.loss_functions.append(loss_fn)            

    @property
    def center_criterion(self):
        return self._center_criterion
    
    @property
    def center_loss_wrapper(self):
        """Return the center loss wrapper if present, else None."""
        return self._center_loss_wrapper
    
    def __call__(self, outputs, targets):
        """
        Compute the total weighted loss.

        Args:
            outputs: Model outputs (single or multi-output).
            targets: Ground truth targets.

        Returns:
            Total combined loss value (torch.Tensor).
        """
        total_loss = 0.0
        for loss_fn in self.loss_functions:
            total_loss += loss_fn(outputs, targets)
        return total_loss






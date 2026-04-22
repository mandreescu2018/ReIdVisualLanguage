import torch
from abc import ABC, abstractmethod

class OptimizerBase(ABC):

    def __init__(self, cfg, model):
        super().__init__()
        self.config = cfg
        self.model = model
        self._base_learning_rate = None
        self._base_weight_decay = None
        self._bias_learning_rate = None
        self._bias_weight_decay = None
        self._momentum = None
        self.params = []    

    @abstractmethod
    def get_optimizer(self, params):
        pass

    @property
    def base_learning_rate(self):
        if self._base_learning_rate is None:
            self._base_learning_rate = self.config.SOLVER.BASE_LR
        return self._base_learning_rate

    @property
    def base_weight_decay(self):
        if self._base_weight_decay is None:
            self._base_weight_decay = self.config.SOLVER.WEIGHT_DECAY
        return self._base_weight_decay

    @property
    def bias_learning_rate(self):
        if self._bias_learning_rate is None:
            self._bias_learning_rate = self.config.SOLVER.BASE_LR * self.config.SOLVER.BIAS_LR_FACTOR
        return self._bias_learning_rate

    @property
    def bias_weight_decay(self):
        if self._bias_weight_decay is None:
            self._bias_weight_decay = self.config.SOLVER.WEIGHT_DECAY_BIAS
        return self._bias_weight_decay
    
    @property
    def momentum(self):
        if self._momentum is None:
            self._momentum = self.config.SOLVER.MOMENTUM
        return self._momentum

    def _get_params(self):
        for key, value in self.model.named_parameters():
            if not value.requires_grad:
                continue
            lr = self.base_learning_rate
            weight_decay = self.base_weight_decay
            if "bias" in key:
                lr, weight_decay = self.bias_learning_rate, self.bias_weight_decay
            if self.config.SOLVER.LARGE_FC_LR and ("classifier" in key or "arcface" in key):
                lr = self.base_learning_rate * 2
                print('Using two times learning rate for fc')
            self.params.append({"params": [value], "lr": lr, "weight_decay": weight_decay})
    

class Adam_Optimizer(OptimizerBase):
    
    def get_optimizer(self):
        self._get_params()
        return torch.optim.Adam(self.params, lr=self.base_learning_rate)

class SGD_Optimizer(OptimizerBase):
    
    def get_optimizer(self):
        self._get_params()
        return torch.optim.SGD(self.params, 
                               lr=self.base_learning_rate, 
                               momentum=self.momentum)

class AdamW_Optimizer(OptimizerBase):
    
    def get_optimizer(self):
        self._get_params()
        return torch.optim.AdamW(self.params, 
                                 lr=self.base_learning_rate, 
                                 weight_decay=self.base_weight_decay)

class OptimizerFactory:
    _optimizer_classes = {
        "SGD": SGD_Optimizer,
        "Adam": Adam_Optimizer,
        "adamw": AdamW_Optimizer
    }
     
    def __init__(self, cfg, model, matcher=None):
        self.cfg = cfg
        self.model = model
        self.matcher = matcher

    def make_optimizer(self):
        if self.cfg.SOLVER.OPTIMIZER_NAME not in self._optimizer_classes:
            # return getattr(torch.optim, self.cfg.SOLVER.OPTIMIZER_NAME)(params)
            raise ValueError(f"Unknown optimizer name: {self.cfg.SOLVER.OPTIMIZER_NAME}")
        if self.cfg.SOLVER.OPTIMIZER_NAME == 'SGD_QACONV':
            return self._optimizer_classes[self.cfg.SOLVER.OPTIMIZER_NAME](self.cfg, self.model, self.matcher).get_optimizer()
        return self._optimizer_classes[self.cfg.SOLVER.OPTIMIZER_NAME](self.cfg, self.model).get_optimizer()



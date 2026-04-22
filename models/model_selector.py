import torch
from collections.abc import Mapping
from .vit_model import build_transformer, build_transformer_local
# from .vit_model_vanilla import build_transformer_vanilla
# from .backbones.mobilenet_v2 import MobileNetV2
# from .resnet_CBN import ResNetBuilder
# from .simple_model import SimpleReIDModel
# from .resnet_BoT import BagOfTricksBuilder
# from .hacnn_model import HACNNBuilder
# from .QAConv import QAConvBuilder
# from .vit_pat_model import build_part_attention_vit
# from .m3l_model import MetaResNet
from utils.device_manager import DeviceManager

model_factory = {
    'vit_transformer': build_transformer,
    # 'vit_transformer_vanilla': build_transformer_vanilla,
    # 'vit_transformer_pytorch': build_transformer_vanilla,
    # 'vit_transformer_jpm': build_transformer_local,
    # 'mobilenet_v2': MobileNetV2,
    # 'resnet50': BagOfTricksBuilder,
    # 'qaconv': QAConvBuilder,
    # 'simple_resnet50': SimpleReIDModel,
    # 'hacnn': HACNNBuilder,
    # 'vit_pat_transformer': build_part_attention_vit,
    # 'm3l': MetaResNet,

}

class ModelLoader:
    def __init__(self, cfg):
        self.cfg = cfg
        self._model = None
        self._start_epoch = 0
        self._optimizer = None
        self._optimizer_center = None
        self.scheduler = None
        self._center_criterion = None
        self._checkpoint = None

    @property
    def checkpoint(self):
        if self._checkpoint is None:
            map_location = DeviceManager.get_device()
            if self.cfg.MODEL.PRETRAIN_CHOICE == 'resume':
                # Keep compatibility with legacy checkpoints containing pickled objects.
                self._checkpoint = torch.load(self.cfg.MODEL.PRETRAIN_PATH, map_location=map_location, weights_only=False)
            elif self.cfg.MODEL.PRETRAIN_CHOICE == 'test' or self.cfg.MODEL.PRETRAIN_CHOICE == 'cross_domain':
                # Keep compatibility with legacy checkpoints containing pickled objects.
                self._checkpoint = torch.load(self.cfg.TEST.WEIGHT, map_location=map_location, weights_only=False)
        return self._checkpoint

    @property
    def model(self):
        if self._model is None:            
            self._model = model_factory[self.cfg.MODEL.NAME](self.cfg).to(DeviceManager.get_device())
        return self._model

    @property
    def start_epoch(self):
        if self._start_epoch == 0 and self.cfg.MODEL.PRETRAIN_CHOICE == 'resume':
            self._start_epoch = self.checkpoint.get('epoch', 0)        
        return self._start_epoch

    @property
    def optimizer(self):
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
    
    @property
    def optimizer_center(self):
        return self._optimizer_center
    
    @optimizer_center.setter
    def optimizer_center(self, optimizer_center):
        self._optimizer_center = optimizer_center
    
    @property
    def center_criterion(self):
        return self._center_criterion
    
    @center_criterion.setter
    def center_criterion(self, center_criterion):
        self._center_criterion = center_criterion
    
    @property
    def scheduler(self):
        return self._scheduler
    
    @scheduler.setter
    def scheduler(self, scheduler):
        self._scheduler = scheduler

    def load_param_cross(self, param_dict):
        for i in param_dict:            
            if 'classifier' in i:
                continue
            self.model.state_dict()[i].copy_(param_dict[i])

    @staticmethod
    def _extract_model_state_dict(checkpoint):
        if isinstance(checkpoint, Mapping):
            if 'model_state_dict' in checkpoint:
                return checkpoint['model_state_dict']
            return checkpoint
        if hasattr(checkpoint, 'state_dict'):
            return checkpoint.state_dict()
        return checkpoint

    def load_param(self):
        model = self.model
        checkpoint = self.checkpoint

        if self.cfg.MODEL.PRETRAIN_CHOICE == 'resume':
            model_state_dict = self._extract_model_state_dict(checkpoint)
            model.load_state_dict(model_state_dict)
            if self._optimizer is not None:
                if isinstance(checkpoint, Mapping) and 'optimizer_state_dict' in checkpoint:
                    self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            optimizer_center_state_dict = checkpoint.get('optimizer_center_state_dict', None) if isinstance(checkpoint, Mapping) else None
            center_criterion_state_dict = checkpoint.get('center_criterion_state_dict', None) if isinstance(checkpoint, Mapping) else None
            if (
                optimizer_center_state_dict is not None
                and center_criterion_state_dict is not None
                and self._optimizer_center is not None
                and self._center_criterion is not None
            ):
                self._optimizer_center.load_state_dict(optimizer_center_state_dict)
                self._center_criterion.load_state_dict(center_criterion_state_dict)
            if self._scheduler is not None:
                if isinstance(checkpoint, Mapping) and 'scheduler_state_dict' in checkpoint:
                    self._scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        elif self.cfg.MODEL.PRETRAIN_CHOICE == 'test':
            model.load_state_dict(self._extract_model_state_dict(checkpoint))
            # self.model.load_state_dict(self.checkpoint)
        elif self.cfg.MODEL.PRETRAIN_CHOICE == 'cross_domain':
            self.load_param_cross(self._extract_model_state_dict(checkpoint))
            # self.load_param_cross(self.checkpoint)
            
        
    
    

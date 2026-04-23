import torch
from torch.utils.data import DataLoader
from .market1501 import Market1501
from .dukemtmcreid import DukeMTMCreID
from .image_dataset import ImageDataset
from .data_transforms import TransformsManager
from .sampler import RandomIdentitySampler
import pandas as pd

class ReIDDataLoader:
    __factory = {
        'market1501': Market1501,
        'dukemtmc': DukeMTMCreID,
        # 'msmt17': MSMT17_Prototype, 
        # 'cuhk03': None, # datasets.CUHK03
        # 'occ_duke': OCC_DukeMTMCreID, 
    }
    def __init__(self, cfg):
        self.cfg = cfg
        self._train_dataset = None
        self._validation_dataset = None
        self._train_loader = None
        self._val_loader = None
        self.num_workers = cfg.DATALOADER.NUM_WORKERS
        
        self.transforms_manager = TransformsManager(cfg)

    @staticmethod
    def collate_fn(batch):
        imgs, pids, camids, viewids = zip(*batch)
        
        imgs = torch.stack(imgs, dim=0)
        pids = torch.tensor(pids, dtype=torch.int64)        
        camids = torch.tensor(camids, dtype=torch.int64)
        viewids = torch.tensor(viewids, dtype=torch.int64)

        return imgs, pids, camids, viewids,
        
    
    @property
    def train_dataset(self):
        if self._train_dataset is None:
            self._train_dataset = ReIDDataLoader.__factory[self.cfg.DATASETS.NAMES](self.cfg)
        return self._train_dataset

    @property
    def validation_set(self):
        if self._validation_dataset is None:
            val_set = self.train_dataset            
            self._validation_dataset = ImageDataset(pd.concat([val_set.query, val_set.gallery], ignore_index=True), self.val_transforms)
        return self._validation_dataset
        
    @property
    def train_transforms(self):
        return self.transforms_manager.image_train_transforms
    
    @property
    def val_transforms(self):
        return self.transforms_manager.image_test_transforms

    @property
    def train_loader(self):
        if self._train_loader is None:
            train_dataset = ImageDataset(self.train_dataset.train, transform=self.train_transforms)
            self._train_loader = DataLoader(
                train_dataset,
                batch_size=self.cfg.SOLVER.IMS_PER_BATCH,
                num_workers=self.num_workers,
                sampler=RandomIdentitySampler(self.train_dataset.train.itertuples(index=False, name=None), self.cfg.SOLVER.IMS_PER_BATCH, self.cfg.DATALOADER.NUM_INSTANCE),
                collate_fn=ReIDDataLoader.collate_fn,
            )
        return self._train_loader

    @property
    def val_loader(self):
        if self._val_loader is None:
            self._val_loader = DataLoader(
                self.validation_set, 
                batch_size=self.cfg.TEST.IMS_PER_BATCH, 
                shuffle=False,   
                num_workers=self.num_workers,
                collate_fn=ReIDDataLoader.collate_fn,
            )
        return self._val_loader

    @property
    def num_classes(self):
        return self.train_dataset.num_train_pids

    @property
    def cameras_number(self):
        return self.train_dataset.num_train_cams

    @property
    def track_view_num(self):
        return self.train_dataset.num_train_vids

    @property
    def query_num(self):
        return len(self.train_dataset.query)

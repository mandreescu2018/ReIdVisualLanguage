"""
datasets/base_dataset.py

Abstract base class for all re-ID datasets.
Subclass this and implement _load() for each new dataset layout.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional

from PIL import Image
from torch.utils.data import Dataset


class BaseReIDDataset(Dataset, ABC):
    """
    Every dataset must populate self.samples with (path, pid, camid) tuples
    and set self.num_pids / self.num_cameras inside _load().
    """

    def __init__(self, root: str, split: str = "train", transform: Optional[Callable] = None):
        self.root      = Path(root)
        self.split     = split
        self.transform = transform

        self.samples:      list[tuple[str, int, int]] = []
        self.num_pids:     int = 0
        self.num_cameras:  int = 0

        self._load()

    @abstractmethod
    def _load(self):
        """Parse the dataset directory and populate self.samples."""
        ...

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        path, pid, camid = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return {"image": img, "pid": pid, "camid": camid, "path": path}

    def __repr__(self):
        return (f"{self.__class__.__name__}(split={self.split}, "
                f"images={len(self.samples)}, pids={self.num_pids}, "
                f"cameras={self.num_cameras})")
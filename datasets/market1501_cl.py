"""
datasets/market1501.py

Market-1501 dataset.
Filename format: <pid>_c<camid>s<seq>_<frame>_<det>.jpg
  e.g. 0065_c1s1_001051_00.jpg  →  pid=65, camid=0 (0-indexed)

Expected root layout:
    root/
      bounding_box_train/
      query/
      bounding_box_test/    ← gallery
"""
import os
import re
from base_dataset_cl import BaseReIDDataset
from config import cfg


class Market1501(BaseReIDDataset):

    _DIRS = {
        "train":   "bounding_box_train",
        "query":   "query",
        "gallery": "bounding_box_test",
    }

    def _load(self):
        split_dir = self.root / self._DIRS[self.split]
        if not split_dir.exists():
            raise FileNotFoundError(f"Not found: {split_dir}")

        pid_set, raw = set(), []
        for fpath in sorted(split_dir.glob("*.jpg")):
            name = fpath.name
            if name.startswith("-1") or name.startswith("0000"):
                continue                            # junk / distractor images
            m = re.match(r"(\d{4})_c(\d+)", name)
            if not m:
                continue
            pid, camid = int(m.group(1)), int(m.group(2)) - 1
            pid_set.add(pid)
            raw.append((str(fpath), pid, camid))

        # Remap pids to contiguous range for training split
        if self.split == "train":
            pid2label  = {p: i for i, p in enumerate(sorted(pid_set))}
            self.samples = [(p, pid2label[pid], cam) for p, pid, cam in raw]
        else:
            self.samples = raw

        self.num_pids    = len(pid_set)
        self.num_cameras = max(c for _, _, c in self.samples) + 1


if __name__ == "__main__":
    # def __init__(self, root: str, split: str = "train", transform: Optional[Callable] = None):
    # from config import Config
    cfg.merge_from_file('configurations/Market/vit_base.yml')
    dataset_dir = os.path.join(cfg.DATASETS.ROOT_DIR, cfg.DATASETS.DIR)
    
    dataset = Market1501(dataset_dir, split="train")
    print(f"Number of images: {len(dataset)}")
    print(f"Number of person IDs: {dataset.num_pids}")
    print(f"Number of cameras: {dataset.num_cameras}")

    img = dataset[0]['image']
    pid = dataset[0]['pid']
    camid = dataset[0]['camid']
    
    # img, pid, camid, _ = dataset[0]
    print(f"First image shape: {img.size}, pid: {pid}, camid: {camid}")
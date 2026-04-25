import sys
from pathlib import Path
import time
import argparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import cfg, load_config, save_resolved_config
from utils import set_seeds
from datasets import ReIDDataLoader, DatasetInfo
from models import ModelLoader
from engine import ImageFeatureTrainer, TrainerConfig


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file", default="configurations/Market/vit_base.yml", help="path to config file", type=str
    )
    
    args = parser.parse_args()

    if args.config_file:
        load_config(args.config_file)
        save_resolved_config(cfg, args.config_file)
    set_seeds(cfg.SOLVER.SEED)

    # Data
    data_loaders = ReIDDataLoader(cfg)
    train_loader = data_loaders.train_loader
    test_loader  = data_loaders.val_loader

    dataset_info = DatasetInfo(
        num_classes=data_loaders.num_classes,
        cameras_number=data_loaders.cameras_number,
        track_view_num=data_loaders.track_view_num,
        query_num=data_loaders.query_num
    )

    # Model
    cfg.MODEL.PRETRAIN_CHOICE = 'test'
    model_loader = ModelLoader(cfg, ds_info=dataset_info)
    model_loader.load_param()
    
    trainer_config = TrainerConfig(
        model = model_loader.model,
        train_loader = train_loader,
        val_loader = test_loader
    )
    
    
    trainer = ImageFeatureTrainer(cfg, trainer_config, ds_info=dataset_info)
    start = time.perf_counter()
    
    trainer.inference()

    print(f"Time taken: {time.perf_counter() - start}")



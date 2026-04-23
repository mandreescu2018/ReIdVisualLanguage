import sys
from pathlib import Path
import time
import argparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import cfg
from utils import set_seeds
from datasets import ReIDDataLoader
# from datasets.make_dataloader_trans import make_dataloader
from models import ModelLoader
from engine import ImageFeatureTrainer, TrainerConfig


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file", default="configurations/Market/vit_base.yml", help="path to config file", type=str
    )
    
    args = parser.parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    set_seeds(cfg.SOLVER.SEED)

    # Data
    data_loaders = ReIDDataLoader(cfg)
    train_loader = data_loaders.train_loader
    test_loader  = data_loaders.val_loader

    cfg.DATASETS.NUMBER_OF_CLASSES          = data_loaders.num_classes
    cfg.DATASETS.NUMBER_OF_CAMERAS          = data_loaders.cameras_number
    cfg.DATASETS.NUMBER_OF_TRACKS           = data_loaders.track_view_num
    cfg.DATASETS.NUMBER_OF_IMAGES_IN_QUERY  = data_loaders.query_num

    
    # Model
    cfg.MODEL.PRETRAIN_CHOICE = 'test'
    model_loader = ModelLoader(cfg)
    model_loader.load_param()
    
    trainer_config = TrainerConfig(cfg)
    trainer_config.model = model_loader.model
    trainer_config.train_loader = train_loader
    trainer_config.val_loader = test_loader
    
    
    trainer = ImageFeatureTrainer(cfg, trainer_config)
    start = time.perf_counter()
    
    trainer.inference()

    print(f"Time taken: {time.perf_counter() - start}")



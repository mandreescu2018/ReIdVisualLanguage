"""
scripts/train.py — training entry point.

Usage:
    python scripts/train.py --config_file configurations/Market/vit_base.yml
    python scripts/train.py --config_file configurations/Market/vit_base.yml --resume checkpoints/reid_last.pth
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import cfg, load_config, save_resolved_config
from datasets import ReIDDataLoader, DatasetInfo
from losses import ComposedLosses
from models import ModelLoader
from engine import ImageFeatureTrainer, TrainerConfig
from solver import LearningRateScheduler
from solver.make_optimizer import OptimizerFactory
from utils import set_seeds
from utils.device_manager import DeviceManager
from functional_logging.stream_logger import StreamLogger


def main():
    parser = argparse.ArgumentParser(description="ReID Prototype Training")
    parser.add_argument(
        "--config_file", default="configurations/Market/vit_base.yml",
        help="path to config file", type=str,
    )
    args = parser.parse_args()

    if args.config_file:
        load_config(args.config_file)
        save_resolved_config(cfg, args.config_file)

    DeviceManager.set_device(cfg.MODEL.DEVICE)
    set_seeds(cfg.SOLVER.SEED)

    stream_logger = StreamLogger(cfg=cfg)
    stream_logger.info(f"Using {args.config_file} as config file")
    stream_logger.info(f"Saving model in the path: {cfg.OUTPUT_DIR}")
    stream_logger.info(cfg)

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
    model_loader = ModelLoader(cfg, ds_info=dataset_info)
    model = model_loader.model

    # Losses
    composed_loss = ComposedLosses(cfg, dataset_info)
    model_loader.center_criterion = composed_loss.center_criterion
    if model_loader.center_criterion is not None:
        model_loader.optimizer_center = torch.optim.SGD(
            model_loader.center_criterion.parameters(),
            lr=cfg.SOLVER.CENTER_LR,
        )

    # Optimizer & scheduler
    optimizer = OptimizerFactory(cfg, model).make_optimizer()
    scheduler = LearningRateScheduler(optimizer, cfg)

    model_loader.optimizer = optimizer
    model_loader.scheduler = scheduler
    model_loader.load_param()

    trainer_cfg = TrainerConfig(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=composed_loss,
        start_epoch=model_loader.start_epoch,
    )

    # Train on images
    image_trainer = ImageFeatureTrainer(cfg, trainer_cfg, ds_info=dataset_info)
    image_trainer.train()


if __name__ == "__main__":
    main()

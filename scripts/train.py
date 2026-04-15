"""
scripts/train.py — training entry point.

Usage:
    python scripts/train.py --config configurations/Market/vit_base.yml
    python scripts/train.py --config configurations/Market/vit_base.yml --resume checkpoints/reid_last.pth
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import cfg
from datasets import ReIDDataLoader
from losses import LossComposer
from models import ModelLoader
from processors.processor_standard import ProcessorStandard
from solver import LearningRateScheduler
from solver.make_optimizer import OptimizerFactory
from utils import set_seeds
from functional_logging.stream_logger import StreamLogger


def main():
    parser = argparse.ArgumentParser(description="ReID Prototype Training")
    parser.add_argument(
        "--config_file", default="configurations/Market/vit_base.yml",
        help="path to config file", type=str,
    )
    args = parser.parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)

    set_seeds(cfg.SOLVER.SEED)

    stream_logger = StreamLogger(cfg=cfg)
    stream_logger.info(f"Using {args.config_file} as config file")
    stream_logger.info(f"Saving model in the path: {cfg.OUTPUT_DIR}")
    stream_logger.info(cfg)

    # Data
    data_loaders = ReIDDataLoader(cfg)
    train_loader = data_loaders.train_dataloader
    test_loader  = data_loaders.val_loader

    cfg.DATASETS.NUMBER_OF_CLASSES          = data_loaders.num_classes
    cfg.DATASETS.NUMBER_OF_CAMERAS          = data_loaders.cameras_number
    cfg.DATASETS.NUMBER_OF_TRACKS           = data_loaders.track_view_num
    cfg.DATASETS.NUMBER_OF_IMAGES_IN_QUERY  = data_loaders.query_num

    # Model
    model_loader = ModelLoader(cfg)

    # Losses
    composed_loss = LossComposer(cfg)

    # Optimizer & scheduler
    optimizer = OptimizerFactory(cfg, model_loader.model).make_optimizer()
    scheduler = LearningRateScheduler(optimizer, cfg)

    model_loader.optimizer = optimizer
    model_loader.scheduler = scheduler
    model_loader.load_param()

    # Processor (train + eval loop)
    proc = ProcessorStandard(
        cfg,
        model_loader.model,
        train_loader,
        test_loader,
        model_loader.optimizer,
        model_loader.optimizer_center,
        model_loader.center_criterion,
        composed_loss,
        model_loader.scheduler,
        start_epoch=model_loader.start_epoch,
    )
    proc.train()


if __name__ == "__main__":
    main()

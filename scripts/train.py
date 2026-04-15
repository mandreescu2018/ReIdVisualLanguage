"""
scripts/train.py — training entry point.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml --resume checkpoints/default/reid_last.pth
"""

import argparse, random, sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader




sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import cfg
from datasets        import ReIDDataLoader
from engine          import Trainer, Evaluator #, LoggerCallback, EvalCheckpointCallback
from losses          import build_losses, LossComposer
# from models          import build_model
from utils           import set_seeds
from functional_logging.stream_logger import StreamLogger
from models import ModelLoader
from processors.processor_selector import get_processor
from solver.make_optimizer import OptimizerFactory
from solver import LearningRateScheduler


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="ReID Prototype Training")
    parser.add_argument(
        "--config_file", default="configurations/Market/vit_base.yml", help="path to config file", type=str
    )
    
    args = parser.parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    # cfg.freeze()
    set_seeds(cfg.SOLVER.SEED)

    # logger
    stream_logger = StreamLogger(cfg=cfg)
    # logger.info(f"Using {DeviceManager.get_device()} device")
    stream_logger.info(f"Using {args.config_file} as config file")
    stream_logger.info(f"Saving model in the path :{cfg.OUTPUT_DIR}")
    stream_logger.info(cfg)

    # Data Loaders

    data_loaders  = ReIDDataLoader(cfg)    
    train_loader = data_loaders.train_dataloader
    test_loader = data_loaders.val_loader

    cfg.DATASETS.NUMBER_OF_CLASSES = data_loaders.num_classes
    cfg.DATASETS.NUMBER_OF_CAMERAS = data_loaders.cameras_number
    cfg.DATASETS.NUMBER_OF_TRACKS = data_loaders.track_view_num
    cfg.DATASETS.NUMBER_OF_IMAGES_IN_QUERY = data_loaders.query_num
    
    # Model
    model_loader = ModelLoader(cfg)

    # Losses
    composed_loss = LossComposer(cfg)

    # Optimizers
    optimizer_fact = OptimizerFactory(cfg, model_loader.model)
    optimizer = optimizer_fact.make_optimizer()
    scheduler = LearningRateScheduler(optimizer, cfg)
    
    model_loader.optimizer = optimizer
    model_loader.scheduler = scheduler

    model_loader.load_param()

    proc = get_processor(cfg)

    kwargs = {}

    proc = proc(cfg, 
                model_loader.model, 
                train_loader, 
                test_loader,
                model_loader.optimizer,
                model_loader.optimizer_center,
                model_loader.center_criterion,
                composed_loss,
                model_loader.scheduler,
                start_epoch=model_loader.start_epoch,
                **kwargs)
    proc.train()

    # ap = argparse.ArgumentParser()
    # ap.add_argument("--config",  required=True)
    # ap.add_argument("--resume",  default="")
    # args = ap.parse_args()

    # cfg    = load_cfg(args.config)
    # device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    # seed   = cfg.get("seed", 42)
    # random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    # out_dir = cfg.get("output_dir", "checkpoints/default")
    # logger  = get_logger(log_file=Path(out_dir) / "train.log")
    # logger.info(f"config={args.config}  device={device}")

    # # ── data ──
    # nw  = cfg["data"].get("num_workers", 4)
    # smp = cfg["sampler"]
    # train_set     = build_dataset(cfg, "train")
    # query_set     = build_dataset(cfg, "query")
    # gallery_set   = build_dataset(cfg, "gallery")
    # logger.info(train_set)

    # train_loader   = DataLoader(train_set,
    #                             batch_sampler=PKSampler(train_set, smp["P"], smp["K"], smp["num_iter"]),
    #                             num_workers=nw, pin_memory=True)
    # query_loader   = DataLoader(query_set,   batch_size=128, shuffle=False, num_workers=nw)
    # gallery_loader = DataLoader(gallery_set, batch_size=128, shuffle=False, num_workers=nw)

    # # ── model + losses ──
    # feat_dim  = cfg["model"]["neck"]["feat_dim"]
    # model     = build_model(cfg, num_classes=train_set.num_pids).to(device)
    # criterion = build_losses(cfg, num_classes=train_set.num_pids, feat_dim=feat_dim).to(device)
    # logger.info(f"model ready  |  {train_set.num_pids} identities")

    # ── optimiser + scheduler ──
    # total_ep = cfg["scheduler"]["num_epochs"]
    # opt, center_opt, sched = build_optimizer_and_scheduler(cfg, model, criterion, total_ep)

    # # ── resume ──
    # start_epoch = 0
    # if args.resume:
    #     ckpt        = load_checkpoint(args.resume, model, opt, sched, device=device)
    #     start_epoch = ckpt.get("epoch", 0) + 1
    #     logger.info(f"resumed from epoch {start_epoch}")

    # # ── callbacks ──
    # evaluator = Evaluator(model, device,
    #                       metric=cfg["eval"].get("metric", "cosine"),
    #                       ranks=cfg["eval"].get("ranks", [1, 5, 10, 20]))
    # callbacks = [
    #     LoggerCallback(logger),
    #     EvalCheckpointCallback(evaluator, query_loader, gallery_loader,
    #                            out_dir, eval_every=cfg["eval"]["eval_every"],
    #                            logger=logger),
    # ]

    # ── train ──
    # trainer = Trainer(model, criterion, opt, sched, train_loader, device, center_opt)
    # trainer.run(num_epochs=total_ep - start_epoch, start_epoch=start_epoch, callbacks=callbacks)
    # logger.info("done.")


if __name__ == "__main__":
    main()

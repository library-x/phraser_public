import argparse
import copy
import time

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt, gridspec
from scipy.ndimage import gaussian_filter1d
from torch.utils.tensorboard import SummaryWriter


from phraser.modules.phraser import PhraserModel
import sys

from phraser.utils.dist import setup_distributed_device

from phraser.utils.data_loader import get_loaders
from phraser.utils.metric_logger import MetricLogger
from phraser.utils.train import cosine_lr, train, evaluate, project
from phraser.utils.utils import print_params
from phraser.paths import RUNS


def train_model(config):
    device = setup_distributed_device(config)

    np.random.seed(config.seed+config.rank)
    torch.manual_seed(config.seed+config.rank)
    model_timestamp = int(time.time())


    train_dataloader, test_dataloader = get_loaders(config)

    model = PhraserModel(dim_embed=128, num_heads=16)  # flagship size (axfix retrain)

    optimizer = torch.optim.AdamW(model.parameters_grad(), lr=config.lr, weight_decay=config.weight_decay, betas=(0.9, 0.95), eps=1e-8)

    model = model.to(device)
    model.muq.to("cpu")

    for param in model.muq.parameters():
        param.requires_grad = False

    if args.distributed:
        model.predictor = torch.nn.parallel.DistributedDataParallel(model.predictor, device_ids=[device], find_unused_parameters=True)

    print("IF", device, args.batch_size)

    if args.is_master:
        print_params(model)

        writer = SummaryWriter(os.path.join(RUNS, f'{config.exp_name}_{model_timestamp}'))
        hparams = copy.deepcopy( config.__dict__)

        writer.add_hparams(hparams, {}, run_name=".")

        train_epoch_logger = MetricLogger(tensorboard=writer, log_frequency=-1, name="train_epoch")
        train_batch_logger = MetricLogger(tensorboard=writer, log_frequency=5, name="train_batch")

        val_epoch_logger = MetricLogger(tensorboard=writer, log_frequency=-1, name="val_epoch")
        val_batch_logger = MetricLogger(tensorboard=writer, log_frequency=5, name="val_batch")

        images_logger = MetricLogger(tensorboard=writer, log_frequency=5, name="images")

        train_loggers = [train_epoch_logger, train_batch_logger]
        val_loggers = [val_epoch_logger, val_batch_logger]

        val_epoch_logger.curr_index = 0

    else:
        train_loggers = []
        val_loggers = []

        images_logger = None

    scheduler = cosine_lr(optimizer, config.lr, config.warmup, config.epochs)
    best_loss = float("inf")

    project_batch = next(iter(test_dataloader))
    project(model=model, project_batch=project_batch, config=config, device=device, logger=images_logger)


    for epoch in range(config.epochs + 1):
        # Reset pattern indices for collate functions at the start of each epoch
        if hasattr(train_dataloader.collate_fn, 'set_epoch'):
            train_dataloader.collate_fn.set_epoch(epoch)
        if hasattr(test_dataloader.collate_fn, 'set_epoch'):
            test_dataloader.collate_fn.set_epoch(epoch)
            
        scheduler(epoch)

        print(f"Train: {epoch}")
        train(model, train_dataloader, optimizer, epoch, device, loggers=train_loggers, cfg=config)

        print(f"Eval: {epoch}")
        evaluate(model, test_dataloader, optimizer, epoch, device, loggers=val_loggers, cfg=config)
        project(model, project_batch, config, device, images_logger)

        if args.is_master:
            train_epoch_logger.update(lr=optimizer.param_groups[0]['lr'])
            train_epoch_logger.log()

            val_epoch_logger.log(reset=False)

            if best_loss > val_epoch_logger.get("loss"):
                best_loss = val_epoch_logger.get("loss")
                model.save(writer.log_dir, config, "best", epoch=epoch, timestamp=model_timestamp)

            if epoch % config.save_freq == 0:
                model.save(writer.log_dir, config, f"epoch_{epoch}", epoch=epoch, timestamp=model_timestamp)

            val_epoch_logger.reset()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training configuration parser")

    # Debugging
    parser.add_argument('--debug', action='store_true', help='Enable debugging mode.')

    # Experiment Configuration
    parser.add_argument('--exp_name', type=str, default='phaser_aug_true64', help='Name of the experiment.')

    # Data Configurations
    parser.add_argument('--sample_rate', type=int, default=24000, help='Sample rate for audio data.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--dim_embed', type=int, default=64, help='Dimension of the embedding vector.')

    # Training Configurations
    parser.add_argument('--segment_size', type=float, default=300.0, help='Segment size for training.')
    parser.add_argument('--lr', type=float, default=0.002, help='Learning rate for the optimizer.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--warmup', type=float, default=20, help='Warmup duration in epochs.')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay value for regularization.')
    parser.add_argument('--validation_interval_epochs', type=int, default=1, help='Validation interval in epochs.')
    parser.add_argument('--gradient_clip', type=float, default=1.0, help='Gradient clipping threshold.')

    # Loss Configurations
    parser.add_argument('--loss_weight_section', type=float, default=20.0, help='Loss weight for the section component.')
    parser.add_argument('--loss_weight_function', type=float, default=0.1, help='Loss weight for the function component.')

    # Distributed Configuration
    parser.add_argument('--distributed', action='store_true', help='Enable distributed training.')
    parser.add_argument('--world_size', type=int, default=1, help='Number of distributed world nodes.')
    parser.add_argument('--rank', type=int, default=0, help='Global rank in distributed training.')
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training.')

    # Miscellaneous Configurations
    parser.add_argument('--seed', type=int, default=2137, help='Random seed for reproducibility.')
    parser.add_argument('--min_hops_per_beat', type=int, default=5, help='Minimum number of hops per beat.')
    parser.add_argument('--save_freq', type=int, default=1, help='Model save frequency during training.')

    args, unknown = parser.parse_known_args()

    train_model(args)

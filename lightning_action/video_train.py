"""Training functionality for video action segmentation models.

This module provides the main training loop and utilities for training
VideoSegmenter models using PyTorch Lightning.

Key functions:
- train_video(): Main entry point for training a video segmentation model
- reset_seeds(): Ensure reproducibility across runs
- get_callbacks(): Configure Lightning callbacks for checkpointing, etc.

Example usage:
    config = load_config('config.yaml')
    model = VideoSegmenter(config)
    trained_model = train_video(config, model, output_dir='runs/experiment1')
"""

import logging
import multiprocessing as mp
import os
import random
from pathlib import Path
from typing import Any, Optional

import lightning as pl
import numpy as np
import torch
import yaml
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities import rank_zero_only
from typeguard import typechecked

from lightning_action.data.video_datamodule import VideoDataModule
from lightning_action.models.video_segmenter import VideoSegmenter
from lightning_action.train import reset_seeds, get_callbacks

logger = logging.getLogger(__name__)


@typechecked
def train_video(
    config: dict[str, Any],
    model: VideoSegmenter,
    output_dir: str | Path,
) -> pl.LightningModule:
    """Train a video action segmentation model.
    
    Args:
        config: Configuration dictionary containing data, training, model,
            and optimizer settings.
        model: Initialized VideoSegmenter model to train.
        output_dir: Directory for saving checkpoints, logs, and config.
    
    Returns:
        Trained model.
    
    Raises:
        ValueError: If required config sections are missing.
    """
    output_dir = Path(output_dir)

    # Set random seeds for reproducibility
    seed = config.get('training', {}).get('seed', 0)
    reset_seeds(seed=seed)

    # Configure multiprocessing for DALI
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    # Configure NCCL for distributed training stability
    if 'NCCL_TIMEOUT' not in os.environ:
        os.environ['NCCL_TIMEOUT'] = '1800'
    if 'NCCL_ASYNC_ERROR_HANDLING' not in os.environ:
        os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'

    # Detect available GPUs
    num_gpus = torch.cuda.device_count()
    accelerator = 'gpu' if num_gpus > 0 else 'cpu'
    devices = num_gpus if num_gpus > 0 else 'auto'

    # Validate configuration
    if 'data' not in config:
        raise ValueError("Configuration must contain 'data' section")
    if 'training' not in config:
        raise ValueError("Configuration must contain 'training' section")

    data_config = config['data']
    training_config = config['training']
    model_config = config.get('model', {})

    # Create data module
    datamodule = VideoDataModule(
        data_config=data_config,
        sequence_length=training_config.get('sequence_length', 128),
        batch_size=training_config.get('batch_size', 1),
        num_workers=training_config.get('num_workers', 0),
        train_probability=training_config.get('train_probability', 0.95),
        val_probability=training_config.get('val_probability', 0.05),
        seed=seed,
        model_config=model_config,
    )

    # Setup data splits (video-level)
    datamodule.setup('fit')

    # Check if validation is enabled
    has_val_data = datamodule.validation_enabled
    
    # For multi-GPU, ensure enough validation videos
    if has_val_data and num_gpus > 1:
        num_val_videos = len(datamodule.val_video_paths) if datamodule.val_video_paths else 0
        if num_val_videos < num_gpus:
            logger.warning(
                f"Not enough validation videos ({num_val_videos}) for {num_gpus} GPUs. "
                "Disabling validation."
            )
            has_val_data = False

    # Configure validation batch limiting
    if has_val_data:
        limit_val_batches = training_config.get('limit_val_batches', 1.0)
    else:
        limit_val_batches = 0

    # Get class weights from dataset
    weight_classes = data_config.get('weight_classes', True)
    if weight_classes:
        class_weights = datamodule.dataset.class_weights
        logger.info(f"Using class weights: {class_weights}")
    else:
        class_weights = None
    
    # Update config with computed class weights
    if 'model' not in config:
        config['model'] = {}
    config['model']['class_weights'] = class_weights
    if hasattr(model, 'config'):
        model.config['model']['class_weights'] = class_weights

    # Store label names in config
    label_names = datamodule.get_label_names()
    if len(label_names) > 0:
        config['data']['label_names'] = label_names
        if hasattr(model, 'config'):
            model.config['data']['label_names'] = label_names

    # Save configuration (only on rank 0)
    if rank_zero_only.rank == 0:
        (output_dir / 'config.yaml').parent.mkdir(exist_ok=True, parents=True)
        with open(output_dir / 'config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

    # Configure distributed training strategy
    if num_gpus > 1:
        try:
            get_ipython()
            strategy = 'ddp_notebook'
        except NameError:
            strategy = 'ddp'
    else:
        strategy = 'auto'

    # Configure Lightning Trainer
    trainer_config = {
        'accelerator': accelerator,
        'devices': devices,
        'strategy': strategy,
        'max_epochs': training_config.get('num_epochs', 100),
        'limit_val_batches': limit_val_batches,
        'precision': '16-mixed' if accelerator == 'gpu' else '32-true',
        'enable_checkpointing': training_config.get('checkpointing', True),
        'callbacks': get_callbacks(
            checkpointing=training_config.get('checkpointing', True),
            lr_monitor=training_config.get('lr_monitor', True),
            ckpt_every_n_epochs=training_config.get('ckpt_every_n_epochs', None),
            early_stopping=training_config.get('early_stopping', False),
            early_stopping_patience=training_config.get('early_stopping_patience', 10),
        ),
        'logger': TensorBoardLogger(
            save_dir=str(output_dir),
            name='',
            version='',
        ),
        'enable_progress_bar': training_config.get('enable_progress_bar', True),
        'num_sanity_val_steps': 0,
        'sync_batchnorm': training_config.get('sync_batchnorm', False),
        'accumulate_grad_batches': training_config.get('accumulate_grad_batches', 1),
        'reload_dataloaders_every_n_epochs': training_config.get('reload_dataloaders_every_n_epochs', 0),
        'use_distributed_sampler': False,
    }
    
    trainer = pl.Trainer(**trainer_config)
    trainer.fit(model, datamodule)

    return model

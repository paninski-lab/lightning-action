"""Training functionality for video action segmentation models.

This module provides the main training loop and utilities for training
VideoSegmenter models using PyTorch Lightning with NVIDIA DALI for
GPU-accelerated video loading.

Key functions:
- train_video: Main entry point for training a video segmentation model

The video pipeline differs from the CSV pipeline in several ways:
- Uses VideoDataModule with NVIDIA DALI for GPU video decoding
- Supports multi-GPU training with DDP
- Uses mixed precision training by default
- Class weights are computed by VideoDataset

Example usage:
    config = load_config('config.yaml')
    model = VideoSegmenter(config)
    trained_model = train_video(config, model, output_dir='runs/experiment1')
"""

import logging
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any

import lightning as pl
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities import rank_zero_only
from typeguard import typechecked

from lightning_action.data.video_datamodule import VideoDataModule
from lightning_action.models.video_segmenter import VideoSegmenter

# Import shared utilities from train_utils
from lightning_action.train_utils import (
    reset_seeds,
    get_callbacks,
    validate_config,
    update_config_with_class_weights,
    update_config_with_label_names,
    save_config,
    get_callbacks_from_config,
)

# Re-export for backward compatibility
__all__ = [
    'train_video',
]

logger = logging.getLogger(__name__)


@typechecked
def train_video(
    config: dict[str, Any],
    model: VideoSegmenter,
    output_dir: str | Path,
) -> pl.LightningModule:
    """Train a video action segmentation model.
    
    This function handles the complete training pipeline for video models:
    1. Configuration validation and seed setup
    2. DALI/multiprocessing configuration
    3. VideoDataModule creation and setup
    4. Class weight computation
    5. Lightning Trainer configuration
    6. Model training
    
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

    # Set random seeds for reproducibility (shared utility)
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

    # Validate configuration (shared utility)
    validate_config(config, required_sections=['data', 'training'])

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

    # Configure validation batch limiting and checkpoint monitoring
    if has_val_data:
        limit_val_batches = training_config.get('limit_val_batches', 1.0)
        checkpoint_monitor = 'val_loss'
    else:
        limit_val_batches = 0
        checkpoint_monitor = 'train_loss'

    # Get class weights from dataset
    # Note: Video pipeline uses VideoDataset.class_weights which pre-computes weights
    weight_classes = data_config.get('weight_classes', True)
    if weight_classes:
        class_weights = datamodule.dataset.class_weights
        logger.info(f"Using class weights: {class_weights}")
    else:
        class_weights = None
    
    # Update config and model with class weights (shared utility)
    update_config_with_class_weights(config, model, class_weights)

    # Store label names in config (shared utility)
    label_names = datamodule.get_label_names()
    update_config_with_label_names(config, model, label_names)

    # Save configuration (shared utility - only on rank 0)
    save_config(config, output_dir)

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
        # Use shared utility for callbacks
        'callbacks': get_callbacks_from_config(
            {**training_config, 'early_stopping': training_config.get('early_stopping', False) and has_val_data},
            monitor=checkpoint_monitor,
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

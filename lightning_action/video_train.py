"""Training functionality for lightning-action video models.

This module provides training functions for video action segmentation models.
"""

import logging
import os
import random
from pathlib import Path
from typing import Any

import lightning as pl
import numpy as np
import yaml
import torch
from lightning.pytorch.utilities import rank_zero_only
from typeguard import typechecked
from tqdm import tqdm
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from lightning_action import __version__
from lightning_action.data.video_datamodule import VideoDataModule
from lightning_action.models.video_segmenter import VideoSegmenter

logger = logging.getLogger(__name__)

@typechecked
def train_video(
    config: dict[str, Any],
    model: VideoSegmenter,
    output_dir: str | Path,
) -> pl.LightningModule:
    """Train a Lightning model for video action segmentation.

    Args:
        config: configuration dictionary with data, model, and training settings
        model: VideoSegmenter model to train
        output_dir: directory to save outputs and checkpoints

    Returns:
        trained model

    Raises:
        ValueError: if required configuration keys are missing
        FileNotFoundError: if data directory doesn't exist
    """
    output_dir = Path(output_dir)

    # log basic info
    if rank_zero_only.rank == 0:
        logger.info(f'Output directory: {output_dir}')
        logger.info(f'Model type: {type(model)}')

    # reset seeds for reproducibility
    seed = config.get('training', {}).get('seed', 0)
    reset_seeds(seed=seed)

    # log configuration
    if rank_zero_only.rank == 0:
        logger.info('Configuration:\n' + yaml.dump(config, default_flow_style=False))

    # validate required config sections
    if 'data' not in config:
        raise ValueError("Configuration must contain 'data' section")
    if 'training' not in config:
        raise ValueError("Configuration must contain 'training' section")

    data_config = config['data']
    training_config = config['training']
    model_config = config.get('model', {})

    # Determine num_lags based on backbone
    backbone_type = model_config.get('backbone', 'dtcn').lower()
    if backbone_type in ['dtcn', 'dilatedtcn', 'temporalmlp']:
        num_lags = model_config.get('num_lags', 1)
    else:
        num_lags = 0 if not model_config.get('bidirectional', False) else model_config.get('num_lags', 1)
    data_config['num_lags'] = num_lags

    # create datamodule
    datamodule = VideoDataModule(
        data_config=data_config,
        sequence_length=training_config.get('sequence_length', 128),
        batch_size=training_config.get('batch_size', 1),
        num_workers=training_config.get('num_workers', 0),
        train_probability=training_config.get('train_probability', 0.95),
        val_probability=training_config.get('val_probability', 0.05),
        seed=seed,
    )

    # setup datamodule to access datasets
    datamodule.setup('fit')

    # log dataset sizes
    if rank_zero_only.rank == 0:
        logger.info(f"Training dataset size: {len(datamodule.dataset_train)} chunks")
        logger.info(f"Validation dataset size: {len(datamodule.dataset_val)} chunks")

    # compute class weights
    weight_classes = data_config.get('weight_classes', True)
    if weight_classes:
        logger.info("Computing class weights...")
        class_weights = compute_class_weights(
            datamodule,
            ignore_index=data_config.get('ignore_index', -100),
        )

        # update model configuration with class weights
        if hasattr(model, 'config'):
            if 'model' not in model.config:
                model.config['model'] = {}
            model.config['model']['class_weights'] = class_weights

        # also store in main config for saving
        config['model']['class_weights'] = class_weights
    else:
        logger.info("Class weighting disabled")
        config['model']['class_weights'] = None

    # save label names to config
    label_names = datamodule.get_label_names()
    if len(label_names) > 0:
        config['data']['label_names'] = label_names
        model.config['data']['label_names'] = label_names

    # save outputs
    if rank_zero_only.rank == 0:
        (output_dir / 'config.yaml').parent.mkdir(exist_ok=True, parents=True)
        with open(output_dir / 'config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

    # ----------------------------------------------------------------------------------
    # Set up trainer
    # ----------------------------------------------------------------------------------

    logger.info("Setting up trainer...")

    # trainer configuration
    trainer_config = {
        'accelerator': training_config.get('device', 'cpu'),
        'devices': 1,
        'max_epochs': training_config.get('num_epochs', 100),
        'precision': training_config.get('precision', '32-true'),
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
    }

    # create trainer
    trainer = pl.Trainer(**trainer_config)

    # ----------------------------------------------------------------------------------
    # Train model
    # ----------------------------------------------------------------------------------

    num_epochs = training_config.get('num_epochs', 100)
    logger.info(f"Starting training for {num_epochs} epochs...")

    # train model
    trainer.fit(model, datamodule)

    logger.info("Training completed")
    return model

@typechecked
def reset_seeds(seed: int = 0):
    """Reset random seeds for reproducibility.
    
    Args:
        seed: seed value to use
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed)

@typechecked
def compute_class_weights(datamodule: pl.LightningDataModule, ignore_index: int = -100) -> list[float]:
    """Compute class weights for imbalanced dataset incrementally to avoid memory issues.
    
    Computes weights inversely proportional to class frequency, with the most
    frequent class having weight 1.0.
    
    Args:
        datamodule: Lightning DataModule with datasets
        ignore_index: class index to ignore
        
    Returns:
        list of class weights
    """
    logger.info("Computing class weights from training data...")
    
    # ensure datamodule is set up
    if datamodule.dataset_train is None:
        datamodule.setup('fit')
    
    counts = np.zeros(datamodule.num_classes)
    
    # Cache for loaded labels to avoid reloading .npy files multiple times
    label_cache = {}
    
    # Get train indices
    train_indices = datamodule.dataset_train.indices
    
    # Loop over train chunks to count labels (without loading videos)
    for idx in tqdm(train_indices, desc="Computing class weights"):
        video, chunk_idx = datamodule.dataset.video_chunks[idx]
        label_path = os.path.join(datamodule.dataset.labels_dir, video.replace('.mp4', '.npy'))
        
        # Load labels if not cached
        if video not in label_cache:
            label_cache[video] = np.load(label_path)
        
        labels = label_cache[video]
        total_frames = len(labels)
        
        # Compute chunk window (mirroring VideoDataset.__getitem__)
        stride = datamodule.dataset.chunk_size
        start_frame = chunk_idx * stride
        end_frame = min(start_frame + datamodule.dataset.chunk_size, total_frames)
        
        # Slice labels
        labels_slice = labels[start_frame:end_frame]
        
        # Handle one-hot if necessary
        if labels_slice.ndim > 1 and labels_slice.shape[-1] > 1:
            labels_slice = np.argmax(labels_slice, axis=-1)
        
        # Filter ignore_index
        labels_slice = labels_slice[labels_slice != ignore_index]
        
        # Count unique classes
        unique, batch_counts = np.unique(labels_slice, return_counts=True)
        for cls, count in zip(unique, batch_counts):
            if 0 <= cls < datamodule.num_classes:
                counts[cls] += count
    
    # compute class weights
    if np.sum(counts) == 0:
        logger.warning("No labeled examples found, using uniform weights")
        return [1.0] * datamodule.num_classes
    
    max_count = np.max(counts)
    class_weights = max_count / (counts + 1e-10)
    class_weights[counts == 0] = 0.0
    
    logger.info(f"Class counts: {counts}")
    logger.info(f"Class weights: {class_weights}")
    
    return class_weights.tolist()

@typechecked
def get_callbacks(
    checkpointing: bool = True,
    lr_monitor: bool = True,
    ckpt_every_n_epochs: int | None = None,
    early_stopping: bool = False,
    early_stopping_patience: int = 10,
) -> list[pl.Callback]:
    """Get Lightning callbacks for training.
    
    Args:
        checkpointing: enable model checkpointing
        lr_monitor: monitor learning rate
        ckpt_every_n_epochs: save checkpoint every N epochs
        early_stopping: enable early stopping
        early_stopping_patience: patience for early stopping
        
    Returns:
        list of callbacks
    """
    callbacks = []
    
    if lr_monitor:
        callbacks.append(LearningRateMonitor(logging_interval='epoch'))
    
    if checkpointing:
        callbacks.append(ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            filename='{epoch}-{step}-best',
            save_top_k=1,
        ))
    
    if ckpt_every_n_epochs is not None:
        callbacks.append(ModelCheckpoint(
            monitor=None,
            every_n_epochs=ckpt_every_n_epochs,
            save_top_k=-1,
            filename='{epoch}-{step}',
        ))
    
    if early_stopping:
        callbacks.append(EarlyStopping(
            monitor='val_loss',
            mode='min',
            patience=early_stopping_patience,
            verbose=True,
        ))
    
    return callbacks

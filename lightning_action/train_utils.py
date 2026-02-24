"""Shared training utilities for lightning-action.

This module provides common functionality shared between the CSV pipeline
(train.py) and video pipeline (video_train.py) to reduce code duplication
while maintaining pipeline-specific behavior where needed.

Functions:
- reset_seeds: Reproducibility through random seed control
- get_callbacks: Configure Lightning callbacks for training
- validate_config: Validate required config sections
- update_config_with_class_weights: Handle class weight updates
- update_config_with_label_names: Store label names in config
- save_config: Save configuration to YAML file
"""

import logging
import os
import random
from pathlib import Path
from typing import Any, Optional

import lightning.pytorch as pl
import numpy as np
import torch
import yaml
from lightning.pytorch.utilities import rank_zero_only
from typeguard import typechecked

logger = logging.getLogger(__name__)


# =============================================================================
# Reproducibility
# =============================================================================

@typechecked
def reset_seeds(seed: int = 0) -> None:
    """Reset all random seeds for reproducibility.

    Sets seeds for Python's random, NumPy, and PyTorch. Also configures
    cuDNN for deterministic behavior (at the cost of some performance).

    Args:
        seed: Random seed value (default: 0).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# =============================================================================
# Lightning Callbacks
# =============================================================================

@typechecked
def get_callbacks(
    checkpointing: bool = True,
    lr_monitor: bool = True,
    ckpt_every_n_epochs: Optional[int] = None,
    early_stopping: bool = False,
    early_stopping_patience: int = 10,
    monitor: Optional[str] = 'val_loss',  # New parameter
) -> list[pl.Callback]:
    """Get Lightning callbacks for training.

    Args:
        checkpointing: Whether to enable model checkpointing (saves best model).
        lr_monitor: Whether to monitor and log learning rate.
        ckpt_every_n_epochs: Save checkpoint every N epochs (None to disable).
        early_stopping: Whether to enable early stopping based on val_loss.
        early_stopping_patience: Patience for early stopping (epochs without improvement).
        monitor: Metric to monitor for checkpointing. Use None to save last model,
            'val_loss' (default) for best validation, or 'train_loss' if no validation.

    Returns:
        List of configured Lightning callbacks.
    """
    callbacks = []

    # Learning rate monitoring
    if lr_monitor:
        lr_monitor_cb = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
        callbacks.append(lr_monitor_cb)

    # Model checkpointing - save best model based on monitored metric
    if checkpointing:
        ckpt_best_callback = pl.callbacks.ModelCheckpoint(
            monitor=monitor,  # Can be None, 'val_loss', 'train_loss', etc.
            mode='min' if monitor else None,
            filename='{epoch}-{step}-best',
            save_top_k=1 if monitor else None,
            save_last=True if monitor is None else False,  # Save last if no monitor
        )
        callbacks.append(ckpt_best_callback)

    # Periodic checkpointing (save every N epochs regardless of performance)
    if ckpt_every_n_epochs is not None:
        ckpt_callback = pl.callbacks.ModelCheckpoint(
            monitor=None,
            every_n_epochs=ckpt_every_n_epochs,
            save_top_k=-1,
            filename='{epoch}-{step}',
        )
        callbacks.append(ckpt_callback)

    # Early stopping
    if early_stopping:
        early_stop_callback = pl.callbacks.EarlyStopping(
            monitor=monitor,
            mode='min',
            patience=early_stopping_patience,
            verbose=True,
        )
        callbacks.append(early_stop_callback)

    return callbacks


# =============================================================================
# Config Validation and Updates
# =============================================================================

@typechecked
def validate_config(config: dict[str, Any], required_sections: list[str]) -> None:
    """Validate that required configuration sections are present.

    Args:
        config: Configuration dictionary to validate.
        required_sections: List of required top-level section names.

    Raises:
        ValueError: If any required section is missing.
    """
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Configuration must contain '{section}' section")


@typechecked
def update_config_with_class_weights(
    config: dict[str, Any],
    model: pl.LightningModule,
    class_weights: Optional[list[float]],
) -> None:
    """Update config and model with computed class weights.

    This function handles the common pattern of storing class weights in both
    the config dictionary (for persistence) and the model's config (for use
    during training).

    Args:
        config: Main configuration dictionary.
        model: Lightning model with an optional 'config' attribute.
        class_weights: Computed class weights, or None if weighting is disabled.
    """
    # Ensure model section exists in config
    if 'model' not in config:
        config['model'] = {}

    # Store in main config
    config['model']['class_weights'] = class_weights

    # Store in model's config if it has one
    if hasattr(model, 'config'):
        if 'model' not in model.config:
            model.config['model'] = {}
        model.config['model']['class_weights'] = class_weights


@typechecked
def update_config_with_label_names(
    config: dict[str, Any],
    model: pl.LightningModule,
    label_names: list[str],
) -> None:
    """Update config and model with label names.

    Stores label names in both the config dictionary and the model's config
    if the label_names list is non-empty.

    Args:
        config: Main configuration dictionary.
        model: Lightning model with an optional 'config' attribute.
        label_names: List of label/class names.
    """
    if len(label_names) > 0:
        # Ensure data section exists
        if 'data' not in config:
            config['data'] = {}

        config['data']['label_names'] = label_names

        if hasattr(model, 'config'):
            if 'data' not in model.config:
                model.config['data'] = {}
            model.config['data']['label_names'] = label_names


@rank_zero_only
@typechecked
def save_config(
    config: dict[str, Any],
    output_dir: Path,
    filename: str = 'config.yaml',
) -> Path:
    """Save configuration to YAML file.

    Only executes on rank 0 to avoid conflicts in distributed training.
    Creates parent directories if they don't exist.

    Args:
        config: Configuration dictionary to save.
        output_dir: Directory where config file will be saved.
        filename: Name of the config file (default: 'config.yaml').

    Returns:
        Path to the saved config file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / filename
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    logger.info(f"Saved configuration to: {config_path}")
    return config_path


# =============================================================================
# Config Pretty Printing
# =============================================================================

@rank_zero_only
@typechecked
def pretty_print_config(config: dict[str, Any]) -> None:
    """Pretty print configuration dictionary.

    Prints each section of the configuration in a readable format.
    Only executes on rank 0 in distributed training.

    Args:
        config: Configuration dictionary to print.
    """
    print("\n" + "=" * 60)
    print("Configuration:")
    print("=" * 60)

    for section, params in config.items():
        if isinstance(params, dict):
            print(f"\n  {section} parameters:")
            for key, value in params.items():
                print(f"    {key}: {value}")
        else:
            print(f"\n  {section}: {params}")

    print("\n" + "=" * 60 + "\n")


# =============================================================================
# Callback Configuration Helper
# =============================================================================

@typechecked
def get_callbacks_from_config(
    training_config: dict[str, Any],
    monitor: str = 'val_loss',
) -> list[pl.Callback]:
    """Create callbacks based on training configuration.

    Convenience function that extracts callback parameters from a training
    config dictionary and creates the appropriate callbacks.

    Args:
        training_config: Training configuration dictionary with optional keys:
            - checkpointing: bool (default: True)
            - lr_monitor: bool (default: True)
            - ckpt_every_n_epochs: int | None (default: None)
            - early_stopping: bool (default: False)
            - early_stopping_patience: int (default: 10)
        monitor: Metric to monitor for checkpointing and early stopping.

    Returns:
        List of configured Lightning callbacks.
    """
    return get_callbacks(
        checkpointing=training_config.get('checkpointing', True),
        lr_monitor=training_config.get('lr_monitor', True),
        ckpt_every_n_epochs=training_config.get('ckpt_every_n_epochs', None),
        early_stopping=training_config.get('early_stopping', False),
        early_stopping_patience=training_config.get('early_stopping_patience', 10),
        monitor=monitor,
    )

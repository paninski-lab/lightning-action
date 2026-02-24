"""Training functionality for lightning-action models (CSV pipeline).

This module provides training functions for action segmentation models
using tabular data (CSV/markers). Uses PyTorch Lightning for training
orchestration.

Key functions:
- train: Main entry point for training a segmentation model
- build_data_config_from_path: Build data configuration from directory structure
- compute_class_weights: Compute class weights from training data

Example usage:
    config = load_config('config.yaml')
    model = Segmenter(config)
    trained_model = train(config, model, output_dir='runs/experiment1')
"""

import logging
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.utilities import rank_zero_only
from typeguard import typechecked

from lightning_action import __version__
from lightning_action.data import DataModule
from lightning_action.data import transforms as transform_module
from lightning_action.data.utils import collect_labels_from_datamodule
from lightning_action.data.utils import compute_class_weights as _compute_class_weights

# Import shared utilities
from lightning_action.train_utils import (
    get_callbacks_from_config,
    pretty_print_config,
    reset_seeds,
    save_config,
    update_config_with_class_weights,
    update_config_with_label_names,
    validate_config,
)

# Re-export for backward compatibility
__all__ = [
    'train',
]

logger = logging.getLogger(__name__)


@typechecked
def train(
    config: dict[str, Any],
    model: pl.LightningModule,
    output_dir: str | Path,
) -> pl.LightningModule:
    """Train a Lightning model.

    Args:
        config: Configuration dictionary with data, model, and training settings.
        model: Lightning model to train.
        output_dir: Directory to save outputs and checkpoints.

    Returns:
        Trained model.

    Raises:
        ValueError: If required configuration keys are missing.
        FileNotFoundError: If data directory doesn't exist.
    """
    output_dir = Path(output_dir)

    # Print basic info
    if rank_zero_only.rank == 0:
        print(f'Output directory: {output_dir}')
        print(f'Model type: {type(model)}')

    # Reset seeds for reproducibility
    seed = config.get('training', {}).get('seed', 0)
    reset_seeds(seed=seed)

    # Pretty print configuration
    pretty_print_config(config)

    # ----------------------------------------------------------------------------------
    # Set up data objects
    # ----------------------------------------------------------------------------------

    logger.info("Setting up data module...")

    # Validate required config sections
    validate_config(config, required_sections=['data', 'training'])

    data_config = config['data']
    training_config = config['training']

    # Build data configuration from path if using simplified format
    if 'data_path' in data_config:
        logger.info("Building data config from data_path...")

        # Build full data config from path
        full_data_config = build_data_config_from_path(
            data_path=data_config['data_path'],
            expt_ids=data_config.get('expt_ids'),
            signal_types=[data_config.get('input_dir', 'markers'), 'labels'],
            transforms=data_config.get('transforms', None),  # for input stream only
        )

        # Use the full config for DataModule
        datamodule_config = full_data_config
    else:
        # Use existing full format
        datamodule_config = data_config

    # Create datamodule
    datamodule = DataModule(
        data_config=datamodule_config,
        sequence_length=training_config.get('sequence_length', 500),
        sequence_pad=training_config.get('sequence_pad', 0),
        batch_size=training_config.get('batch_size', 32),
        num_workers=training_config.get('num_workers', 4),
        train_probability=training_config.get('train_probability', 0.9),
        val_probability=training_config.get('val_probability', 0.1),
        seed=seed,
    )

    # Setup datamodule to access datasets
    datamodule.setup('fit')

    # Compute class weights if enabled
    weight_classes = data_config.get('weight_classes', True)
    if weight_classes:
        logger.info("Computing class weights...")
        class_weights = compute_class_weights(
            datamodule,
            ignore_index=data_config.get('ignore_index', -100),
        )
    else:
        logger.info("Class weighting disabled")
        class_weights = None

    # Update config and model with class weights (shared utility)
    update_config_with_class_weights(config, model, class_weights)

    # Save feature/label names to config
    feature_names = datamodule.dataset.feature_names
    if len(feature_names) > 0:
        config['data']['feature_names'] = feature_names
        model.config['data']['feature_names'] = feature_names

    label_names = datamodule.dataset.label_names
    update_config_with_label_names(config, model, label_names)

    # Update training steps information for schedulers
    num_epochs = training_config.get('num_epochs', 100)
    batch_size = training_config.get('batch_size', 32)

    steps_per_epoch = int(np.ceil(len(datamodule.dataset_train) / batch_size))
    total_steps = steps_per_epoch * num_epochs

    # Update model config with step information
    if hasattr(model, 'config'):
        if 'optimizer' not in model.config:
            model.config['optimizer'] = {}
        model.config['optimizer']['steps_per_epoch'] = steps_per_epoch
        model.config['optimizer']['total_steps'] = total_steps

    logger.info(f"Training steps: {steps_per_epoch} per epoch, {total_steps} total")

    # ----------------------------------------------------------------------------------
    # Save configuration in output directory
    # ----------------------------------------------------------------------------------

    logger.info(f"Creating output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Log package version
    config['version'] = __version__

    # Save config file (shared utility)
    save_config(config, output_dir)

    # ----------------------------------------------------------------------------------
    # Set up and run training
    # ----------------------------------------------------------------------------------

    logger.info("Setting up trainer...")

    # Logger
    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir=output_dir / 'tb_logs',
        name='',
        version='',
    )

    # Callbacks (using shared utility)
    callbacks = get_callbacks_from_config(training_config)

    # Trainer configuration
    trainer_config = {
        'max_epochs': num_epochs,
        'min_epochs': training_config.get('min_epochs', 1),
        'callbacks': callbacks,
        'logger': tb_logger,
        'default_root_dir': output_dir,
        'enable_checkpointing': training_config.get('checkpointing', True),
        'enable_progress_bar': training_config.get('progress_bar', True),
        'enable_model_summary': training_config.get('model_summary', True),
    }

    # Add GPU support if available and requested
    if torch.cuda.is_available() and training_config.get('device', 'cpu') == 'gpu':
        trainer_config['accelerator'] = 'gpu'
        trainer_config['devices'] = 1  # single GPU only
        logger.info("Using GPU for training")
    else:
        trainer_config['accelerator'] = 'cpu'
        logger.info("Using CPU for training")

    # Create trainer
    trainer = pl.Trainer(**trainer_config)

    # Log model summary
    logger.info(f"Model summary:\n{model}")

    # Train model
    logger.info("Starting training...")
    trainer.fit(model=model, datamodule=datamodule)

    # Training completed
    logger.info("Training completed!")

    # Save final model state
    final_model_path = output_dir / 'final_model.ckpt'
    trainer.save_checkpoint(final_model_path)
    logger.info(f"Saved final model to: {final_model_path}")

    return model


@typechecked
def compute_class_weights(
    datamodule: DataModule,
    ignore_index: int = -100,
) -> list[float]:
    """Compute class weights from training data.

    Computes weights inversely proportional to class frequency, with the most
    frequent class having weight 1.0. Based on daart's class weight computation.

    Args:
        datamodule: Lightning DataModule with class counting capability.
        ignore_index: Class index to ignore (typically background class).

    Returns:
        List of class weights.
    """
    logger.info("Computing class weights from training data...")

    # Ensure datamodule is set up
    if not hasattr(datamodule, 'dataset_train') or datamodule.dataset_train is None:
        datamodule.setup('fit')

    # Collect labels from training dataset
    all_labels, num_classes = collect_labels_from_datamodule(
        datamodule.dataset_train,
        ignore_index=ignore_index,
    )

    if len(all_labels) == 0:
        logger.warning("No labels found in training dataset, using uniform weights")
        # Try to get num_classes from dataset
        if hasattr(datamodule.dataset_train, 'label_names'):
            num_classes = len(datamodule.dataset_train.label_names)
        else:
            num_classes = 4  # default fallback
        return [1.0] * num_classes

    # Compute weights (no sqrt dampening for CSV pipeline to match original behavior)
    return _compute_class_weights(
        labels=all_labels,
        num_classes=num_classes,
        ignore_index=ignore_index,
        sqrt_dampening=False,
    )


@typechecked
def build_data_config_from_path(
    data_path: str | Path,
    expt_ids: list[str] | None = None,
    signal_types: list[str] | None = None,
    transforms: list[str] | None = None,
) -> dict[str, Any]:
    """Build data configuration from directory structure.

    Creates a data configuration dictionary by scanning a data directory
    for signal subdirectories and experiment files.

    Expected directory structure:
        data_path/
            markers/
                exp1.csv
                exp2.csv
            labels/
                exp1.csv
                exp2.csv

    Args:
        data_path: Root directory containing signal subdirectories.
        expt_ids: List of experiment IDs to include. If None, includes all.
        signal_types: List of signal types to include. If None, auto-detects.
        transforms: List of transform class names for input signals.
            If None, applies ZScore by default to marker/feature signals.

    Returns:
        Data configuration dictionary with keys: ids, signals, transforms, paths.

    Raises:
        FileNotFoundError: If data_path doesn't exist or experiments not found.
        ValueError: If no signal directories found or invalid transform name.
        NotADirectoryError: If expected signal directory is missing.
    """
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Data path does not exist: {data_path}")

    # Find signal directories
    if signal_types is None:
        # Auto-detect signal directories
        signal_dirs = [d for d in data_path.iterdir() if d.is_dir()]
        if not signal_dirs:
            raise ValueError(f"No signal directories found in: {data_path}")
        signal_types = [d.name for d in signal_dirs]
    else:
        # Validate specified signal types exist
        for sig_type in signal_types:
            sig_dir = data_path / sig_type
            if not sig_dir.is_dir():
                raise NotADirectoryError(
                    f"Signal directory not found: {sig_dir}"
                )

    # Find experiments from first signal directory
    first_sig_dir = data_path / signal_types[0]
    all_expt_ids = [f.stem for f in first_sig_dir.glob('*.csv')]

    if not all_expt_ids:
        raise ValueError(f"No CSV files found in: {first_sig_dir}")

    # Filter experiments if specified
    if expt_ids is not None:
        for expt_id in expt_ids:
            if expt_id not in all_expt_ids:
                raise FileNotFoundError(f"Did not find expt_id={expt_id}")
        selected_expt_ids = expt_ids
    else:
        selected_expt_ids = sorted(all_expt_ids)

    # Build transforms list
    if transforms is None:
        # Default: ZScore for markers/features, None for labels
        transform_classes = [transform_module.ZScore()]
    else:
        # Create transform instances from class names
        transform_classes = []
        for t_name in transforms:
            if not hasattr(transform_module, t_name):
                raise ValueError(f"Unknown transform class: {t_name}")
            transform_classes.append(getattr(transform_module, t_name)())

    # Build config
    config = {
        'ids': [],
        'signals': [],
        'transforms': [],
        'paths': [],
    }

    for expt_id in selected_expt_ids:
        config['ids'].append(expt_id)

        expt_signals = []
        expt_transforms = []
        expt_paths = []

        for sig_type in signal_types:
            expt_signals.append(sig_type)

            # Apply transforms to input signals (markers/features), not labels
            if sig_type.startswith(('markers', 'features')):
                expt_transforms.append(transform_classes)
            else:
                expt_transforms.append(None)

            sig_path = data_path / sig_type / f'{expt_id}.csv'
            expt_paths.append(str(sig_path))

        config['signals'].append(expt_signals)
        config['transforms'].append(expt_transforms)
        config['paths'].append(expt_paths)

    return config

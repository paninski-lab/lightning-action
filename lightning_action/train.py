"""Training functionality for lightning-action models.

This module provides training functions adapted from beast and daart patterns,
using PyTorch Lightning for action segmentation models.
"""

import logging
import os
import random
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import numpy as np
import torch
import yaml
from lightning.pytorch.utilities import rank_zero_only
from typeguard import typechecked

from lightning_action import __version__
from lightning_action.data import DataModule
from lightning_action.data import transforms as transform_module

logger = logging.getLogger(__name__)


@typechecked
def train(
    config: dict[str, Any],
    model: pl.LightningModule,
    output_dir: str | Path,
) -> pl.LightningModule:
    """Train a Lightning model.

    Args:
        config: configuration dictionary with data, model, and training settings
        model: Lightning model to train
        output_dir: directory to save outputs and checkpoints

    Returns:
        trained model

    Raises:
        ValueError: if required configuration keys are missing
        FileNotFoundError: if data directory doesn't exist
    """
    output_dir = Path(output_dir)

    # print basic info
    if rank_zero_only.rank == 0:
        print(f'Output directory: {output_dir}')
        print(f'Model type: {type(model)}')

    # reset seeds for reproducibility
    seed = config.get('training', {}).get('seed', 0)
    reset_seeds(seed=seed)

    # pretty print configuration
    pretty_print_config(config)

    # ----------------------------------------------------------------------------------
    # Set up data objects
    # ----------------------------------------------------------------------------------

    logger.info("Setting up data module...")

    # validate required config sections
    if 'data' not in config:
        raise ValueError("Configuration must contain 'data' section")
    if 'training' not in config:
        raise ValueError("Configuration must contain 'training' section")

    data_config = config['data']
    training_config = config['training']

    # build data configuration from path if using simplified format
    if 'data_path' in data_config:
        logger.info("Building data config from data_path...")
        
        # build full data config from path
        full_data_config = build_data_config_from_path(
            data_path=data_config['data_path'],
            expt_ids=data_config.get('expt_ids'),
            signal_types=[data_config.get('input_dir', 'markers'), 'labels'],
            transforms=data_config.get('transforms', None),  # for input stream only
        )
        
        # use the full config for DataModule
        datamodule_config = full_data_config
    else:
        # use existing full format
        datamodule_config = data_config

    # create datamodule
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

    # setup datamodule to access datasets
    datamodule.setup('fit')

    # compute class weights if enabled
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

    # save feature/label names to config
    feature_names = datamodule.dataset.feature_names
    if len(feature_names) > 0:
        config['data']['feature_names'] = feature_names
        model.config['data']['feature_names'] = feature_names
    label_names = datamodule.dataset.label_names
    if len(label_names) > 0:
        config['data']['label_names'] = label_names
        model.config['data']['label_names'] = label_names

    # update training steps information for schedulers
    num_epochs = training_config.get('num_epochs', 100)
    batch_size = training_config.get('batch_size', 32)

    steps_per_epoch = int(np.ceil(len(datamodule.dataset_train) / batch_size))
    total_steps = steps_per_epoch * num_epochs

    # update model config with step information
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

    # log package version
    config['version'] = __version__

    # save config file
    dest_config_file = output_dir / 'config.yaml'
    with open(dest_config_file, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)
    logger.info(f"Saved configuration to: {dest_config_file}")

    # ----------------------------------------------------------------------------------
    # Set up and run training
    # ----------------------------------------------------------------------------------

    logger.info("Setting up trainer...")

    # logger
    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir=output_dir / 'tb_logs',
        name='',
        version='',
    )

    # callbacks
    callbacks = get_callbacks(
        checkpointing=training_config.get('checkpointing', True),
        lr_monitor=training_config.get('lr_monitor', True),
        ckpt_every_n_epochs=training_config.get('ckpt_every_n_epochs', None),
        early_stopping=training_config.get('early_stopping', False),
        early_stopping_patience=training_config.get('early_stopping_patience', 10),
    )

    # trainer configuration
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

    # add GPU support if available and requested
    if torch.cuda.is_available() and training_config.get('device', 'cpu') == 'gpu':
        trainer_config['accelerator'] = 'gpu'
        trainer_config['devices'] = 1  # single GPU only
        logger.info("Using GPU for training")
    else:
        trainer_config['accelerator'] = 'cpu'
        logger.info("Using CPU for training")

    # create trainer
    trainer = pl.Trainer(**trainer_config)

    # log model summary
    logger.info(f"Model summary:\n{model}")

    # train model
    logger.info("Starting training...")
    trainer.fit(model=model, datamodule=datamodule)

    # training completed
    logger.info("Training completed!")

    # save final model state
    final_model_path = output_dir / 'final_model.ckpt'
    trainer.save_checkpoint(final_model_path)
    logger.info(f"Saved final model to: {final_model_path}")

    return model


@typechecked
def reset_seeds(seed: int = 0) -> None:
    """Reset all random seeds for reproducibility.
    
    Args:
        seed: random seed value
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@rank_zero_only
@typechecked
def pretty_print_config(config: dict[str, Any]) -> None:
    """Pretty print configuration dictionary.
    
    Args:
        config: configuration dictionary to print
    """
    print('Configuration:')
    for key, val in config.items():
        print('--------------------')
        print(f'{key} parameters')
        print('--------------------')
        if isinstance(val, dict):
            for k, v in val.items():
                print(f'{k}: {v}')
        else:
            print(f'{val}')
        print()
    print('\n\n')


@typechecked
def build_data_config_from_path(
    data_path: str | Path,
    expt_ids: list[str] | None = None,
    signal_types: list[str] | None = None,
    transforms: list[str] | None = None,
) -> dict[str, Any]:
    """Build DataModule configuration from data directory path.
    
    Makes assumptions about data directory structure:
    - Signal types are top-level directories under data_path
      (e.g., 'markers', 'labels', 'features_0')
    - Each signal directory contains CSV files named after experiment IDs
    - Default signal types are ['markers', 'labels']
    - Applies configurable transforms to input signals, defaults to Z-score for markers/features
    
    Expected structure:
    data_path/
      markers/
        experiment1.csv
        experiment2.csv
      labels/
        experiment1.csv
        experiment2.csv
      features_0/  # optional additional signal types
        experiment1.csv
        experiment2.csv
    
    Args:
        data_path: path to data directory containing signal type subdirectories
        expt_ids: list of experiment IDs to include (None for all)
        signal_types: list of signal types to load (None for auto-detect)
        transforms: list of transform class names to apply (None for default ZScore)
        
    Returns:
        DataModule configuration dictionary
        
    Raises:
        FileNotFoundError: if data_path doesn't exist
        ValueError: if no valid experiments found
    """
    data_path = Path(data_path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data path does not exist: {data_path}")
    
    # discover available signal types if not specified
    if signal_types is None:
        # find all directories that could be signal types
        signal_dirs = [d for d in data_path.iterdir() if d.is_dir()]
        signal_types = [d.name for d in signal_dirs]
        logger.info(f"Auto-detected signal types: {signal_types}")
    else:
        logger.info(f"Using specified signal types: {signal_types}")
    
    if not signal_types:
        raise ValueError(f"No signal directories found in {data_path}")
    
    # set up default transforms if not specified
    if transforms is None:
        transforms = ['ZScore']
        logger.info(f"Using default transforms: {transforms}")
    else:
        logger.info(f"Using specified transforms: {transforms}")
    
    # create transform instances from class names
    def create_transform_instance(transform_name: str):
        """Create transform instance from class name."""
        if hasattr(transform_module, transform_name):
            transform_class = getattr(transform_module, transform_name)
            return transform_class()
        else:
            raise ValueError(f"Unknown transform class: {transform_name}")
    
    # discover experiment IDs if not specified
    if expt_ids is None:
        # find experiment IDs by looking at CSV files in the first signal directory
        first_signal_dir = data_path / signal_types[0]
        if first_signal_dir.exists():
            csv_files = list(first_signal_dir.glob("*.csv"))
            expt_ids = [f.stem for f in csv_files]  # use filename without .csv extension
            logger.info(f"Auto-detected {len(expt_ids)} experiments: {expt_ids}")
        else:
            raise ValueError(f"Signal directory {first_signal_dir} not found")
    else:
        logger.info(f"Using specified experiments: {expt_ids}")
    
    if not expt_ids:
        raise ValueError(f"No experiment CSV files found in {data_path}")
    
    # build configuration for each experiment
    ids_all = []
    signals_all = []
    transforms_all = []
    paths_all = []
    
    for expt_id in expt_ids:
        # build paths for each signal type for this experiment
        expt_signals = []
        expt_transforms = []
        expt_paths = []

        for signal_type in signal_types:

            signal_dir = data_path / signal_type
            signal_file = next(signal_dir.glob(f"{expt_id}*.csv"))
            if not signal_file.exists():
                raise FileNotFoundError(f"Signal file not found: {signal_file}")
            expt_paths.append(signal_file)

            expt_signals.append(signal_type)

            # set up transforms: configurable transforms for markers/features, None for labels
            if not signal_type.startswith('labels'):
                signal_transforms = []
                for transform in transforms:
                    signal_transforms.append(create_transform_instance(transform))
                expt_transforms.append(signal_transforms)
            else:
                expt_transforms.append(None)

        # add experiment
        ids_all.append(expt_id)
        signals_all.append(expt_signals)
        transforms_all.append(expt_transforms)
        paths_all.append(expt_paths)
    
    if not ids_all:
        raise ValueError(f"No valid experiments found in {data_path}")
    
    logger.info(
        f"Built data config for {len(ids_all)} experiments with {len(signal_types)} signal types"
    )
    
    return {
        'ids': ids_all,
        'signals': signals_all,
        'transforms': transforms_all,
        'paths': paths_all,
    }


@typechecked
def compute_class_weights(datamodule: DataModule, ignore_index: int = -100) -> list[float]:
    """Compute class weights for imbalanced dataset.
    
    Computes weights inversely proportional to class frequency, with the most
    frequent class having weight 1.0. Based on daart's class weight computation.
    
    Args:
        datamodule: Lightning DataModule with class counting capability
        ignore_index: class index to ignore (typically background class)
        
    Returns:
        list of class weights
    """
    logger.info("Computing class weights from training data...")
    
    # ensure datamodule is set up
    if not hasattr(datamodule, 'dataset_train') or datamodule.dataset_train is None:
        datamodule.setup('fit')
    
    # count examples per class in training set
    dataset_train = datamodule.dataset_train
    
    # get all labels from training dataset
    all_labels = []
    num_classes = None
    
    for i in range(len(dataset_train)):
        batch = dataset_train[i]
        if 'labels' in batch:
            labels = batch['labels']
            if isinstance(labels, torch.Tensor):
                labels = labels.numpy()
            # handle both one-hot and class index formats
            if labels.ndim > 1 and labels.shape[-1] > 1:
                # one-hot encoded - convert to class indices
                labels = np.argmax(labels, axis=-1)
                if num_classes is None:
                    num_classes = batch['labels'].shape[-1]
            all_labels.append(labels.flatten())
    
    if not all_labels:
        logger.warning("No labels found in training dataset, using uniform weights")
        # try to get num_classes from dataset
        if hasattr(dataset_train, 'label_names'):
            num_classes = len(dataset_train.label_names)
        else:
            num_classes = 4  # default fallback
        return [1.0] * num_classes
    
    # concatenate all labels
    all_labels = np.concatenate(all_labels)
    
    # count occurrences of each class
    unique_classes, counts = np.unique(all_labels, return_counts=True)
    
    # determine total number of classes
    if num_classes is None:
        max_class = int(max(unique_classes))
        num_classes = max_class + 1
    
    # create totals array with counts for each class
    totals = np.zeros(num_classes)
    for cls, count in zip(unique_classes, counts):
        if int(cls) < num_classes:  # ensure class index is valid
            totals[int(cls)] = count
    
    # ignore background class if specified
    if 0 <= ignore_index < len(totals):
        totals[ignore_index] = 0
    
    # compute class weights: most frequent class gets weight 1.0,
    # others get weights inversely proportional to frequency
    max_count = np.max(totals)
    if max_count == 0:
        logger.warning("No labeled examples found, using uniform weights")
        class_weights = np.ones(num_classes)
    else:
        class_weights = max_count / (totals + 1e-10)
        class_weights[totals == 0] = 0.0
    
    logger.info(f"Class counts: {totals}")
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
        checkpointing: whether to enable model checkpointing
        lr_monitor: whether to monitor learning rate
        ckpt_every_n_epochs: save checkpoint every N epochs (None to disable)
        early_stopping: whether to enable early stopping
        early_stopping_patience: patience for early stopping
        
    Returns:
        list of Lightning callbacks
    """
    callbacks = []
    
    # learning rate monitoring
    if lr_monitor:
        lr_monitor_cb = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
        callbacks.append(lr_monitor_cb)
    
    # model checkpointing - save best model
    if checkpointing:
        ckpt_best_callback = pl.callbacks.ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            filename='{epoch}-{step}-best',
            save_top_k=1,
        )
        callbacks.append(ckpt_best_callback)
    
    # periodic checkpointing
    if ckpt_every_n_epochs is not None:
        ckpt_callback = pl.callbacks.ModelCheckpoint(
            monitor=None,
            every_n_epochs=ckpt_every_n_epochs,
            save_top_k=-1,
            filename='{epoch}-{step}',
        )
        callbacks.append(ckpt_callback)
    
    # early stopping
    if early_stopping:
        early_stop_callback = pl.callbacks.EarlyStopping(
            monitor='val_loss',
            mode='min',
            patience=early_stopping_patience,
            verbose=True,
        )
        callbacks.append(early_stop_callback)
    
    return callbacks

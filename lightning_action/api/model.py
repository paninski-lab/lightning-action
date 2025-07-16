"""High-level API for lightning-action models.

This module provides a high-level interface for training, loading, and using
lightning-action models for action segmentation.
"""

import contextlib
import os
from pathlib import Path
from typing import Any

import lightning as pl
import numpy as np
import torch
import yaml
from typeguard import typechecked

from lightning_action.data import DataModule
from lightning_action.models.segmenter import Segmenter
from lightning_action.train import train


@contextlib.contextmanager
def chdir(path: Path):
    """Context manager for changing directories.
    
    Args:
        path: directory to change to
    """
    old_cwd = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(old_cwd)


@typechecked
class Model:
    """High-level API wrapper for lightning-action models.

    This class manages both the Lightning model and the training/inference processes,
    providing a convenient interface for action segmentation tasks.
    """

    def __init__(
        self,
        model: Segmenter,
        config: dict[str, Any],
        model_dir: str | Path | None = None,
    ) -> None:
        """Initialize with Lightning model and config.
        
        Args:
            model: Lightning segmentation model
            config: configuration dictionary
            model_dir: directory containing model files (optional)
        """
        self.model = model
        self.config = config
        self.model_dir = Path(model_dir) if model_dir is not None else None

    @classmethod
    def from_dir(cls, model_dir: str | Path):
        """Load a Lightning model from a directory.

        Args:
            model_dir: path to directory containing model checkpoint and config

        Returns:
            initialized model wrapper
        
        Raises:
            FileNotFoundError: if config or checkpoint files are not found
        """
        model_dir = Path(model_dir)
        
        # load config
        config_path = model_dir / 'config.yaml'
        if not config_path.exists():
            # fallback to hparams.yaml for compatibility
            config_path = model_dir / 'hparams.yaml'
        
        if not config_path.exists():
            raise FileNotFoundError(f'Config file not found in {model_dir}')
            
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # create model
        model = Segmenter(config)

        # load Lightning checkpoint
        checkpoint_patterns = ['*best*.ckpt', '*.ckpt', '*best*.pt', '*.pt']
        checkpoint_path = None
        
        for pattern in checkpoint_patterns:
            checkpoints = list(model_dir.rglob(pattern))
            if checkpoints:
                checkpoint_path = checkpoints[0]
                break
                
        if checkpoint_path is None:
            raise FileNotFoundError(f'No checkpoint files found in {model_dir}')
            
        # load checkpoint
        if checkpoint_path.suffix == '.ckpt':
            # Lightning checkpoint
            model = Segmenter.load_from_checkpoint(checkpoint_path, config=config)
        else:
            # PyTorch state dict
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(state_dict)
            
        model.eval()
        print(f'Loaded model weights from {checkpoint_path}')

        return cls(model, config, model_dir)

    @classmethod
    def from_config(cls, config_path: str | Path | dict):
        """Create a new Lightning model from a config file.

        Args:
            config_path: path to config file or config dictionary

        Returns:
            initialized model wrapper with untrained model
        """
        if not isinstance(config_path, dict):
            config_path = Path(config_path)
            if not config_path.exists():
                raise FileNotFoundError(f'Config file not found: {config_path}')
            with open(config_path) as f:
                config = yaml.safe_load(f)
        else:
            config = config_path

        # update config options if necessary
        if config['data'].get('transforms'):
            if 'VelocityConcat' in config['data']['transforms']:
                config['model']['input_size'] *= 2

        model = Segmenter(config)

        return cls(model, config, model_dir=None)

    def train(self, output_dir: str | Path = 'runs/default'):
        """Train the model using PyTorch Lightning.

        Args:
            output_dir: directory to save checkpoints and logs
        """
        self.model_dir = Path(output_dir)
        self.model_dir.mkdir(exist_ok=True, parents=True)
        with chdir(self.model_dir):
            self.model = train(self.config, self.model, output_dir=self.model_dir)
    
    def predict(
        self,
        data_path: str | Path,
        input_dir: str,
        output_dir: str | Path,
        expt_ids: list[str] | None = None,
    ):
        """Generate predictions for data using the trained model.

        Creates separate prediction files for each experiment in the output directory.

        Args:
            data_path: path to data directory with signal directories
            input_dir: 'markers' | 'features' | etc.
            output_dir: directory to save prediction files (one per experiment)
            expt_ids: list of experiment IDs to predict on (None for all)
            
        Raises:
            ValueError: if model is not trained
        """
        if self.model is None:
            raise ValueError('Model must be trained or loaded before prediction')
            
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # build data configuration to get available experiments
        from lightning_action.train import build_data_config_from_path
        
        data_config = build_data_config_from_path(
            data_path=data_path,
            expt_ids=expt_ids,
            signal_types=[input_dir],
        )
        
        # get training config
        training_config = self.config.get('training', {})
        
        # loop over each experiment and create separate predictions
        experiment_ids = data_config['ids']
        
        for expt_id in experiment_ids:
            print(f'Generating predictions for experiment: {expt_id}')
            
            # create data config for single experiment
            single_expt_config = build_data_config_from_path(
                data_path=data_path,
                expt_ids=[expt_id],
                signal_types=[input_dir],
            )
            
            # create datamodule for this experiment
            datamodule = DataModule(
                data_config=single_expt_config,
                sequence_length=training_config.get('sequence_length', 500),
                sequence_pad=training_config.get('sequence_pad', 0),
                batch_size=training_config.get('batch_size', 32),
                num_workers=training_config.get('num_workers', 4),
                train_probability=1.0,  # use all data for prediction
                val_probability=0.0,
                seed=training_config.get('seed', 42),
            )
            
            # setup for prediction
            datamodule.setup('predict')
            
            # create trainer for prediction
            device = training_config.get('device', 'cpu')
            trainer_config = {
                'accelerator': 'gpu' if device == 'gpu' and torch.cuda.is_available() else 'cpu',
                'devices': 1,
                'logger': False,
                'enable_checkpointing': False,
                'enable_progress_bar': False,  # disable for cleaner output when looping
            }
            
            trainer = pl.Trainer(**trainer_config)
            
            # generate predictions for this experiment
            predictions = trainer.predict(self.model, datamodule=datamodule)
            
            # concatenate predictions from all batches
            all_probs = []
            for batch_preds in predictions:
                probs = batch_preds['probabilities'][0]  # remove batch dim
                all_probs.append(probs.cpu().numpy())
            
            # stack and save predictions for this experiment
            final_probs = np.vstack(all_probs)
            output_file = output_dir / f'{expt_id}_predictions.npy'
            np.save(output_file, final_probs)
            print(f'Saved predictions to {output_file}')
        
        print(f'Completed predictions for {len(experiment_ids)} experiments in {output_dir}')

"""High-level API for lightning-action video models.

This module provides a high-level interface for training, loading, and using
lightning-action video models for action segmentation.
"""

import contextlib
import os
from pathlib import Path
from typing import Any

import lightning as pl
import numpy as np
import pandas as pd
import torch
import yaml
from typeguard import typechecked

from lightning_action.data.video_datamodule import VideoDataModule
from lightning_action.models.video_segmenter import VideoSegmenter
from lightning_action.video_train import train_video
import logging


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
class VideoModel:
    """High-level API wrapper for lightning-action video models.

    This class manages both the Lightning model and the training/inference processes,
    providing a convenient interface for video action segmentation tasks.
    """

    def __init__(
        self,
        model: VideoSegmenter,
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
        model = VideoSegmenter(config)

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
            model = VideoSegmenter.load_from_checkpoint(checkpoint_path, config=config)
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

        model = VideoSegmenter(config)

        return cls(model, config, model_dir=None)

    def train(self, output_dir: str | Path = 'runs/default', post_inference: bool = True):
        """Train the model using PyTorch Lightning.

        After training is complete, automatically runs inference on all experiment IDs
        used for training and saves predictions to output_dir/predictions/.

        Args:
            output_dir: directory to save checkpoints and logs
            post_inference: run inference on all training expts and store in model_dir/predictions
        """
        self.model_dir = Path(output_dir)
        self.model_dir.mkdir(exist_ok=True, parents=True)
        with chdir(self.model_dir):
            self.model = train_video(self.config, self.model, output_dir=self.model_dir)

        # automatically run inference on training experiments
        if post_inference:
            self._run_post_training_inference()
    
    def _run_post_training_inference(self):
        """Run inference on all training experiment IDs after training completes.
        
        This method extracts the experiment IDs from the training configuration,
        determines the appropriate data path and input directory, and runs inference
        on all experiments used for training.
        """
        if self.model is None:
            print('Warning: No trained model found, skipping post-training inference')
            return
            
        if self.model_dir is None:
            print('Warning: No model directory found, skipping post-training inference')
            return
            
        # extract data configuration to get experiment IDs
        videos_dir = self.config['data']['videos_dir']
        labels_dir = self.config['data']['labels_dir']
        expt_ids = self.config['data']['expt_ids']
        
        # create predictions directory
        predictions_dir = self.model_dir / 'predictions'
        predictions_dir.mkdir(exist_ok=True)
        
        print(f'Running post-training inference on all training experiments...')
        print(f'Videos directory: {videos_dir}')
        print(f'Labels directory: {labels_dir}')
        print(f'Experiment IDs: {expt_ids if expt_ids else "all"}')
        print(f'Predictions will be saved to: {predictions_dir}')
        
        try:
            # run inference using the existing predict method
            self.predict(
                videos_dir=videos_dir,
                labels_dir=labels_dir,
                output_dir=predictions_dir,
                expt_ids=expt_ids,
            )
            print('Post-training inference completed successfully!')
        except Exception as e:
            print(f'Warning: Post-training inference failed with error: {e}')
            print('Training completed successfully, but automatic inference was skipped.')
    
    def predict(
        self,
        videos_dir: str | Path,
        labels_dir: str | Path,
        output_dir: str | Path,
        output_file: str | Path | None = None,
        expt_ids: list[str] | None = None,
    ):
        """Generate predictions for data using the trained model.

        Creates separate prediction files for each experiment in the output directory.

        Args:
            videos_dir: directory containing MP4 video files
            labels_dir: directory containing NumPy (.npy) label files
            output_dir: directory to save prediction files (one per experiment)
            output_file: full path to save prediction file; overwrites output_dir if not None
            expt_ids: list of experiment IDs to predict on (None for all)
            
        Raises:
            ValueError: if model is not trained
        """
        videos_dir = Path(videos_dir)
        labels_dir = Path(labels_dir)

        if self.model is None:
            raise ValueError('Model must be trained or loaded before prediction')

        if output_file is not None and (expt_ids is not None and len(expt_ids) > 1):
            raise RuntimeError('Can only supply `output_file` when specifying a single expt_id')

        # get all expt_ids if not provided
        if expt_ids is None:
            expt_ids = [f[:-4] for f in os.listdir(videos_dir) if f.endswith('.mp4')]

        # get training config
        training_config = self.config.get('training', {})

        # loop over each experiment and create separate predictions
        for expt_id in expt_ids:
            print(f'Generating predictions for experiment: {expt_id}')

            # create data config for single experiment
            data_config = self.config['data'].copy()
            data_config['expt_ids'] = [expt_id]
            data_config['videos_dir'] = str(videos_dir)
            data_config['labels_dir'] = str(labels_dir)

            # create datamodule for this experiment
            datamodule = VideoDataModule(
                data_config=data_config,
                sequence_length=training_config.get('sequence_length', 128),
                batch_size=1,
                num_workers=training_config.get('num_workers', 0),
                train_probability=1.0,
                val_probability=0.0,
                seed=training_config.get('seed', 42),
            )
            
            # setup for prediction
            datamodule.setup('predict')
            
            # get total_frames for this video
            label_path = labels_dir / f'{expt_id}.npy'
            if not label_path.exists():
                raise FileNotFoundError(f'Missing label file: {label_path}')
            labels_array = np.load(str(label_path), mmap_mode='r')
            total_frames = labels_array.shape[0]
            
            # create trainer for prediction
            device = training_config.get('device', 'cpu')
            trainer_config = {
                'accelerator': 'gpu' if device == 'gpu' and torch.cuda.is_available() else 'cpu',
                'devices': 1,
                'logger': False,
                'enable_checkpointing': False,
                'enable_progress_bar': False,
            }
            
            trainer = pl.Trainer(**trainer_config)
            
            # generate predictions for this experiment
            predictions = trainer.predict(self.model, datamodule=datamodule)
            
            # concatenate predictions from all batches
            all_probs = []
            num_lags = self.config['model']['num_lags']
            for batch_preds in predictions:
                probs_full = batch_preds['probabilities'][0].cpu().numpy() 
                all_probs.append(probs_full)
            
            # stack predictions from all chunks
            final_probs = np.vstack(all_probs)
            
            # get original data length for this experiment and pad with NaNs if needed
            current_length = final_probs.shape[0]
            if current_length < total_frames:
                # pad with NaNs to match original input file length
                num_classes = final_probs.shape[1]
                padding_rows = total_frames - current_length
                nan_padding = np.full((padding_rows, num_classes), np.nan)
                final_probs = np.vstack([final_probs, nan_padding])
                print(f'Padded predictions from {current_length} to {total_frames} rows')
            
            # create dataframe and save predictions for this experiment
            df = pd.DataFrame(data=final_probs, columns=datamodule.get_label_names())
            if output_file is not None:
                output_file_ = Path(output_file)
            else:
                output_file_ = output_dir / f'{expt_id}_predictions.csv'
            output_file_.parent.mkdir(exist_ok=True, parents=True)
            df.to_csv(output_file_)
            print(f'Saved predictions to {output_file_}')

        print(f'Completed predictions for {len(expt_ids)} experiments in {output_dir}')

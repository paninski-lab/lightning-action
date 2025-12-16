"""High-level API for video action segmentation models.

This module provides a user-friendly VideoModel class that wraps the
underlying VideoSegmenter model and handles common workflows:

- Loading trained models from disk
- Creating new models from configuration
- Training models with automatic post-training inference
- Running inference on new videos

The VideoModel class abstracts away the complexity of Lightning modules,
data modules, and file I/O to provide a simple interface for end users.

Example usage:
    # Load a trained model
    model = VideoModel.from_dir('runs/my_experiment')
    
    # Run predictions on new videos
    model.predict(
        videos_dir='/path/to/videos',
        labels_dir='/path/to/labels',
        output_dir='/path/to/predictions',
    )
    
    # Or train a new model
    model = VideoModel.from_config('config.yaml')
    model.train(output_dir='runs/new_experiment')
"""

import contextlib
import os
from pathlib import Path
from typing import Any, Optional

import lightning as pl
import numpy as np
import pandas as pd
import torch
import yaml
from typeguard import typechecked

from lightning_action.data.video_datamodule import VideoDataModule
from lightning_action.models.video_segmenter import VideoSegmenter
from lightning_action.video_train import train_video


@contextlib.contextmanager
def chdir(path: Path):
    """Context manager for temporarily changing working directory.
    
    Useful when training requires relative paths from the output directory.
    
    Args:
        path: Directory to change to.
    
    Yields:
        None. Working directory is restored on exit.
    
    Example:
        with chdir(Path('/some/dir')):
            # Working directory is /some/dir here
            do_something()
        # Working directory is restored here
    """
    old_cwd = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(old_cwd)


@typechecked
class VideoModel:
    """High-level wrapper for video action segmentation models.
    
    This class provides a simplified interface for working with video
    segmentation models, handling the complexity of model loading,
    training, and inference behind a clean API.
    
    Attributes:
        model: The underlying VideoSegmenter Lightning module.
        config: Configuration dictionary used to create/load the model.
        model_dir: Directory where the model is stored (after training/loading).
    
    Example:
        # Training workflow
        model = VideoModel.from_config('config.yaml')
        model.train(output_dir='runs/exp1')
        
        # Inference workflow
        model = VideoModel.from_dir('runs/exp1')
        model.predict(
            videos_dir='data/videos',
            labels_dir='data/labels',
            output_dir='results',
        )
    """

    def __init__(
        self,
        model: VideoSegmenter,
        config: dict[str, Any],
        model_dir: str | Path | None = None,
    ) -> None:
        """Initialize the VideoModel wrapper.
        
        This constructor is typically not called directly. Use the
        class methods `from_dir()` or `from_config()` instead.
        
        Args:
            model: Initialized VideoSegmenter model.
            config: Configuration dictionary.
            model_dir: Optional path to model directory.
        """
        self.model = model
        self.config = config
        self.model_dir = Path(model_dir) if model_dir is not None else None

    @classmethod
    def from_dir(cls, model_dir: str | Path) -> 'VideoModel':
        """Load a trained model from a directory.
        
        Searches for configuration and checkpoint files in the directory
        and loads the model with trained weights.
        
        Expected directory structure:
            model_dir/
                config.yaml (or hparams.yaml)
                checkpoints/
                    best-*.ckpt (or any .ckpt/.pt file)
        
        Args:
            model_dir: Path to directory containing model files.
        
        Returns:
            VideoModel instance with loaded weights.
        
        Raises:
            FileNotFoundError: If config or checkpoint files are not found.
        """
        model_dir = Path(model_dir)
        
        # Try to find config file (prefer config.yaml over hparams.yaml)
        config_path = model_dir / 'config.yaml'
        if not config_path.exists():
            config_path = model_dir / 'hparams.yaml'
        
        if not config_path.exists():
            raise FileNotFoundError(f'Config file not found in {model_dir}')
            
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Create model architecture from config
        model = VideoSegmenter(config)

        # Find checkpoint file (prefer 'best' checkpoints)
        checkpoint_patterns = ['*best*.ckpt', '*.ckpt', '*best*.pt', '*.pt']
        checkpoint_path = None
        
        for pattern in checkpoint_patterns:
            checkpoints = list(model_dir.rglob(pattern))
            if checkpoints:
                checkpoint_path = checkpoints[0]
                break
                
        if checkpoint_path is None:
            raise FileNotFoundError(f'No checkpoint files found in {model_dir}')
        
        # Load weights based on file format
        if checkpoint_path.suffix == '.ckpt':
            # Lightning checkpoint format
            model = VideoSegmenter.load_from_checkpoint(
                checkpoint_path, config=config
            )
        else:
            # Plain PyTorch state dict
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(state_dict)
            
        model.eval()

        return cls(model, config, model_dir)

    @classmethod
    def from_config(cls, config_path: str | Path | dict) -> 'VideoModel':
        """Create a new untrained model from configuration.
        
        Args:
            config_path: Path to YAML config file, or dict with config.
        
        Returns:
            VideoModel instance with randomly initialized weights.
        
        Raises:
            FileNotFoundError: If config file path doesn't exist.
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

    def train(
        self, 
        output_dir: str | Path = 'runs/default', 
        post_inference: bool = True,
    ) -> None:
        """Train the model and optionally run inference on training data.
        
        Args:
            output_dir: Directory to save checkpoints, logs, and predictions.
            post_inference: If True, run inference on all training videos
                after training completes to generate predictions.
        """
        self.model_dir = Path(output_dir)
        self.model_dir.mkdir(exist_ok=True, parents=True)
        
        # Train the model
        with chdir(self.model_dir):
            self.model = train_video(
                self.config, self.model, output_dir=self.model_dir
            )

        # Optionally generate predictions on training data
        if post_inference:
            self._run_post_training_inference()
    
    def _run_post_training_inference(self) -> None:
        """Run inference on all training experiments after training.
        
        This generates prediction CSV files for each video in the training
        set, which is useful for evaluating model performance and debugging.
        """
        if self.model is None or self.model_dir is None:
            return
            
        videos_dir = self.config['data']['videos_dir']
        labels_dir = self.config['data']['labels_dir']
        expt_ids = self.config['data']['expt_ids']
        
        predictions_dir = self.model_dir / 'predictions'
        predictions_dir.mkdir(exist_ok=True)
        
        try:
            self.predict(
                videos_dir=videos_dir,
                labels_dir=labels_dir,
                output_dir=predictions_dir,
                expt_ids=expt_ids,
            )
        except Exception as e:
            print(f'Warning: Post-training inference failed: {e}')
    
    def predict(
        self,
        videos_dir: str | Path,
        labels_dir: str | Path,
        output_dir: str | Path,
        output_file: Optional[str | Path] = None,
        expt_ids: Optional[list[str]] = None,
    ) -> None:
        """Generate predictions for videos using single GPU.
        
        Processes videos sequentially on a single GPU for simplicity
        and reliability. For large-scale inference, videos can be split
        across multiple calls.
        
        Output format is CSV with columns:
        - frame: Frame index (0-based)
        - class_0, class_1, ...: Probability for each class
        
        Args:
            videos_dir: Directory containing .mp4 video files.
            labels_dir: Directory containing .npy label files.
                (Labels are needed for determining video lengths)
            output_dir: Directory to save prediction CSV files.
            output_file: If predicting a single video, specify output filename.
            expt_ids: List of experiment IDs to predict. If None, predicts
                all videos in videos_dir.
        
        Raises:
            ValueError: If model hasn't been trained/loaded.
            RuntimeError: If output_file specified with multiple videos.
            FileNotFoundError: If video or label files are missing.
        """
        videos_dir = Path(videos_dir)
        labels_dir = Path(labels_dir)
        output_dir = Path(output_dir)

        if self.model is None:
            raise ValueError('Model must be trained or loaded before prediction')

        # Validate output_file usage
        if output_file is not None and (expt_ids is not None and len(expt_ids) > 1):
            raise RuntimeError(
                'Can only supply `output_file` when specifying a single expt_id'
            )

        # Discover videos if not specified
        if expt_ids is None:
            expt_ids = [f[:-4] for f in os.listdir(videos_dir) if f.endswith('.mp4')]

        num_videos = len(expt_ids)
        if num_videos == 0:
            print("No videos to predict")
            return

        # Validate that all required files exist
        missing_files = []
        for expt_id in expt_ids:
            video_path = videos_dir / f'{expt_id}.mp4'
            label_path = labels_dir / f'{expt_id}.npy'
            if not video_path.exists():
                missing_files.append(f"Video missing: {video_path}")
            if not label_path.exists():
                missing_files.append(f"Label missing: {label_path}")
        
        if missing_files:
            raise FileNotFoundError(
                f"Missing files for prediction:\n" + "\n".join(missing_files)
            )

        # Configure trainer for prediction
        training_config = self.config.get('training', {})
        batch_size = training_config.get('batch_size', 1)
        num_gpus = torch.cuda.device_count()
        
        # Use single GPU for prediction (simpler and more reliable)
        if num_gpus == 0:
            raise RuntimeError(
                'No GPU detected. DALI-based video processing requires a GPU.'
            )
        accelerator = 'gpu'
        devices = 1
        
        trainer_config = {
            'accelerator': accelerator,
            'devices': devices,
            'strategy': 'auto',
            'logger': False,
            'enable_checkpointing': False,
            'enable_progress_bar': True,
            'precision': '16-mixed' if accelerator == 'gpu' else '32-true',
            'use_distributed_sampler': False,
        }
        
        trainer = pl.Trainer(**trainer_config)
        
        # Create datamodule for prediction
        data_config = self.config['data'].copy()
        data_config['expt_ids'] = expt_ids
        data_config['videos_dir'] = str(videos_dir)
        data_config['labels_dir'] = str(labels_dir)

        datamodule = VideoDataModule(
            data_config=data_config,
            sequence_length=training_config.get('sequence_length', 128),
            batch_size=batch_size,
            num_workers=0,  # Avoid multiprocessing issues during prediction
            train_probability=1.0,
            val_probability=0.0,
            seed=training_config.get('seed', 42),
            model_config=self.config.get('model', {}),
        )
        
        # Setup for prediction
        datamodule.setup('predict')
        
        # Run prediction
        predictions = trainer.predict(self.model, datamodule=datamodule)
        
        if predictions is None:
            return
        
        # Get the videos that were processed
        processed_videos = datamodule._predict_video_paths
        
        if processed_videos is None or len(processed_videos) == 0:
            return
        
        # Flatten predictions from batches
        flat_predictions = []
        for batch_predictions in predictions:
            if batch_predictions is not None:
                for sample_probs in batch_predictions:
                    flat_predictions.append(sample_probs.cpu().numpy())
        
        if not flat_predictions:
            return
        
        # Calculate how many chunks each video produced
        step_size = datamodule.sequence_length
        chunk_counts = []
        for video_path in processed_videos:
            video_name = os.path.basename(video_path)
            label_path = labels_dir / video_name.replace('.mp4', '.npy')
            if label_path.exists():
                labels_array = np.load(str(label_path), mmap_mode='r')
                video_frames = len(labels_array)
                num_chunks = max(1, (video_frames + step_size - 1) // step_size)
                chunk_counts.append(num_chunks)
            else:
                chunk_counts.append(0)
        
        # Assign predictions to videos and save
        chunk_idx = 0
        for video_idx, video_path in enumerate(processed_videos):
            video_name = Path(video_path).stem
            num_chunks = chunk_counts[video_idx]
            
            # Collect all predictions for this video
            video_probs = []
            for _ in range(num_chunks):
                if chunk_idx < len(flat_predictions):
                    video_probs.append(flat_predictions[chunk_idx])
                    chunk_idx += 1
            
            if not video_probs:
                continue
            
            # Stack predictions from all chunks
            stacked_probs = np.vstack(video_probs)
            num_classes = stacked_probs.shape[1]
            
            # Adjust prediction count to match actual video length
            label_path = labels_dir / f'{video_name}.npy'
            if label_path.exists():
                labels_array = np.load(str(label_path), mmap_mode='r')
                total_frames = labels_array.shape[0]
                
                predicted_frames = stacked_probs.shape[0]
                if predicted_frames != total_frames:
                    if predicted_frames > total_frames:
                        # Trim excess predictions
                        final_probs = stacked_probs[:total_frames, :]
                    else:
                        # Pad with NaN for missing frames
                        pad_rows = total_frames - predicted_frames
                        nan_pad = np.full((pad_rows, num_classes), np.nan)
                        final_probs = np.vstack([stacked_probs, nan_pad])
                else:
                    final_probs = stacked_probs
            else:
                final_probs = stacked_probs
            
            # Create output DataFrame
            df = pd.DataFrame(
                data=final_probs, 
                columns=datamodule.get_label_names()
            )
            df.insert(0, 'frame', np.arange(len(df)))
            
            # Determine output path
            if output_file is not None and len(expt_ids) == 1:
                output_file_ = Path(output_file)
            else:
                output_file_ = output_dir / f'{video_name}_predictions.csv'
            
            # Save predictions
            output_file_.parent.mkdir(exist_ok=True, parents=True)
            df.to_csv(output_file_, index=False)

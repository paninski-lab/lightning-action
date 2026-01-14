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

import cv2
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
def _get_video_frame_count(video_path: str | Path) -> int:
    """Get the total frame count of a video using OpenCV.
    
    Args:
        video_path: Path to the video file.
    
    Returns:
        Number of frames in the video.
    
    Raises:
        RuntimeError: If the video cannot be opened or frame count is invalid.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    if frame_count <= 0:
        raise RuntimeError(f"Invalid frame count ({frame_count}) for video: {video_path}")
    
    return frame_count


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
            output_dir='results',
        )
    """

    @typechecked
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
        self._trainer: Optional[pl.Trainer] = None

    @classmethod
    @typechecked
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
    @typechecked
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

    @typechecked
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
        expt_ids = self.config['data']['expt_ids']
        
        predictions_dir = self.model_dir / 'predictions'
        predictions_dir.mkdir(exist_ok=True)
        
        try:
            self.predict(
                videos_dir=videos_dir,
                output_dir=predictions_dir,
                expt_ids=expt_ids,
            )
        except Exception as e:
            print(f'Warning: Post-training inference failed: {e}')
    
    @typechecked
    def _setup_trainer(self) -> pl.Trainer:
        """Set up a Lightning Trainer for prediction.
        
        Returns:
            Configured Trainer instance for single-GPU prediction.
        
        Raises:
            RuntimeError: If no GPU is available.
        """
        num_gpus = torch.cuda.device_count()
        
        if num_gpus == 0:
            raise RuntimeError(
                'No GPU detected. DALI-based video processing requires a GPU.'
            )
        
        trainer_config = {
            'accelerator': 'gpu',
            'devices': 1,
            'strategy': 'auto',
            'logger': False,
            'enable_checkpointing': False,
            'enable_progress_bar': True,
            'precision': '16-mixed',
            'use_distributed_sampler': False,
        }
        
        return pl.Trainer(**trainer_config)
    
    @typechecked
    def _predict_single_video(
        self,
        video_path: Path,
        output_path: Path,
        trainer: pl.Trainer,
    ) -> None:
        """Run prediction on a single video and save results.
        
        This method handles the complete prediction pipeline for one video:
        1. Get video length using OpenCV
        2. Set up a VideoDataModule for this video
        3. Run inference
        4. Reformat predictions and save to CSV
        
        Args:
            video_path: Path to the input video file.
            output_path: Path where the prediction CSV will be saved.
            trainer: Lightning Trainer instance for running prediction.
        
        Raises:
            FileNotFoundError: If the video file doesn't exist.
            RuntimeError: If prediction fails.
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Get video metadata using OpenCV
        video_frame_count = _get_video_frame_count(video_path)
        
        # Extract config values
        training_config = self.config.get('training', {})
        batch_size = training_config.get('batch_size', 1)
        sequence_length = training_config.get('sequence_length', 128)
        
        # Create data config for single video
        model_config = self.config.get('model', {})
        data_config_from_model = self.config.get('data', {})
        data_config = {
            'videos_dir': str(video_path.parent),
            'expt_ids': [video_path.stem],
            'video_lengths': {video_path.stem: video_frame_count},
            'num_classes': model_config.get('output_size', 3),
            'label_names': data_config_from_model.get('label_names'),
        }
        
        # Create datamodule for this single video
        datamodule = VideoDataModule(
            data_config=data_config,
            sequence_length=sequence_length,
            batch_size=batch_size,
            num_workers=0,
            train_probability=1.0,
            val_probability=0.0,
            seed=training_config.get('seed', 42),
            model_config=self.config.get('model', {}),
        )
        
        # Setup for prediction
        datamodule.setup('predict')
        
        # Run prediction
        predictions = trainer.predict(self.model, datamodule=datamodule)
        
        if predictions is None or len(predictions) == 0:
            print(f"Warning: No predictions generated for {video_path.name}")
            return
        
        # Flatten predictions from batches
        flat_predictions = []
        for batch_predictions in predictions:
            if batch_predictions is not None:
                for sample_probs in batch_predictions:
                    flat_predictions.append(sample_probs.cpu().numpy())
        
        if not flat_predictions:
            print(f"Warning: Empty predictions for {video_path.name}")
            return
        
        # Stack all chunk predictions
        stacked_probs = np.vstack(flat_predictions)
        num_classes = stacked_probs.shape[1]
        predicted_frames = stacked_probs.shape[0]
        
        # Adjust to match actual video length
        if predicted_frames != video_frame_count:
            if predicted_frames > video_frame_count:
                # Trim excess predictions
                final_probs = stacked_probs[:video_frame_count, :]
            else:
                # Pad with NaN for missing frames
                pad_rows = video_frame_count - predicted_frames
                nan_pad = np.full((pad_rows, num_classes), np.nan)
                final_probs = np.vstack([stacked_probs, nan_pad])
        else:
            final_probs = stacked_probs
        
        # Get label names from datamodule or generate defaults
        label_names = datamodule.get_label_names()
        if not label_names:
            label_names = [f'class_{i}' for i in range(num_classes)]
        
        # Create output DataFrame
        df = pd.DataFrame(data=final_probs, columns=label_names)
        df.insert(0, 'frame', np.arange(len(df)))
        
        # Save predictions
        output_path.parent.mkdir(exist_ok=True, parents=True)
        df.to_csv(output_path, index=False)
    
    @typechecked
    def predict(
        self,
        videos_dir: str | Path,
        output_dir: str | Path,
        output_file: Optional[str | Path] = None,
        expt_ids: Optional[list[str]] = None,
    ) -> None:
        """Generate predictions for videos.
        
        Processes each video sequentially on a single GPU. For large-scale
        inference, videos can be split across multiple calls.
        
        Output format is CSV with columns:
        - frame: Frame index (0-based)
        - class_0, class_1, ...: Probability for each class
        
        Args:
            videos_dir: Directory containing .mp4 video files.
            output_dir: Directory to save prediction CSV files.
            output_file: If predicting a single video, specify output filename.
                Only valid when expt_ids contains exactly one video.
            expt_ids: List of experiment IDs (video stems) to predict.
                If None, predicts all .mp4 files in videos_dir.
        
        Raises:
            ValueError: If model hasn't been trained/loaded.
            RuntimeError: If output_file specified with multiple videos.
            FileNotFoundError: If any video file is missing.
        """
        videos_dir = Path(videos_dir)
        output_dir = Path(output_dir)

        if self.model is None:
            raise ValueError('Model must be trained or loaded before prediction')

        # Validate output_file usage
        if output_file is not None and expt_ids is not None and len(expt_ids) > 1:
            raise RuntimeError(
                'Can only supply `output_file` when specifying a single expt_id'
            )

        # Discover videos if not specified
        if expt_ids is None:
            expt_ids = [f.stem for f in videos_dir.glob('*.mp4')]

        if len(expt_ids) == 0:
            print("No videos to predict")
            return

        # Validate that all video files exist
        missing_videos = []
        for expt_id in expt_ids:
            video_path = videos_dir / f'{expt_id}.mp4'
            if not video_path.exists():
                missing_videos.append(str(video_path))
        
        if missing_videos:
            raise FileNotFoundError(
                f"Missing video files:\n" + "\n".join(missing_videos)
            )

        # Set up trainer once for all predictions
        trainer = self._setup_trainer()
        
        # Process each video
        for i, expt_id in enumerate(expt_ids):
            video_path = videos_dir / f'{expt_id}.mp4'
            
            # Determine output path
            if output_file is not None and len(expt_ids) == 1:
                output_path = Path(output_file)
            else:
                output_path = output_dir / f'{expt_id}_predictions.csv'
            
            print(f"Processing video {i+1}/{len(expt_ids)}: {expt_id}")
            
            try:
                self._predict_single_video(
                    video_path=video_path,
                    output_path=output_path,
                    trainer=trainer,
                )
            except Exception as e:
                print(f"Error processing {expt_id}: {e}")
                raise

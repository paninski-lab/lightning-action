"""High-level API for video action segmentation models.

This module provides a user-friendly VideoModel class that wraps the
underlying VideoSegmenter model and handles common workflows:

- Loading trained models from disk
- Creating new models from configuration
- Training models with automatic post-training inference
- Running inference on new videos

The VideoModel class inherits from BaseModelAPI and provides video-specific:
- GPU-accelerated prediction using DALI
- Frame-by-frame classification from raw video files
- Video-level post-training inference

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

from pathlib import Path
from typing import Any

import cv2
import lightning as pl
import numpy as np
import pandas as pd
import torch
from typeguard import typechecked

from lightning_action.api.base import BaseModelAPI
from lightning_action.data.video_datamodule import VideoDataModule
from lightning_action.models.video_segmenter import VideoSegmenter
from lightning_action.video_train import train_video


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
class VideoModel(BaseModelAPI[VideoSegmenter]):
    """High-level wrapper for video action segmentation models.

    This class provides a simplified interface for working with video
    segmentation models, handling the complexity of model loading,
    training, and inference behind a clean API.

    Inherits from BaseModelAPI and provides video-specific implementations
    for model creation, training, and GPU-accelerated prediction.

    Attributes:
        model: The underlying VideoSegmenter Lightning module.
        config: Configuration dictionary used to create/load the model.
        model_dir: Directory where the model is stored (after training/loading).
    """

    @classmethod
    def _get_model_class(cls) -> type:
        """Return the VideoSegmenter model class."""
        return VideoSegmenter

    @classmethod
    def _get_train_function(cls):
        """Return the video pipeline training function."""
        return train_video

    @classmethod
    def _create_model_from_config(cls, config: dict[str, Any]) -> VideoSegmenter:
        """Create a VideoSegmenter model from configuration.

        Args:
            config: Configuration dictionary.

        Returns:
            Initialized VideoSegmenter model.
        """
        return VideoSegmenter(config)

    def _run_post_training_inference(self) -> None:
        """Run inference on all training videos after training.

        Generates prediction CSV files for each video in the training set,
        which is useful for evaluating model performance and debugging.
        """
        if self.model is None or self.model_dir is None:
            return

        videos_dir = self.config['data']['videos_dir']
        expt_ids = self.config['data'].get('expt_ids')

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

        # Get video metadata
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

        # Create datamodule
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
                final_probs = stacked_probs[:video_frame_count, :]
            else:
                pad_rows = video_frame_count - predicted_frames
                nan_pad = np.full((pad_rows, num_classes), np.nan)
                final_probs = np.vstack([stacked_probs, nan_pad])
        else:
            final_probs = stacked_probs

        # Get label names
        label_names = datamodule.get_label_names()
        if not label_names:
            label_names = [f'class_{i}' for i in range(num_classes)]

        # Create and save output DataFrame
        df = pd.DataFrame(data=final_probs, columns=label_names)
        df.insert(0, 'frame', np.arange(len(df)))

        output_path.parent.mkdir(exist_ok=True, parents=True)
        df.to_csv(output_path, index=False)

    @typechecked
    def predict(
        self,
        videos_dir: str | Path,
        output_dir: str | Path,
        output_file: str | Path | None = None,
        expt_ids: list[str] | None = None,
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

        if output_file is not None and expt_ids is not None and len(expt_ids) > 1:
            raise RuntimeError('Can only supply `output_file` when specifying a single expt_id')

        # Discover videos if not specified
        if expt_ids is None:
            expt_ids = [f.stem for f in videos_dir.glob('*.mp4')]

        if len(expt_ids) == 0:
            print("No videos to predict")
            return

        # Validate video files exist
        missing_videos = []
        for expt_id in expt_ids:
            video_path = videos_dir / f'{expt_id}.mp4'
            if not video_path.exists():
                missing_videos.append(str(video_path))

        if missing_videos:
            raise FileNotFoundError("Missing video files:\n" + "\n".join(missing_videos))

        # Set up trainer once for all predictions
        trainer = self._setup_trainer()

        # Process each video
        for i, expt_id in enumerate(expt_ids):
            video_path = videos_dir / f'{expt_id}.mp4'

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

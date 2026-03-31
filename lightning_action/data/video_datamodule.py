"""DataModule for video action segmentation using NVIDIA DALI.

This module provides GPU-accelerated video loading and preprocessing using
NVIDIA DALI (Data Loading Library). DALI handles video decoding, resizing,
and normalization on the GPU, significantly improving training throughput.

Key components:
- VideoPipeline: DALI pipeline definition for video processing
- DALIIterator: Custom iterator that pairs video frames with labels
- VideoDataModule: Lightning DataModule orchestrating the data pipeline

The data flow is:
1. VideoDataModule.setup() splits VIDEOS into train/val sets
2. train_dataloader() creates a DALI pipeline for training videos
3. DALIIterator fetches batches and attaches corresponding labels
4. Frames arrive as (B, T, C, H, W) tensors ready for the model

Train/Val Split Strategy:
- Splits are done at the VIDEO level, not chunk level
- This ensures no data leakage between train and validation
- DALI handles chunking independently at runtime

Typical usage:
    datamodule = VideoDataModule(
        data_config={'videos_dir': '...', 'labels_dir': '...', 'expt_ids': [...]},
        sequence_length=500,
        batch_size=2,
    )
    datamodule.setup('fit')

    trainer = pl.Trainer(...)
    trainer.fit(model, datamodule)
"""

import logging
import os
import tempfile
from typing import Any, Optional, Union

import cv2
import lightning as pl
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from typeguard import typechecked

from lightning_action.data.utils import (
    compute_class_weights,
    split_sizes_from_probabilities,
)
from lightning_action.data.video_dataset import VideoDataset

logger = logging.getLogger(__name__)

DALI_AVAILABLE = False
try:
    from nvidia.dali import fn, types
    from nvidia.dali.pipeline import Pipeline
    from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
    DALI_AVAILABLE = True
except ImportError:
    fn = types = LastBatchPolicy = None
    Pipeline = object
    DALIGenericIterator = object

def _check_dali_available():
    """Raise informative error if DALI is not available."""
    if not DALI_AVAILABLE:
        raise ImportError(
            "NVIDIA DALI is required for GPU video loading but is not installed. "
            "Install with: pip install nvidia-dali-cuda120  # (or appropriate CUDA version)\n"
            "See: https://docs.nvidia.com/deeplearning/dali/user-guide/docs/installation.html"
        )


TRANSFORM_PRESETS = {
    # Current default - standard ImageNet RGB normalization
    "imagenet": {
        "mean": [0.485 * 255, 0.456 * 255, 0.406 * 255],
        "std": [0.229 * 255, 0.224 * 255, 0.225 * 255],
        "grayscale": False,
    },
    # Convert to grayscale first, then ImageNet normalization
    # (matches BEAST's training: grayscale -> 3ch RGB -> ImageNet norm)
    "imagenet_gray": {
        "mean": [0.485 * 255, 0.456 * 255, 0.406 * 255],
        "std": [0.229 * 255, 0.224 * 255, 0.225 * 255],
        "grayscale": True,
    },
    # Uniform mean/std across channels (luminance-weighted ImageNet average)
    "imagenet_uniform": {
        "mean": [0.449 * 255] * 3,
        "std": [0.226 * 255] * 3,
        "grayscale": False,
    },
    # Uniform ImageNet on grayscale-converted frames
    "imagenet_uniform_gray": {
        "mean": [0.449 * 255] * 3,
        "std": [0.226 * 255] * 3,
        "grayscale": True,
    },
    # Simple 0-1 scaling (no ImageNet stats)
    "scale_01": {
        "mean": [0.0, 0.0, 0.0],
        "std": [255.0, 255.0, 255.0],
        "grayscale": False,
    },
    # Grayscale + 0-1 scaling
    "scale_01_gray": {
        "mean": [0.0, 0.0, 0.0],
        "std": [255.0, 255.0, 255.0],
        "grayscale": True,
    },
    # Scale to [-1, 1]
    "scale_neg1_1": {
        "mean": [127.5, 127.5, 127.5],
        "std": [127.5, 127.5, 127.5],
        "grayscale": False,
    },
    # Grayscale + scale to [-1, 1]
    "scale_neg1_1_gray": {
        "mean": [127.5, 127.5, 127.5],
        "std": [127.5, 127.5, 127.5],
        "grayscale": True,
    },
    # No normalization - raw uint8 values as float
    "none": {
        "mean": [0.0, 0.0, 0.0],
        "std": [1.0, 1.0, 1.0],
        "grayscale": False,
    },
}


AUGMENTATION_PRESETS = {
    "none": {},
    "default": {
        "horizontal_flip": {"probability": 0.5},
        "random_crop": {"scale_min": 0.9, "scale_max": 1.0},
        "brightness_contrast": {
            "brightness_range": 0.15,
            "contrast_range": 0.15,
            "probability": 0.5,
        },
    },
    "strong": {
        "horizontal_flip": {"probability": 0.5},
        "random_crop": {"scale_min": 0.8, "scale_max": 1.0},
        "brightness_contrast": {
            "brightness_range": 0.25,
            "contrast_range": 0.25,
            "probability": 0.7,
        },
        "rotation": {"max_angle": 10.0, "probability": 0.3},
    },
    "topdown": {
        "horizontal_flip": {"probability": 0.5},
        "random_crop": {"scale_min": 0.85, "scale_max": 1.0},
        "brightness_contrast": {
            "brightness_range": 0.15,
            "contrast_range": 0.15,
            "probability": 0.5,
        },
        "rotation": {"max_angle": 15.0, "probability": 0.5},
    },
}


def _resolve_augmentation_config(config) -> dict:
    """Resolve augmentation config from string preset, dict, or None.

    Args:
        config: One of:
            - None -> no augmentation (empty dict)
            - str -> preset name from AUGMENTATION_PRESETS
            - dict with 'preset' key -> base preset with overrides
            - dict without 'preset' key -> raw augmentation config

    Returns:
        Resolved augmentation config dict.

    Raises:
        ValueError: If string preset name is unknown.
    """
    if config is None:
        return {}
    if isinstance(config, str):
        if config not in AUGMENTATION_PRESETS:
            raise ValueError(
                f"Unknown augmentation preset '{config}'. "
                f"Available presets: {sorted(AUGMENTATION_PRESETS.keys())}"
            )
        return AUGMENTATION_PRESETS[config].copy()
    if isinstance(config, dict):
        if 'preset' in config:
            preset_name = config['preset']
            if preset_name not in AUGMENTATION_PRESETS:
                raise ValueError(
                    f"Unknown augmentation preset '{preset_name}'. "
                    f"Available presets: {sorted(AUGMENTATION_PRESETS.keys())}"
                )
            resolved = {}
            for key, value in AUGMENTATION_PRESETS[preset_name].items():
                resolved[key] = value.copy() if isinstance(value, dict) else value
            # Merge overrides
            for key, value in config.items():
                if key == 'preset':
                    continue
                if key in resolved and isinstance(resolved[key], dict) and isinstance(value, dict):
                    resolved[key].update(value)
                else:
                    resolved[key] = value
            return resolved
        return config.copy()
    return {}


class VideoPipeline(Pipeline):
    """DALI pipeline for GPU-accelerated video decoding and preprocessing.
    
    This pipeline:
    1. Reads video files using GPU-accelerated decoding
    2. Extracts sequences of frames with configurable step size
    3. Resizes frames to target resolution
    4. Normalizes using ImageNet statistics
    
    The pipeline outputs:
    - frames: Preprocessed video frames (FCHW format)
    - frame_idx: Video index in the file list
    - start_frame: Starting frame number within the video
    """
    
    def __init__(
        self,
        batch_size: int,
        num_threads: int,
        device_id: int,
        seed: int,
        file_list: str,
        sequence_length: int,
        resolution: int,
        random_shuffle: bool,
        step: Optional[int] = None,
        shard_id: int = 0,
        num_shards: int = 1,
        pad_sequences: bool = False,
        transform_mode: str = "imagenet",
        augmentations: Optional[dict] = None,
    ):
        """Initialize the DALI video pipeline.

        Args:
            batch_size: Number of sequences per batch.
            num_threads: Number of CPU threads for data loading.
            device_id: GPU device ID for decoding.
            seed: Random seed for shuffling.
            file_list: Path to text file listing videos (format: "path index").
            sequence_length: Number of frames per sequence (including padding).
            resolution: Target frame resolution (square).
            random_shuffle: Whether to shuffle sequences.
            step: Frame step between sequences. Defaults to sequence_length.
            shard_id: Shard index for distributed training.
            num_shards: Total number of shards for distributed training.
            pad_sequences: Whether to pad the last sequence of each video.
            transform_mode: Transform preset name from TRANSFORM_PRESETS.
            augmentations: Augmentation config dict (from _resolve_augmentation_config).
        """
        _check_dali_available()

        if transform_mode not in TRANSFORM_PRESETS:
            raise ValueError(
                f"Unknown transform_mode '{transform_mode}'. "
                f"Available presets: {sorted(TRANSFORM_PRESETS.keys())}"
            )

        super().__init__(
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=device_id,
            seed=seed,
            prefetch_queue_depth=batch_size,
            exec_pipelined=True,
            exec_async=True,
            py_num_workers=0
        )
        self.file_list = file_list
        self.sequence_length = sequence_length
        self.resolution = resolution
        self.random_shuffle = random_shuffle
        self.step = step if step is not None else sequence_length
        self.shard_id = shard_id
        self.num_shards = num_shards
        self.pad_sequences = pad_sequences
        self.transform_mode = transform_mode
        self._transform_preset = TRANSFORM_PRESETS[transform_mode]
        self._augmentations = augmentations or {}

    def _apply_augmentations(self, frames):
        """Apply DALI-native augmentations to video frames.

        All random parameters are generated per-sample (not per-frame) to
        maintain temporal consistency across frames in each sequence.

        Args:
            frames: DALI tensor of video frames (FHWC layout from reader).

        Returns:
            Augmented frames tensor.
        """
        aug = self._augmentations

        # 1. Horizontal flip
        if "horizontal_flip" in aug:
            prob = aug["horizontal_flip"].get("probability", 0.5)
            do_flip = fn.random.coin_flip(probability=prob)
            frames = fn.flip(frames, horizontal=do_flip)

        # 2. Rotation
        if "rotation" in aug:
            rot_cfg = aug["rotation"]
            prob = rot_cfg.get("probability", 0.3)
            max_angle = rot_cfg.get("max_angle", 10.0)
            do_rotate = fn.random.coin_flip(probability=prob)
            angle = fn.random.uniform(range=(-max_angle, max_angle))
            # Multiply angle by coin flip to get 0 when not rotating
            angle = angle * fn.cast(do_rotate, dtype=types.FLOAT)
            frames = fn.rotate(frames, angle=angle, keep_size=True, fill_value=0)

        # 3. Random crop via fn.random_resized_crop (avoids GPU->CPU shape query)
        # Outputs at (resolution, resolution) so the subsequent fn.resize is a no-op.
        # Scale range is converted to area range (area = scale^2).
        # No probability gate needed: randomness comes from scale/position sampling,
        # and crop degrades to a no-op when scale_min == scale_max == 1.0.
        if "random_crop" in aug:
            crop_cfg = aug["random_crop"]
            scale_min = crop_cfg.get("scale_min", 0.9)
            scale_max = crop_cfg.get("scale_max", 1.0)
            frames = fn.random_resized_crop(
                frames,
                size=(self.resolution, self.resolution),
                random_area=(scale_min ** 2, scale_max ** 2),
                random_aspect_ratio=(1.0, 1.0),
            )

        # 4. Brightness/contrast
        if "brightness_contrast" in aug:
            bc_cfg = aug["brightness_contrast"]
            prob = bc_cfg.get("probability", 0.5)
            b_range = bc_cfg.get("brightness_range", 0.15)
            c_range = bc_cfg.get("contrast_range", 0.15)
            do_bc = fn.random.coin_flip(probability=prob)
            brightness = fn.random.uniform(range=(-b_range, b_range))
            contrast = fn.random.uniform(range=(1.0 - c_range, 1.0 + c_range))
            # When not applying, use neutral values (brightness=0, contrast=1)
            do_bc_f = fn.cast(do_bc, dtype=types.FLOAT)
            brightness = brightness * do_bc_f
            contrast = contrast * do_bc_f + (1.0 - do_bc_f)
            frames = fn.brightness_contrast(
                frames, brightness=brightness, contrast=contrast,
            )

        return frames

    def define_graph(self):
        """Define the DALI processing graph."""
        frames, frame_idx, start_frame = fn.readers.video(
            device="gpu",
            file_list=self.file_list,
            file_list_frame_num=True,
            file_list_include_preceding_frame=False,
            enable_frame_num=True,
            sequence_length=self.sequence_length,
            step=self.step,
            shard_id=self.shard_id,
            num_shards=self.num_shards,
            random_shuffle=self.random_shuffle,
            dtype=types.UINT8,
            normalized=False,
            name="VideoReader",
            pad_sequences=self.pad_sequences,
            pad_last_batch=self.pad_sequences,
        )
        
        # Optional grayscale conversion (RGB -> GRAY -> RGB) to match
        # backbones pretrained on grayscale images converted to 3-channel RGB
        if self._transform_preset["grayscale"]:
            frames = fn.color_space_conversion(
                frames, image_type=types.RGB, output_type=types.GRAY)
            frames = fn.color_space_conversion(
                frames, image_type=types.GRAY, output_type=types.RGB)

        # Apply data augmentations (after grayscale, before resize)
        if self._augmentations:
            frames = self._apply_augmentations(frames)

        frames = fn.resize(frames, size=(224, 224))

        frames = fn.crop_mirror_normalize(
            frames,
            dtype=types.FLOAT,
            output_layout="FCHW",
            mean=self._transform_preset["mean"],
            std=self._transform_preset["std"],
        )

        return frames, frame_idx, start_frame



class DALIIterator(DALIGenericIterator):
    """Custom DALI iterator that pairs video frames with labels.
    
    This iterator wraps DALI's generic iterator to:
    1. Extract frames, video indices, and start frames from DALI batches
    2. Look up corresponding labels from a preloaded 2D label tensor
    3. Track metadata about each sample (video index, position, boundaries)
    """
    
    def __init__(
        self,
        pipe: Pipeline,
        output_map: list[str],
        sequence_length: int,
        tcn_padding: int,
        ignore_index: int,
        all_labels_2d: Optional[torch.Tensor] = None,
        include_labels: bool = False,
        include_lengths: bool = False,
        video_lengths: Optional[dict[int, int]] = None,
        last_batch_policy: Optional[Any] = None,
    ):
        """Initialize the DALI iterator.

        Args:
            pipe: Built DALI pipeline.
            output_map: Names for pipeline outputs.
            sequence_length: Number of frames in chunk middle.
            tcn_padding: Padding frames on each side.
            ignore_index: Label value for ignored frames.
            all_labels_2d: Preloaded labels tensor, shape (num_videos, max_frames).
            include_labels: Whether to return labels (True for train/val).
            include_lengths: Whether to return sequence lengths (True for predict).
            video_lengths: Dict mapping video_idx -> total frame count.
            last_batch_policy: How to handle incomplete final batch. Defaults to DROP.
        """

        _check_dali_available()

        # Set default last_batch_policy after DALI check (can't use as default param
        # because LastBatchPolicy may be None when DALI isn't installed)
        if last_batch_policy is None:
            last_batch_policy = LastBatchPolicy.DROP

        super().__init__(
            pipe,
            output_map,
            size=-1,
            auto_reset=True,
            last_batch_policy=last_batch_policy,
            reader_name="VideoReader",
        )
        self.sequence_length = sequence_length
        self.tcn_padding = tcn_padding
        self.ignore_index = ignore_index
        self.include_labels = include_labels
        self.include_lengths = include_lengths
        self.all_labels_2d = all_labels_2d
        self.video_lengths = video_lengths

    def _process_batch(self, batch):
        """Extract frames, labels, and metadata from a raw DALI batch."""
        frames = batch[0]['frames']
        frame_indices = batch[0]['frame_idx']
        start_frames = batch[0]['start_frame']

        frame_indices = frame_indices.squeeze()
        start_frames = start_frames.squeeze()

        if frame_indices.dim() == 0:
            frame_indices = frame_indices.unsqueeze(0)
        if start_frames.dim() == 0:
            start_frames = start_frames.unsqueeze(0)

        B, T = frames.shape[0], frames.shape[1]
        device = frames.device

        metadata = []
        for i in range(B):
            video_idx = frame_indices[i].item()
            start_frame = start_frames[i].item()

            is_start = (start_frame == 0)

            is_end = False
            if self.video_lengths and video_idx in self.video_lengths:
                video_len = self.video_lengths[video_idx]
                if start_frame + T >= video_len:
                    is_end = True

            meta = {
                'video_idx': video_idx,
                'dali_start_frame': start_frame,
                'num_frames': T,
                'is_start': is_start,
                'is_end': is_end,
            }
            metadata.append(meta)

        labels = None
        if self.include_labels and self.all_labels_2d is not None:
            seq_idxs = torch.arange(T, device=device)
            time_idx = start_frames.unsqueeze(1) + seq_idxs.unsqueeze(0)
            video_idx = frame_indices.unsqueeze(1).expand(-1, T)

            max_frames = self.all_labels_2d.shape[1]
            time_idx = torch.clamp(time_idx, 0, max_frames - 1)

            labels = self.all_labels_2d[video_idx, time_idx]

        return frames, labels, metadata

    def __next__(self):
        """Get the next batch with frames, labels, and metadata."""
        max_retries = 50
        for attempt in range(max_retries + 1):
            batch = super().__next__()
            frames, labels, metadata = self._process_batch(batch)

            if labels is not None:
                return frames, labels, metadata

            if self.include_lengths:
                actual_lengths = [self.sequence_length for _ in range(frames.shape[0])]
                return frames, actual_lengths, metadata

            return frames, None, metadata

        logger.warning(
            "Sequence filter exhausted %d retries; returning unfiltered batch", max_retries
        )
        return frames, labels, metadata


def _get_video_frame_count_cv2(video_path: str) -> int:
    """Get frame count from video file using OpenCV.
    
    Args:
        video_path: Path to video file.
    
    Returns:
        Number of frames, or 0 if video cannot be opened.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count


def _load_label_file(label_path: str) -> Optional[np.ndarray]:
    """Load a single label file and convert to 1D class indices.

    Handles both 1D integer arrays and 2D one-hot encoded arrays.

    Args:
        label_path: Path to a .npy label file.

    Returns:
        1D array of integer class indices, or None if the file doesn't exist.
    """
    try:
        labels = np.load(label_path)
    except FileNotFoundError:
        return None

    if labels.ndim > 1 and labels.shape[1] > 1:
        labels = np.argmax(labels, axis=1)
    elif labels.ndim > 1:
        labels = labels.squeeze()

    return labels


def _label_path_from_video(video_path: str, labels_dir: str) -> str:
    """Construct a label file path from a video file path."""
    video_name = os.path.basename(video_path)
    return os.path.join(labels_dir, video_name.replace('.mp4', '.npy'))


def load_labels_for_videos(
    video_paths: list[str], 
    labels_dir: Optional[str], 
    max_frames: int, 
    ignore_index: int = -100,
) -> Optional[torch.Tensor]:
    """Load labels into a 2D tensor for efficient batched lookup.
    
    Args:
        video_paths: List of video file paths (order matches DALI file_list).
        labels_dir: Directory containing .npy label files, or None.
        max_frames: Maximum frame count across all videos.
        ignore_index: Value for padding/missing frames.
    
    Returns:
        Tensor of shape (num_videos, max_frames) with class labels,
        or None if labels_dir is not provided.
    """
    if labels_dir is None:
        return None
    
    num_videos = len(video_paths)
    all_labels = np.full((num_videos, max_frames), ignore_index, dtype=np.int64)
    
    for idx, video_path in enumerate(video_paths):
        label_path = _label_path_from_video(video_path, labels_dir)
        labels = _load_label_file(label_path)
        if labels is None:
            continue
        n_frames = min(len(labels), max_frames)
        all_labels[idx, :n_frames] = labels[:n_frames]
    
    return torch.from_numpy(all_labels)


def get_video_lengths(
    video_paths: list[str], 
    labels_dir: Optional[str] = None,
    precomputed_lengths: Optional[dict[str, int]] = None,
) -> dict[int, int]:
    """Get video lengths (frame counts) for each video index.
    
    Attempts to get lengths from:
    1. Precomputed lengths dict (if provided)
    2. Label files (if labels_dir provided)
    3. OpenCV as fallback
    
    Args:
        video_paths: List of video file paths.
        labels_dir: Directory containing .npy label files, or None.
        precomputed_lengths: Optional dict mapping video stem -> frame count.
    
    Returns:
        Dict mapping video index -> frame count.
    """
    lengths = {}
    
    for idx, video_path in enumerate(video_paths):
        video_name = os.path.basename(video_path)
        video_stem = video_name.replace('.mp4', '')
        
        # Try precomputed lengths first
        if precomputed_lengths and video_stem in precomputed_lengths:
            lengths[idx] = precomputed_lengths[video_stem]
            continue
        
        # Try label files
        if labels_dir:
            label_path = os.path.join(labels_dir, video_name.replace('.mp4', '.npy'))
            if os.path.exists(label_path):
                labels = np.load(label_path, mmap_mode='r')
                lengths[idx] = len(labels)
                continue
        
        # Fall back to OpenCV
        frame_count = _get_video_frame_count_cv2(video_path)
        if frame_count > 0:
            lengths[idx] = frame_count
    
    return lengths


def get_max_frames(
    video_paths: list[str], 
    labels_dir: Optional[str] = None,
    precomputed_lengths: Optional[dict[str, int]] = None,
) -> int:
    """Get maximum frame count across a list of videos.
    
    Args:
        video_paths: List of video file paths.
        labels_dir: Directory containing .npy label files, or None.
        precomputed_lengths: Optional dict mapping video stem -> frame count.
    
    Returns:
        Maximum frame count across all videos.
    """
    max_frames = 0
    
    for video_path in video_paths:
        video_name = os.path.basename(video_path)
        video_stem = video_name.replace('.mp4', '')
        
        # Try precomputed lengths first
        if precomputed_lengths and video_stem in precomputed_lengths:
            max_frames = max(max_frames, precomputed_lengths[video_stem])
            continue
        
        # Try label files
        if labels_dir:
            label_path = os.path.join(labels_dir, video_name.replace('.mp4', '.npy'))
            if os.path.exists(label_path):
                labels = np.load(label_path, mmap_mode='r')
                max_frames = max(max_frames, len(labels))
                continue
        
        # Fall back to OpenCV
        frame_count = _get_video_frame_count_cv2(video_path)
        if frame_count > 0:
            max_frames = max(max_frames, frame_count)
    
    return max_frames



class VideoDataModule(pl.LightningDataModule):
    """Lightning DataModule for video action segmentation.
    
    This DataModule orchestrates the entire data pipeline:
    1. Creates a VideoDataset to discover videos and compute class weights
    2. Splits VIDEOS into train/val sets (video-level, not chunk-level)
    3. Creates DALI-based dataloaders for efficient GPU video processing
    
    The train/val split is done at the VIDEO level to prevent data leakage.
    DALI handles chunking independently at runtime.
    
    Attributes:
        data_config: Configuration dict with videos_dir, labels_dir, etc.
        sequence_length: Number of frames per chunk (excluding padding).
        batch_size: Number of sequences per batch.
        train_probability: Fraction of videos for training.
        val_probability: Fraction of videos for validation.
        dataset: VideoDataset instance with video metadata.
        train_video_paths: Video paths assigned to training.
        val_video_paths: Video paths assigned to validation.
    """

    @typechecked
    def __init__(
        self,
        data_config: dict[str, Any],
        sequence_length: int = 128,
        batch_size: int = 1,
        num_workers: int = 0,
        train_probability: float = 0.95,
        val_probability: float = 0.05,
        seed: int = 0,
        model_config: Optional[dict[str, Any]] = None,
    ):
        """Initialize the VideoDataModule.

        Args:
            data_config: Dict containing videos_dir, and optionally labels_dir,
                expt_ids, video_lengths, etc.
            sequence_length: Number of frames per chunk middle section.
            batch_size: Sequences per batch.
            num_workers: CPU workers for data loading.
            train_probability: Fraction of videos for training.
            val_probability: Fraction of videos for validation.
            seed: Random seed for reproducible splits.
            model_config: Model configuration for TCN padding calculation.

        Raises:
            ValueError: If probabilities are out of range or sum > 1.
        """
        super().__init__()

        self.data_config = data_config
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_threads = max(1, num_workers // 2)
        self.train_probability = train_probability
        self.val_probability = val_probability
        self.seed = seed
        self.model_config = model_config or {}
        
        self.videos_dir = data_config['videos_dir']
        self.labels_dir = data_config.get('labels_dir')  # Now optional
        self.precomputed_lengths = data_config.get('video_lengths')  # Optional precomputed

        # Video transform preset for normalization/grayscale conversion
        self.transform_mode = data_config.get('transform_mode', 'imagenet')
        if self.transform_mode not in TRANSFORM_PRESETS:
            raise ValueError(
                f"Unknown transform_mode '{self.transform_mode}'. "
                f"Available presets: {sorted(TRANSFORM_PRESETS.keys())}"
            )

        # Data augmentation config (only applied to training data)
        self._augmentation_config = _resolve_augmentation_config(
            data_config.get('augmentations')
        )
        
        # Validate probabilities
        if not 0 <= train_probability <= 1:
            raise ValueError('train_probability must be in [0, 1]')
        if not 0 <= val_probability <= 1:
            raise ValueError('val_probability must be in [0, 1]')
        if train_probability + val_probability > 1:
            raise ValueError('train_probability + val_probability must be <= 1')
        
        # State variables - will be set in setup()
        self.train_video_paths: Optional[list[str]] = None
        self.val_video_paths: Optional[list[str]] = None
        self.current_epoch: int = 0
        self.validation_enabled: bool = True
        self._predict_video_paths: Optional[list[str]] = None
        
        # Adjust threads for distributed training
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            if world_size > 1:
                self.num_threads = max(1, os.cpu_count() // world_size)

        # Determine num_lags based on head type
        head_type = self.model_config.get('head', 'dtcn').lower()
        if head_type in ['dtcn', 'dilatedtcn', 'temporalmlp']:
            num_lags = self.model_config.get('num_lags', 2)
        else:
            num_lags = (
                0 if not self.model_config.get('bidirectional', False) 
                else self.model_config.get('num_lags', 2)
            )

        # Create dataset (discovers videos, computes class weights if labels available)
        self.dataset = VideoDataset(
            videos_dir=self.data_config['videos_dir'],
            labels_dir=self.labels_dir,
            sequence_length=self.sequence_length,
            resolution=self.data_config.get('resolution', 224),
            expt_ids=self.data_config.get('expt_ids'),
            input_size=self.data_config.get('input_size', 1536),
            num_lags=num_lags,
            ignore_index=self.data_config.get('ignore_index', -100),
            head=self.model_config.get('head', 'dtcn'),
            num_layers=self.model_config.get('num_layers', 4),
            require_labels=self.labels_dir is not None,
            num_classes=self.data_config.get('num_classes'),
            label_names=self.data_config.get('label_names'),
        )


    def setup(self, stage: Optional[str] = None) -> None:
        """Set up train/val/predict video splits.
        
        Splits are done at the VIDEO level for clean separation.
        
        Args:
            stage: One of 'fit', 'validate', 'test', 'predict', or None.
        """
        if stage in [None, 'fit', 'validate']:
            if self.train_video_paths is None or self.val_video_paths is None:
                # Split at VIDEO level
                num_videos = len(self.dataset)
                train_size, val_size = split_sizes_from_probabilities(
                    num_videos,
                    self.train_probability,
                    self.val_probability,
                )
                
                # Reproducible shuffle and split
                np.random.seed(self.seed)
                indices = np.random.permutation(num_videos)
                
                train_indices = indices[:train_size]
                val_indices = indices[train_size:train_size + val_size]
                
                self.train_video_paths = [self.dataset.video_paths[i] for i in train_indices]
                self.val_video_paths = [self.dataset.video_paths[i] for i in val_indices]
                
                # Determine if validation is possible
                if val_size == 0 or len(self.val_video_paths) == 0:
                    self.validation_enabled = False
                else:
                    self.validation_enabled = True
        
        if stage == 'predict':
            all_videos = self.dataset.video_paths
            
            # Distribute videos across GPUs for multi-GPU prediction
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
                world_size = torch.distributed.get_world_size()
                self._predict_video_paths = all_videos[rank::world_size]
            else:
                self._predict_video_paths = all_videos

    def compute_filtered_class_weights(self) -> list[float]:
        """Compute class weights from training video labels.

        Returns:
            List of class weights, one per class.
        """
        return list(self.dataset.class_weights)

    def _create_dataloader(
        self,
        video_list: list[str],
        labels_2d: Optional[torch.Tensor],
        include_labels: bool,
        include_lengths: bool,
        shuffle: bool = True,
        last_batch_policy_str: str = 'drop',
        use_dali_sharding: bool = True,
        for_prediction: bool = False,
        augment: bool = False,
    ) -> Optional[DALIIterator]:
        """Create a DALI-based dataloader for a list of videos.

        Args:
            video_list: List of video paths to include.
            labels_2d: Preloaded labels tensor, or None for prediction.
            include_labels: Whether iterator should return labels.
            include_lengths: Whether iterator should return sequence lengths.
            shuffle: Whether to shuffle sequences.
            last_batch_policy_str: 'drop' or 'partial' for final batch.
            use_dali_sharding: Whether to use DALI's built-in sharding.
            for_prediction: Whether this is for prediction (enables padding).
            augment: Whether to apply data augmentations.

        Returns:
            DALIIterator instance, or None if video_list is empty.
        """

        _check_dali_available()

        if not video_list or len(video_list) == 0:
            return None
        
        last_batch_policy = (
            LastBatchPolicy.DROP if last_batch_policy_str.lower() == 'drop' 
            else LastBatchPolicy.PARTIAL
        )
        
        # Vary seed by epoch for training shuffles
        epoch_seed = self.seed + self.current_epoch if shuffle else self.seed
        
        # Create DALI file list
        file_list_entries = []
        for idx, video_path in enumerate(video_list):
            file_list_entries.append(f"{video_path} {idx}\n")
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tmp:
            tmp.writelines(file_list_entries)
            file_list = tmp.name
        
        # Calculate sequence length including TCN padding
        extended_sequence = self.sequence_length + 2 * self.dataset.tcn_padding
        device_id = torch.cuda.current_device() if torch.cuda.is_available() else 0
        
        # Set up distributed sharding
        if use_dali_sharding and torch.distributed.is_available() and torch.distributed.is_initialized():
            shard_id = torch.distributed.get_rank()
            num_shards = torch.distributed.get_world_size()
        else:
            shard_id = 0
            num_shards = 1
        
        step_size = self.sequence_length
        
        try:
            pipe = VideoPipeline(
                batch_size=self.batch_size,
                num_threads=self.num_threads,
                device_id=device_id,
                seed=epoch_seed,
                file_list=file_list,
                sequence_length=extended_sequence,
                step=step_size,
                resolution=self.dataset.resolution,
                random_shuffle=shuffle,
                shard_id=shard_id,
                num_shards=num_shards,
                pad_sequences=for_prediction,
                transform_mode=self.transform_mode,
                augmentations=self._augmentation_config if augment else None,
            )
            pipe.build()
        except RuntimeError as e:
            raise RuntimeError(f"DALI pipeline build failed: {e}")
        finally:
            os.unlink(file_list)

        if torch.cuda.is_available() and labels_2d is not None:
            labels_2d = labels_2d.to(device_id)

        video_lengths = get_video_lengths(
            video_list, 
            self.labels_dir,
            self.precomputed_lengths,
        )

        return DALIIterator(
            pipe,
            output_map=['frames', 'frame_idx', 'start_frame'],
            sequence_length=self.sequence_length,
            tcn_padding=self.dataset.tcn_padding,
            ignore_index=self.dataset.ignore_index,
            all_labels_2d=labels_2d,
            include_labels=include_labels,
            include_lengths=include_lengths,
            video_lengths=video_lengths,
            last_batch_policy=last_batch_policy,
        )

    def train_dataloader(self) -> DALIIterator:
        """Create training dataloader."""

        _check_dali_available()

        if hasattr(self, 'trainer') and self.trainer is not None:
            self.current_epoch = self.trainer.current_epoch
        
        # Compute max frames and load labels for training videos
        max_frames = get_max_frames(
            self.train_video_paths, 
            self.labels_dir,
            self.precomputed_lengths,
        )
        max_frames += self.sequence_length + 2 * self.dataset.tcn_padding
        
        labels_2d = load_labels_for_videos(
            self.train_video_paths, 
            self.labels_dir, 
            max_frames, 
            self.dataset.ignore_index,
        )
        
        return self._create_dataloader(
            self.train_video_paths,
            labels_2d,
            include_labels=True,
            include_lengths=False,
            shuffle=True,
            last_batch_policy_str='drop',
            use_dali_sharding=True,
            for_prediction=False,
            augment=True,
        )

    def val_dataloader(self) -> Union[DALIIterator, DataLoader]:
        """Create validation dataloader."""

        _check_dali_available()

        if not self.validation_enabled or not self.val_video_paths:
            return DataLoader(TensorDataset(torch.empty(0)), batch_size=1)
        
        max_frames = get_max_frames(
            self.val_video_paths, 
            self.labels_dir,
            self.precomputed_lengths,
        )
        max_frames += self.sequence_length + 2 * self.dataset.tcn_padding
        
        labels_2d = load_labels_for_videos(
            self.val_video_paths, 
            self.labels_dir, 
            max_frames, 
            self.dataset.ignore_index,
        )
        
        return self._create_dataloader(
            self.val_video_paths, 
            labels_2d,
            include_labels=True, 
            include_lengths=False, 
            shuffle=False,
            last_batch_policy_str='drop',
            use_dali_sharding=True, 
            for_prediction=False,
        )

    def predict_dataloader(self) -> Union[DALIIterator, DataLoader]:
        """Create prediction dataloader."""

        _check_dali_available()
        
        if self._predict_video_paths is None:
            self._predict_video_paths = self.dataset.video_paths
        
        if not self._predict_video_paths or len(self._predict_video_paths) == 0:
            return DataLoader(TensorDataset(torch.empty(0)), batch_size=1)
        
        dataloader = self._create_dataloader(
            self._predict_video_paths, 
            labels_2d=None,
            include_labels=False, 
            include_lengths=True,
            shuffle=False,
            last_batch_policy_str='partial',
            use_dali_sharding=False,
            for_prediction=True,
        )
        
        if dataloader is None:
            return DataLoader(TensorDataset(torch.empty(0)), batch_size=1)
        
        return dataloader

    def get_label_names(self) -> list[str]:
        """Get the list of class label names."""
        return self.dataset.get_label_names()

    @property
    def input_size(self) -> int:
        """Get the input feature dimension for the model."""
        return self.dataset.input_size

    @property
    def num_classes(self) -> int:
        """Get the number of action classes."""
        return self.dataset.num_classes

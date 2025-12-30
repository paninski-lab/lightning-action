"""Dataset for video action segmentation - video discovery and metadata.

This module provides the VideoDataset class which handles:
- Discovery and validation of video/label file pairs
- Class information extraction from label files
- Class weight computation for handling imbalanced datasets
- TCN padding calculation for temporal receptive fields

Note: This class does NOT handle chunking or frame-level indexing.
Actual video decoding and chunking is handled by NVIDIA DALI pipelines
in video_datamodule.py for GPU-accelerated performance.

Typical usage:
    dataset = VideoDataset(
        videos_dir='/path/to/videos',
        labels_dir='/path/to/labels',
        expt_ids=['video1', 'video2'],
    )
    
    # Access video list and class info
    print(f"Videos: {dataset.video_paths}")
    print(f"Class weights: {dataset.class_weights}")
"""

import os
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm
from typeguard import typechecked


class VideoDataset:
    """Dataset for video action segmentation - manages video metadata.
    
    This class discovers videos, validates label files exist, extracts
    class information, and computes class weights. It does NOT perform
    chunking - that is handled by DALI at runtime.
    
    Attributes:
        videos_dir: Directory containing .mp4 video files.
        labels_dir: Directory containing .npy label files (same basename as videos).
            Optional - can be None for prediction-only mode.
        video_paths: List of full paths to valid video files.
        label_paths: List of full paths to corresponding label files (may be empty).
        num_classes: Number of action classes detected from label files.
        label_names: List of class names (auto-generated if not provided).
        class_weights: Inverse-frequency weights for handling class imbalance.
        tcn_padding: Number of frames of context needed on each side of a chunk.
    """

    @typechecked
    def __init__(
        self,
        videos_dir: str,
        labels_dir: Optional[str] = None,
        sequence_length: int = 128,
        resolution: int = 224,
        expt_ids: Optional[list[str]] = None,
        input_size: int = 1536,
        num_lags: int = 0,
        ignore_index: int = -100,
        backbone: str = 'dtcn',
        num_layers: int = 2,
        require_labels: bool = True,
        num_classes: Optional[int] = None,
        label_names: Optional[list[str]] = None,
    ):
        """Initialize the VideoDataset.
        
        Args:
            videos_dir: Path to directory containing .mp4 video files.
            labels_dir: Path to directory containing .npy label files, or None
                for prediction-only mode.
                Labels should be either:
                - 1D array of class indices, shape (num_frames,)
                - 2D one-hot array, shape (num_frames, num_classes)
            sequence_length: Number of frames per chunk (used for TCN padding calc).
            resolution: Target frame resolution for preprocessing (square).
            expt_ids: Optional list of experiment IDs to include. If provided,
                only videos whose names start with one of these IDs are used.
            input_size: Feature dimension for the temporal backbone.
            num_lags: Number of temporal lags for the backbone (affects padding).
            ignore_index: Label value to ignore in loss computation (default -100).
            backbone: Type of temporal backbone ('dtcn', 'temporalmlp', 'rnn').
            num_layers: Number of layers in the temporal backbone.
            require_labels: If True (default), raise error if labels are missing.
                If False, allow videos without labels (for prediction).
            num_classes: Number of classes (required if labels_dir is None).
            label_names: Optional list of class names. If not provided and labels
                are available, will be auto-generated as class_0, class_1, etc.
        
        Raises:
            FileNotFoundError: If require_labels=True and any video is missing
                its corresponding .npy label file.
            ValueError: If labels_dir is None and num_classes is not provided.
        """
        # Store configuration
        self.videos_dir = videos_dir
        self.labels_dir = labels_dir
        self.sequence_length = sequence_length
        self.resolution = resolution
        self.input_size = input_size
        self.num_lags = num_lags
        self.ignore_index = ignore_index
        self.backbone = backbone
        self.num_layers = num_layers
        self.require_labels = require_labels
        
        # Calculate TCN padding based on backbone architecture
        self.tcn_padding = self._calculate_tcn_padding(backbone, num_layers, num_lags)
        
        # Discover videos and validate label files
        self.video_paths: list[str] = []
        self.label_paths: list[str] = []
        self.label_names: list[str] = label_names if label_names is not None else []
        self.num_classes: int = num_classes if num_classes is not None else 0
        
        # Set default label names if num_classes provided but label_names not
        if num_classes is not None and num_classes > 0 and not self.label_names:
            self.label_names = [f'class_{i}' for i in range(num_classes)]
        
        self._discover_videos(expt_ids)
        
        # Validate we have num_classes if no labels
        if self.labels_dir is None and self.num_classes == 0:
            raise ValueError(
                "num_classes must be provided when labels_dir is None"
            )
        
        # Compute class weights for handling imbalanced datasets
        if self.labels_dir is not None and len(self.label_paths) > 0:
            self.class_weights = self._compute_class_weights()
        else:
            # Default uniform weights when no labels available
            self.class_weights = [1.0] * self.num_classes

    def _discover_videos(self, expt_ids: Optional[list[str]] = None) -> None:
        """Discover videos and optionally validate corresponding label files exist.
        
        Populates video_paths, label_paths, and extracts class information.
        
        Args:
            expt_ids: Optional list of experiment IDs to filter by.
        
        Raises:
            FileNotFoundError: If require_labels=True and any video is missing 
                its label file.
        """
        # Find all MP4 files
        all_videos = [f for f in os.listdir(self.videos_dir) if f.endswith('.mp4')]
        
        # Filter by experiment IDs if specified
        if expt_ids:
            all_videos = [
                v for v in all_videos 
                if any(v.startswith(eid) for eid in expt_ids)
            ]
        
        missing_labels = []
        
        for video_name in tqdm(all_videos, desc="Discovering videos"):
            video_path = os.path.join(self.videos_dir, video_name)
            
            # Check for labels if labels_dir is provided
            if self.labels_dir is not None:
                label_path = os.path.join(self.labels_dir, video_name.replace('.mp4', '.npy'))
                
                if not os.path.exists(label_path):
                    if self.require_labels:
                        missing_labels.append(video_name)
                        continue
                    else:
                        # Add video without label
                        self.video_paths.append(video_path)
                        continue
                
                # Load labels to extract class information (only need to do once)
                if self.num_classes == 0:
                    labels = np.load(label_path)
                    
                    if labels.ndim > 1 and labels.shape[1] > 1:
                        # One-hot encoded: shape (num_frames, num_classes)
                        self.num_classes = labels.shape[1]
                    else:
                        # Class indices: shape (num_frames,)
                        if labels.ndim > 1:
                            labels = labels.squeeze()
                        unique_labels = np.unique(labels[labels >= 0])
                        self.num_classes = int(max(unique_labels) + 1) if unique_labels.size > 0 else 1
                    
                    # Only set default label names if not already provided
                    if not self.label_names:
                        self.label_names = [f'class_{i}' for i in range(self.num_classes)]
                
                self.video_paths.append(video_path)
                self.label_paths.append(label_path)
            else:
                # No labels_dir - just add the video
                self.video_paths.append(video_path)
        
        if missing_labels and self.require_labels:
            raise FileNotFoundError(
                f"Missing .npy label files for {len(missing_labels)} videos: "
                f"{missing_labels[:5]}{'...' if len(missing_labels) > 5 else ''}"
            )

    def _compute_class_weights(self) -> list[float]:
        """Compute inverse frequency class weights from all labels.
        
        Class weights help handle imbalanced datasets by giving higher weight
        to underrepresented classes in the loss function.
        
        Uses square root dampening to avoid over-weighting very rare classes.
        Classes with zero counts get a weight of 0.0.
        
        Returns:
            List of class weights, one per class.
        """
        from lightning_action.utils.class_weights import (
            compute_class_weights,
            collect_labels_from_files,
        )
        
        if not self.label_paths:
            return [1.0] * self.num_classes
        
        # Collect all labels from files
        all_labels, inferred_num_classes = collect_labels_from_files(
            self.label_paths, 
            ignore_index=self.ignore_index,
        )
        
        # Use inferred num_classes if we don't have one yet
        num_classes = self.num_classes if self.num_classes > 0 else inferred_num_classes
        
        # Compute weights with sqrt dampening (video pipeline default)
        return compute_class_weights(
            labels=all_labels,
            num_classes=num_classes,
            ignore_index=self.ignore_index,
            sqrt_dampening=True,
        )

    def _calculate_tcn_padding(
        self, 
        backbone: str = 'dtcn', 
        num_layers: int = 4, 
        num_lags: int = 2,
    ) -> int:
        """Calculate padding needed for temporal backbone receptive field.
        
        Different backbone architectures have different receptive fields:
        - DilatedTCN: Exponentially growing dilation pattern
        - TemporalMLP: Simple fixed lag
        - RNN: No padding needed (unless bidirectional)
        
        Args:
            backbone: Type of backbone ('dtcn', 'dilatedtcn', 'temporalmlp', 'rnn').
            num_layers: Number of layers in the backbone.
            num_lags: Number of lags/kernel size for temporal convolutions.
        
        Returns:
            Number of frames of padding needed on each side of a chunk.
        """
        if backbone.lower() in ['dtcn', 'dilatedtcn']:
            # DilatedTCN uses exponentially increasing dilations: 1, 2, 4, 8, ...
            # Total receptive field = sum of per-layer receptive fields
            total_pad = sum([2 * (2**n) * num_lags for n in range(num_layers)])
            return total_pad
        elif backbone.lower() == 'temporalmlp':
            return num_lags
        else:
            # RNN and other architectures: no padding needed
            return 0

    def __len__(self) -> int:
        """Return the number of videos in the dataset."""
        return len(self.video_paths)

    def get_label_names(self) -> list[str]:
        """Return a copy of the class label names."""
        return self.label_names.copy()
    
    def get_video_length(self, video_idx: int) -> int:
        """Get the frame count for a specific video.
        
        Args:
            video_idx: Index into video_paths.
        
        Returns:
            Number of frames in the video.
        """
        # Try to get from label file first
        if video_idx < len(self.label_paths) and self.label_paths[video_idx]:
            labels = np.load(self.label_paths[video_idx], mmap_mode='r')
            return len(labels)
        
        # Fall back to OpenCV
        video_path = self.video_paths[video_idx]
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            return frame_count
        
        return 0

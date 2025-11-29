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
        video_paths: List of full paths to valid video files.
        label_paths: List of full paths to corresponding label files.
        num_classes: Number of action classes detected from label files.
        label_names: List of class names (auto-generated if not provided).
        class_weights: Inverse-frequency weights for handling class imbalance.
        tcn_padding: Number of frames of context needed on each side of a chunk.
    """

    @typechecked
    def __init__(
        self,
        videos_dir: str,
        labels_dir: str,
        chunk_size: int = 128,
        resolution: int = 224,
        expt_ids: Optional[list[str]] = None,
        input_size: int = 1536,
        num_lags: int = 0,
        ignore_index: int = -100,
        num_threads: int = 2,
        backbone: str = 'dtcn',
        num_layers: int = 4,
    ):
        """Initialize the VideoDataset.
        
        Args:
            videos_dir: Path to directory containing .mp4 video files.
            labels_dir: Path to directory containing .npy label files.
                Labels should be either:
                - 1D array of class indices, shape (num_frames,)
                - 2D one-hot array, shape (num_frames, num_classes)
            chunk_size: Number of frames per chunk (used for TCN padding calc).
            resolution: Target frame resolution for preprocessing (square).
            expt_ids: Optional list of experiment IDs to include. If provided,
                only videos whose names start with one of these IDs are used.
            input_size: Feature dimension for the temporal backbone.
            num_lags: Number of temporal lags for the backbone (affects padding).
            ignore_index: Label value to ignore in loss computation (default -100).
            num_threads: Number of threads for DALI video decoding.
            backbone: Type of temporal backbone ('dtcn', 'temporalmlp', 'rnn').
            num_layers: Number of layers in the temporal backbone.
        
        Raises:
            FileNotFoundError: If any video is missing its corresponding .npy label file.
        """
        # Store configuration
        self.videos_dir = videos_dir
        self.labels_dir = labels_dir
        self.chunk_size = chunk_size
        self.resolution = resolution
        self.input_size = input_size
        self.num_lags = num_lags
        self.ignore_index = ignore_index
        self.num_threads = num_threads
        self.backbone = backbone
        self.num_layers = num_layers
        
        # Calculate TCN padding based on backbone architecture
        self.tcn_padding = self._calculate_tcn_padding(backbone, num_layers, num_lags)
        
        # Discover videos and validate label files
        self.video_paths: list[str] = []
        self.label_paths: list[str] = []
        self.label_names: list[str] = []
        self.num_classes: int = 0
        
        self._discover_videos(expt_ids)
        
        # Compute class weights for handling imbalanced datasets
        self.class_weights = self._compute_class_weights()

    def _discover_videos(self, expt_ids: Optional[list[str]] = None) -> None:
        """Discover videos and validate corresponding label files exist.
        
        Populates video_paths, label_paths, and extracts class information.
        
        Args:
            expt_ids: Optional list of experiment IDs to filter by.
        
        Raises:
            FileNotFoundError: If any video is missing its label file.
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
            label_path = os.path.join(self.labels_dir, video_name.replace('.mp4', '.npy'))
            
            # Check label file exists
            if not os.path.exists(label_path):
                missing_labels.append(video_name)
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
                
                self.label_names = [f'class_{i}' for i in range(self.num_classes)]
            
            self.video_paths.append(video_path)
            self.label_paths.append(label_path)
        
        if missing_labels:
            raise FileNotFoundError(
                f"Missing .npy label files for {len(missing_labels)} videos: "
                f"{missing_labels[:5]}{'...' if len(missing_labels) > 5 else ''}"
            )

    def _compute_class_weights(self) -> list[float]:
        """Compute inverse frequency class weights from all labels.
        
        Class weights help handle imbalanced datasets by giving higher weight
        to underrepresented classes in the loss function.
        
        The weight for class c is: sqrt(max_count / count_c)
        
        This uses square root dampening to avoid over-weighting very rare classes.
        Classes with zero counts get a weight of 0.0.
        
        Returns:
            List of class weights, one per class.
        """
        counts = np.zeros(self.num_classes)
        
        for label_path in tqdm(self.label_paths, desc="Computing class weights"):
            labels = np.load(label_path)
            
            # Handle one-hot encoded labels
            if labels.ndim > 1 and labels.shape[1] > 1:
                labels = np.argmax(labels, axis=1)
            elif labels.ndim > 1:
                labels = labels.squeeze()
            
            # Count only valid (non-ignored) labels
            valid_labels = labels[labels != self.ignore_index]
            unique, label_counts = np.unique(valid_labels, return_counts=True)
            
            for cls, count in zip(unique, label_counts):
                if 0 <= cls < self.num_classes:
                    counts[cls] += count
        
        # Handle edge case of empty dataset
        if np.sum(counts) == 0:
            return [1.0] * self.num_classes
        
        # Compute inverse frequency weights with square root dampening
        max_count = np.max(counts)
        weights = (max_count / (counts + 1e-10)) ** 0.5
        
        # Zero weight for classes with no samples
        weights[counts == 0] = 0.0
        
        return weights.tolist()

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
        labels = np.load(self.label_paths[video_idx], mmap_mode='r')
        return len(labels)

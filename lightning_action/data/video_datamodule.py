"""Lightning DataModule for video action segmentation datasets.

This module provides a PyTorch Lightning DataModule that wraps the VideoDataset
for easy integration with Lightning training workflows.
"""
import logging
from typing import Any

import lightning as pl
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from typeguard import typechecked

from lightning_action.data.video_dataset import VideoDataset
from lightning_action.data.utils import split_sizes_from_probabilities

logger = logging.getLogger(__name__)

class VideoDataModule(pl.LightningDataModule):
    """Lightning DataModule for video action segmentation tasks.
    
    This DataModule handles loading, splitting, and serving video data
    for training action segmentation models. It wraps the VideoDataset
    and provides train/validation DataLoaders.
    """

    @typechecked
    def __init__(
        self,
        data_config: dict[str, Any],
        sequence_length: int = 128,
        batch_size: int = 1,
        num_workers: int = 0,
        train_probability: float = 0.95,
        val_probability: float | None = None,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        seed: int = 42,
    ):
        """Initialize VideoDataModule.
        
        Args:
            data_config: configuration dictionary with keys:
                - 'videos_dir': directory with MP4 videos
                - 'labels_dir': directory with NumPy labels
                - 'expt_ids': list of dataset identifiers
                - 'resolution': output frame resolution (optional)
                - 'num_lags': number of context frames on each side
            sequence_length: length of each video chunk for predictions
            batch_size: batch size for DataLoaders
            num_workers: number of worker processes for data loading
            train_probability: fraction of data used for training
            val_probability: fraction of data used for validation (defaults to 1-train_probability)
            pin_memory: whether to use pinned memory for faster GPU transfer
            persistent_workers: whether to keep workers alive between epochs
            seed: random seed for splitting
            
        Raises:
            ValueError: if data_config is missing required keys
        """
        super().__init__()
        
        # validate data config
        required_keys = ['videos_dir', 'labels_dir', 'expt_ids']
        if not all(key in data_config for key in required_keys):
            raise ValueError(f'data_config must contain keys: {required_keys}')
        
        # store configuration
        self.data_config = data_config
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_probability = train_probability
        self.val_probability = val_probability or (1 - train_probability)
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.seed = seed

        # create full dataset
        logger.info('Creating VideoDataset')
        self.dataset = VideoDataset(
            videos_dir=self.data_config['videos_dir'],
            labels_dir=self.data_config['labels_dir'],
            chunk_size=self.sequence_length,
            resolution=self.data_config.get('resolution', 224),
            expt_ids=self.data_config['expt_ids'],
            input_size=self.data_config.get('input_size', 1536),
            num_lags=self.data_config.get('num_lags', 0),
            ignore_index=self.data_config.get('ignore_index', -100),
        )

        logger.info(f'Created dataset with {len(self.dataset)} chunks')

        # split datasets
        self.dataset_train = None
        self.dataset_val = None

    def setup(self, stage: str | None = None):
        """Set up datasets for training and validation.
        
        Args:
            stage: training stage ('fit', 'validate', 'test', or None)
        """

        if stage in ['test', 'predict']:
            # no test data support as requested
            return
            
        if stage in [None, 'fit', 'validate']:
            if self.dataset_train is None or self.dataset_val is None:
                total_size = len(self.dataset)
                train_size, val_size = split_sizes_from_probabilities(
                    total_size,
                    self.train_probability,
                    self.val_probability,
                )
                
                logger.info(f'Splitting dataset: {train_size} train, {val_size} val chunks')
                np.random.seed(self.seed)
                self.dataset_train, self.dataset_val = random_split(
                    self.dataset,
                    [train_size, val_size],
                )

    def train_dataloader(self) -> DataLoader:
        """Create training DataLoader.
        
        Returns:
            DataLoader for training data
        """
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation DataLoader.
        
        Returns:
            DataLoader for validation data
        """
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=self._collate_fn,
        )

    def predict_dataloader(self) -> DataLoader:
        """Create prediction DataLoader.
        
        Returns:
            DataLoader for prediction
        """
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=self._collate_fn,
        )

    @typechecked
    def _collate_fn(self, batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
        """Custom collate function to stack batches.
        
        Args:
            batch: list of (frames, labels) tuples from VideoDataset
            
        Returns:
            tuple of stacked frames and labels
        """
        frames, labels = zip(*batch)
        frames = torch.stack(frames)
        labels = torch.stack(labels)
        return frames, labels

    def get_label_names(self) -> list[str]:
        """Get label names from the dataset.
        
        Returns:
            list of label names
        """
        return self.dataset.get_label_names()

    @property
    def input_size(self) -> int:
        """Get input size from the dataset.
        
        Returns:
            dimensionality of input features
        """
        return self.dataset.input_size

    @property
    def num_classes(self) -> int:
        """Get number of classes from the dataset.
        
        Returns:
            number of label classes
        """
        return self.dataset.num_classes

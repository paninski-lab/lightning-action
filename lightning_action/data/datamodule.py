"""Lightning DataModule for action segmentation datasets.

This module provides a PyTorch Lightning DataModule that wraps the FeatureDataset
for easy integration with Lightning training workflows.
"""

import logging
from typing import Any

import lightning as pl
import numpy as np
from torch.utils.data import DataLoader, random_split
from typeguard import typechecked

from lightning_action.data.datasets import FeatureDataset
from lightning_action.data.utils import split_sizes_from_probabilities

logger = logging.getLogger(__name__)


class DataModule(pl.LightningDataModule):
    """Lightning DataModule for action segmentation tasks.
    
    This DataModule handles loading, splitting, and serving behavioral data
    for training action segmentation models. It wraps the FeatureDataset
    and provides train/validation DataLoaders.
    """

    @typechecked
    def __init__(
        self,
        data_config: dict[str, Any],
        sequence_length: int = 500,
        sequence_pad: int = 0,
        batch_size: int = 8,
        num_workers: int = 4,
        train_probability: float = 0.9,
        val_probability: float | None = None,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        seed: int = 42,
    ):
        """Initialize DataModule.
        
        Args:
            data_config: configuration dictionary with keys:
                - 'ids': list of dataset identifiers
                - 'signals': list of signal lists for each dataset
                - 'transforms': list of transform lists for each dataset
                - 'paths': list of file path lists for each dataset
            sequence_length: length of each sequence
            sequence_pad: additional padding for sequences
            batch_size: batch size for DataLoaders
            num_workers: number of worker processes for data loading
            train_probability: fraction of data used for training
            val_probability: fraction of data used for validation (defaults to 1-train_probability)
            pin_memory: whether to use pinned memory for faster GPU transfer
            persistent_workers: whether to keep workers alive between epochs
            seed: random seed for weight initialization
            
        Raises:
            ValueError: if data_config is missing required keys
        """
        super().__init__()
        
        # validate data config
        required_keys = ['ids', 'signals', 'transforms', 'paths']
        if not all(key in data_config for key in required_keys):
            raise ValueError(f'data_config must contain keys: {required_keys}')
        
        # store configuration
        self.data_config = data_config
        self.sequence_length = sequence_length
        self.sequence_pad = sequence_pad
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_probability = train_probability
        self.val_probability = val_probability
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.seed = seed

        # create full dataset
        logger.info('Creating FeatureDataset')
        self.dataset = FeatureDataset(
            ids=self.data_config['ids'],
            signals=self.data_config['signals'],
            transforms=self.data_config['transforms'],
            paths=self.data_config['paths'],
            sequence_length=self.sequence_length,
            sequence_pad=self.sequence_pad,
        )

        logger.info(f'Created dataset with {len(self.dataset)} sequences')
        logger.info(f'Input size: {self.dataset.input_size}')
        logger.info(f'Feature names: {self.dataset.get_feature_names()[:5]}...')  # show first 5
        logger.info(f'Label names: {self.dataset.get_label_names()}')

        # split datasets
        self.dataset_train = None
        self.dataset_val = None
        self.setup()

    def setup(self, stage: str | None = None):
        """Set up datasets for training and validation.
        
        Args:
            stage: training stage ('fit', 'validate', 'test', or None)
        """
        if stage in ['test', 'predict']:
            # no test data support as requested
            return
        
        # split into train/val if not already split
        if self.dataset_train is None or self.dataset_val is None:
            total_size = len(self.dataset)
            train_size, val_size = split_sizes_from_probabilities(
                total_size,
                self.train_probability,
                self.val_probability,
            )
            
            logger.info(f'Splitting dataset: {train_size} train, {val_size} val sequences')
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
        )

    def predict_dataloader(self) -> DataLoader:
        """Create validation DataLoader.

        Returns:
            DataLoader for prediction
        """
        return DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def get_feature_names(self) -> list[str]:
        """Get feature names from the dataset.
        
        Returns:
            list of feature names
            
        Raises:
            RuntimeError: if dataset has not been set up yet
        """
        return self.dataset.get_feature_names()

    def get_label_names(self) -> list[str]:
        """Get label names from the dataset.
        
        Returns:
            list of label names
            
        Raises:
            RuntimeError: if dataset has not been set up yet
        """
        return self.dataset.get_label_names()

    def get_dataset_ids(self) -> list[str]:
        """Get dataset IDs from the dataset.
        
        Returns:
            list of dataset identifiers
            
        Raises:
            RuntimeError: if dataset has not been set up yet
        """
        return self.dataset.get_dataset_ids()

    @property
    def input_size(self) -> int:
        """Get input size from the dataset.
        
        Returns:
            dimensionality of input features
            
        Raises:
            RuntimeError: if dataset has not been set up yet
        """
        return self.dataset.input_size

    @property
    def num_classes(self) -> int:
        """Get number of classes from the dataset.
        
        Returns:
            number of label classes
            
        Raises:
            RuntimeError: if dataset has not been set up yet
        """
        return len(self.dataset.get_label_names())

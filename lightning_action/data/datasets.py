"""Dataset classes for action segmentation.

This module contains PyTorch Dataset classes adapted from daart for use with
PyTorch Lightning. The main class FeatureDataset handles loading and processing
of behavioral data including markers and labels.
"""

import logging
from collections import OrderedDict
from typing import Any

import numpy as np
import torch
from jaxtyping import Float, Int
from torch.utils.data import Dataset
from typeguard import typechecked

from .utils import compute_sequences, load_feature_csv, load_label_csv, load_marker_csv

logger = logging.getLogger(__name__)


class FeatureDataset(Dataset):
    """Dataset for loading and processing behavioral features and labels.
    
    This class handles multiple datasets with markers and labels, applying transforms
    and creating sequences for action segmentation tasks. Based on daart's SingleDataset
    but simplified for Lightning integration.
    """

    @typechecked
    def __init__(
        self,
        ids: list[str],
        signals: list[list[str]],
        transforms: list[list[Any]],
        paths: list[list[str | None]],
        sequence_length: int = 500,
        sequence_pad: int = 0,
    ):
        """Initialize FeatureDataset.
        
        Args:
            ids: list of dataset identifiers
            signals: list of signal lists (e.g., [['markers', 'labels'], ...])
            transforms: list of transform lists (one per signal per dataset)
            paths: list of file path lists (one per signal per dataset)
            sequence_length: length of each sequence
            sequence_pad: additional padding for sequences
            
        Raises:
            ValueError: if input parameters are inconsistent
        """
        self.ids = ids
        self.signals = signals
        self.transforms = transforms
        self.paths = paths
        self.sequence_length = sequence_length
        self.sequence_pad = sequence_pad
        
        # validate input consistency
        self._validate_inputs()
        
        # initialize storage
        self.sequences = []  # flat list of all sequences
        self.sequence_info = []  # metadata for each sequence (dataset_id, idx)
        
        # data properties (set during loading)
        self.input_size = 0
        self.data_lengths = []
        self.feature_names = []
        self.label_names = []
        
        # load and process all data
        self._load_all_data()

    def _validate_inputs(self):
        """Validate that input lists have consistent lengths."""
        n_datasets = len(self.ids)
        if not all(len(lst) == n_datasets for lst in [self.signals, self.transforms, self.paths]):
            raise ValueError('ids, signals, transforms, and paths must have same length')
        
        for i in range(n_datasets):
            n_signals = len(self.signals[i])
            if not all(len(lst) == n_signals for lst in [self.transforms[i], self.paths[i]]):
                raise ValueError(
                    f'Dataset {i}: signals, transforms, and paths must have same length'
                )

    def _load_all_data(self):
        """Load and process data from all datasets."""
        for dataset_idx, dataset_id in enumerate(self.ids):
            logger.debug(f'Loading dataset {dataset_id}')
            
            # load data for this dataset
            dataset_data, dataset_length = self._load_dataset_data(dataset_idx)
            
            # create sequences for this dataset
            dataset_sequences = self._create_dataset_sequences(dataset_data, dataset_id)
            
            # add to global sequence list
            for seq_idx, sequence in enumerate(dataset_sequences):
                self.sequences.append(sequence)
                self.sequence_info.append({
                    'dataset_id': dataset_id,
                    'dataset_idx': dataset_idx,
                    'sequence_idx': seq_idx
                })
                
            # record length of original inputs
            self.data_lengths.append(dataset_length)

    def _load_dataset_data(self, dataset_idx: int) -> OrderedDict:
        """Load raw data for a single dataset.
        
        Args:
            dataset_idx: index of dataset to load
            
        Returns:
            OrderedDict mapping signal names to loaded data arrays
            
        Raises:
            ValueError: if data loading fails or data lengths are inconsistent
        """
        dataset_signals = self.signals[dataset_idx]
        dataset_transforms = self.transforms[dataset_idx]
        dataset_paths = self.paths[dataset_idx]
        
        data = OrderedDict()
        data_lengths = []
        
        for signal_idx, signal in enumerate(dataset_signals):
            signal_path = dataset_paths[signal_idx]
            signal_transform = dataset_transforms[signal_idx]
            
            # skip if no path provided
            if signal_path is None:
                continue
                
            logger.debug(f'Loading {signal} from {signal_path}')
            
            # load data based on signal type
            if signal == 'markers':
                data_curr = self._load_markers(signal_path)
            elif signal == 'features':
                data_curr = self._load_features(signal_path)
            elif signal == 'labels':
                data_curr = self._load_labels(signal_path)
            else:
                try:
                    data_curr = self._load_features(signal_path)
                except Exception:
                    raise ValueError(f'Unknown signal type: {signal}')

            # apply transforms
            if signal_transform is not None:
                logger.debug(f'Applying transform to {signal}')
                for transform in signal_transform:
                    data_curr = transform(data_curr)
            
            # store data and track length
            data[signal] = data_curr
            data_lengths.append(data_curr.shape[0])
        
        # validate all signals have same length
        if len(set(data_lengths)) > 1:
            raise ValueError(f'All signals must have same length, got {data_lengths}')
        
        return data, data_lengths[0]

    def _load_markers(
        self,
        file_path: str,
        include_likelihoods: bool = False,
    ) -> Float[np.ndarray, 'n_frames n_features']:
        """Load marker data from CSV file.
        
        Args:
            file_path: path to marker CSV file
            include_likelihoods: True to append likelihoods to x/y coordinates
            
        Returns:
            marker data array
        """

        x_coords, y_coords, likelihoods, marker_names = load_marker_csv(file_path)

        # store feature names if first dataset
        if not self.feature_names:
            # create feature names: marker_x, marker_y, marker_likelihood
            self.feature_names = []
            for marker in marker_names:
                self.feature_names.extend([f'{marker}_x', f'{marker}_y'])
                if include_likelihoods:
                    self.feature_names.extend([f'{marker}_likelihood'])

        # concatenate coordinates and likelihoods
        if include_likelihoods:
            data = np.concatenate([x_coords, y_coords, likelihoods], axis=1)
        else:
            data = np.concatenate([x_coords, y_coords], axis=1)
        
        # store input size if first dataset
        if self.input_size == 0:
            self.input_size = data.shape[1]
        
        return data.astype(np.float32)

    def _load_features(self, file_path: str) -> Float[np.ndarray, 'n_frames n_features']:
        """Load feature data from CSV file.

        Args:
            file_path: path to marker CSV file

        Returns:
            marker data array
        """
        data, feature_names = load_feature_csv(file_path)

        # store feature names if first dataset
        if not self.feature_names:
            self.feature_names = feature_names

        # store input size if first dataset
        if self.input_size == 0:
            self.input_size = data.shape[1]

        return data.astype(np.float32)

    def _load_labels(self, file_path: str) -> Int[np.ndarray, 'n_frames n_classes']:
        """Load label data from CSV file.
        
        Args:
            file_path: path to label CSV file
            
        Returns:
            label data array (one-hot encoded)
        """
        labels, class_names = load_label_csv(file_path)
        
        # store class names if first dataset
        if not self.label_names:
            self.label_names = class_names
        
        return labels.astype(np.float32)

    def _create_dataset_sequences(
        self,
        dataset_data: OrderedDict,
        dataset_id: str
    ) -> list[dict[str, np.ndarray]]:
        """Create sequences from dataset data.
        
        Args:
            dataset_data: loaded data for the dataset
            dataset_id: identifier for the dataset
            
        Returns:
            list of sequence dictionaries
        """
        sequences = []
        
        # convert each signal to sequences
        signal_sequences = {}
        n_sequences = None
        
        for signal, data in dataset_data.items():
            # create sequences for this signal
            seq_data = compute_sequences(data, self.sequence_length, self.sequence_pad)

            # standardize input name
            if signal == 'labels':
                signal_ = signal
            else:
                signal_ = 'input'

            signal_sequences[signal_] = seq_data
            
            # track number of sequences (should be same for all signals)
            if n_sequences is None:
                n_sequences = len(seq_data)
            elif n_sequences != len(seq_data):
                raise ValueError(f'Sequence count mismatch for signal {signal}')
        
        # create sequence dictionaries
        for seq_idx in range(n_sequences):
            sequence = {}
            for signal, seq_data in signal_sequences.items():
                sequence[signal] = seq_data[seq_idx]
            
            # add metadata
            sequence['dataset_id'] = dataset_id
            sequence['batch_idx'] = seq_idx
            
            sequences.append(sequence)
        
        logger.debug(f'Created {len(sequences)} sequences for dataset {dataset_id}')
        return sequences

    def __len__(self) -> int:
        """Return total number of sequences across all datasets."""
        return len(self.sequences)

    def __getitem__(
        self,
        idx: int,
        as_numpy: bool = False,
    ) -> dict[str, torch.Tensor | np.ndarray]:
        """Get a single sequence by index.
        
        Args:
            idx: sequence index
            as_numpy: True to return np arrays, False to return torch arrays

        Returns:
            dictionary containing sequence data for all signals
        """
        if idx >= len(self.sequences):
            raise IndexError(f'Index {idx} out of range for dataset of size {len(self.sequences)}')
        
        sequence = self.sequences[idx].copy()
        
        # convert to tensors if requested
        if not as_numpy:
            for signal, data in sequence.items():
                if isinstance(data, np.ndarray):
                    tensor = torch.from_numpy(data)
                    sequence[signal] = tensor
        
        return sequence

    def get_sequence_info(self, idx: int) -> dict[str, Any]:
        """Get metadata for a sequence.
        
        Args:
            idx: sequence index
            
        Returns:
            dictionary with sequence metadata
        """
        if idx >= len(self.sequence_info):
            raise IndexError(
                f'Index {idx} out of range for dataset of size {len(self.sequence_info)}'
            )
        
        return self.sequence_info[idx].copy()

    def get_dataset_ids(self) -> list[str]:
        """Get list of all dataset IDs."""
        return self.ids.copy()

    def get_feature_names(self) -> list[str]:
        """Get list of feature names."""
        return self.feature_names.copy()

    def get_label_names(self) -> list[str]:
        """Get list of label/class names."""
        return self.label_names.copy()

"""Core data utilities for action segmentation.

This module contains functions for loading, processing, and splitting behavioral data
adapted from the daart package with modern type hints and Lightning compatibility.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from jaxtyping import Float, Int
from typeguard import typechecked

logger = logging.getLogger(__name__)


@typechecked
def compute_sequences(
    data: Float[np.ndarray, 'n_frames ...'] | list,
    sequence_length: int,
    sequence_pad: int = 0,
) -> list:
    """Convert continuous data into fixed-length sequences.
    
    Creates sliding windows over the data with the specified sequence length.
    Optionally pads sequences for models that need context (e.g., TCNs).
    
    Args:
        data: input data with shape (n_frames, ...)
        sequence_length: length of each sequence
        sequence_pad: additional padding for model context
        
    Returns:
        sequences with shape (n_sequences, sequence_length + sequence_pad, ...)
        
    Raises:
        ValueError: if sequence parameters are invalid
    """

    if isinstance(data, list):
        # assume data has already been batched
        return data

    if sequence_length <= 0:
        raise ValueError(f'sequence_length must be positive, got {sequence_length}')

    if sequence_pad < 0:
        raise ValueError(f'sequence_pad must be non-negative, got {sequence_pad}')

    if len(data.shape) == 2:
        batch_dims = (sequence_length + 2 * sequence_pad, data.shape[1])
    else:
        batch_dims = (sequence_length + 2 * sequence_pad,)

    n_batches = int(np.floor(data.shape[0] / sequence_length))
    batched_data = [np.zeros(batch_dims) for _ in range(n_batches)]
    for b in range(n_batches):
        idx_beg = b * sequence_length
        idx_end = (b + 1) * sequence_length
        if sequence_pad > 0:
            if idx_beg == 0:
                # initial vals are zeros; rest are real data
                batched_data[b][sequence_pad:] = data[idx_beg:idx_end + sequence_pad]
            elif (idx_end + sequence_pad) > data.shape[0]:
                batched_data[b][:-sequence_pad] = data[idx_beg - sequence_pad:idx_end]
            else:
                batched_data[b] = data[idx_beg - sequence_pad:idx_end + sequence_pad]
        else:
            batched_data[b] = data[idx_beg:idx_end]

    return batched_data


@typechecked
def compute_sequence_pad(model_type: str, **model_params: Any) -> int:
    """Compute required sequence padding based on model architecture.
    
    Different model types require different amounts of padding to maintain
    temporal context and handle boundary effects.
    
    Args:
        model_type: type of model ('temporal-mlp', 'tcn', 'dtcn', 'lstm', 'gru', etc.)
        **model_params: model-specific parameters
        
    Returns:
        required padding in number of timesteps
        
    Raises:
        ValueError: if model_type is not recognized
    """
    model_type = model_type.lower()
    
    if model_type in ['temporal-mlp', 'temporalmlp']:
        return model_params['num_lags']
    
    elif model_type == 'tcn':
        n_layers = model_params['num_layers']
        n_lags = model_params['num_lags']
        return (2 ** n_layers) * n_lags

    elif model_type == 'dtcn':
        # dilated TCN with more complex calculation
        # dilattion of each dilation block is 2 ** layer_num
        # 2 conv layers per dilation block
        return sum(
            [2 * (2 ** n) * model_params['num_lags'] for n in range(model_params['num_layers'])]
        )
    
    elif model_type in ['lstm', 'gru', 'rnn']:
        # fixed warmup period for recurrent models
        return 4
    
    else:
        raise ValueError(f'Unknown model type: {model_type}')


@typechecked
def load_marker_csv(file_path: str | Path) -> tuple[
    Float[np.ndarray, 'n_frames n_markers'],
    Float[np.ndarray, 'n_frames n_markers'],
    Float[np.ndarray, 'n_frames n_markers'],
    list[str],
]:
    """Load DLC-format marker data from CSV file.
    
    Handles the multi-level header structure typical of DLC output files.
    
    Args:
        file_path: path to CSV file with DLC marker data
        
    Returns:
        tuple containing:
        - x_coords: x coordinates for each marker and frame
        - y_coords: y coordinates for each marker and frame
        - likelihoods: confidence scores for each marker and frame
        - marker_names: list of marker names
        
    Raises:
        FileNotFoundError: if file does not exist
        ValueError: if file format is invalid
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f'File not found: {file_path}')
    
    try:
        # load with multi-level headers
        df = pd.read_csv(file_path, header=[0, 1, 2], index_col=0)
        
        # extract marker names from column structure
        marker_names = df.columns.get_level_values(1).unique().tolist()
        
        # extract coordinates and likelihoods
        x_coords = []
        y_coords = []
        likelihoods = []
        
        for marker in marker_names:
            x_coords.append(
                df.loc[:, (df.columns.get_level_values(1) == marker) &
                          (df.columns.get_level_values(2) == 'x')
                ].values.flatten())
            y_coords.append(
                df.loc[:, (df.columns.get_level_values(1) == marker) &
                          (df.columns.get_level_values(2) == 'y')
                ].values.flatten())
            likelihoods.append(
                df.loc[:, (df.columns.get_level_values(1) == marker) &
                          (df.columns.get_level_values(2) == 'likelihood')
                ].values.flatten())
        
        x_coords = np.column_stack(x_coords)
        y_coords = np.column_stack(y_coords)
        likelihoods = np.column_stack(likelihoods)
        
        return x_coords, y_coords, likelihoods, marker_names
        
    except Exception as e:
        raise ValueError(f'Error loading marker CSV file {file_path}: {e}')


@typechecked
def load_feature_csv(file_path: str | Path) -> tuple[
    Float[np.ndarray, 'n_frames n_features'],
    list[str],
]:
    """Load feature data from simple CSV file.
    
    Args:
        file_path: path to CSV file with feature data
        
    Returns:
        tuple containing:
        - features: feature values for each frame
        - feature_names: list of feature names
        
    Raises:
        FileNotFoundError: if file does not exist
        ValueError: if file format is invalid
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f'File not found: {file_path}')
    
    try:
        df = pd.read_csv(file_path, index_col=0)
        features = df.values.astype(np.float32)
        feature_names = df.columns.tolist()
        
        return features, feature_names
        
    except Exception as e:
        raise ValueError(f'Error loading feature CSV file {file_path}: {e}')


@typechecked
def load_label_csv(file_path: str | Path) -> tuple[
    Int[np.ndarray, 'n_frames n_classes'],
    list[str],
]:
    """Load behavioral labels from CSV file.
    
    Expects one-hot encoded labels that will be converted to integer indices.
    
    Args:
        file_path: path to CSV file with label data
        
    Returns:
        tuple containing:
        - labels: one-hot encoded labels for each frame
        - class_names: list of behavior class names
        
    Raises:
        FileNotFoundError: if file does not exist
        ValueError: if file format is invalid
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f'File not found: {file_path}')
    
    try:
        df = pd.read_csv(file_path, index_col=0)
        labels = df.values.astype(np.int32)
        class_names = df.columns.tolist()
        
        return labels, class_names
        
    except Exception as e:
        raise ValueError(f'Error loading label CSV file {file_path}: {e}')


@typechecked
def split_sizes_from_probabilities(
    total_number: int,
    train_probability: float,
    val_probability: float | None = None,
) -> list[int]:
    """Returns the number of examples for train, val and test given split probs.

    Args:
        total_number: total number of examples in dataset
        train_probability: fraction of examples used for training
        val_probability: fraction of examples used for validation

    Returns:
        list [num training examples, num validation examples]

    """

    if val_probability is None:
        val_probability = 1.0 - train_probability

    # probabilities should add to one
    assert train_probability + val_probability == 1.0, 'Split probabilities must add to 1'

    # compute numbers from probabilities
    train_number = int(np.ceil(train_probability * total_number))
    val_number = total_number - train_number

    # make sure that we have at least one validation sample
    if val_number == 0:
        train_number -= 1
        val_number += 1
        if train_number < 1:
            raise ValueError('Must have at least two sequences, one train and one validation')

    # assert that we're using all datapoints
    assert train_number + val_number == total_number

    return [train_number, val_number]

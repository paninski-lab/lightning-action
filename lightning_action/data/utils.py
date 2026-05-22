"""Core data utilities for action segmentation.

This module contains functions for loading, processing, and splitting behavioral data
adapted from the daart package with modern type hints and Lightning compatibility.
"""

import logging
from pathlib import Path
from typing import Any, Literal, get_args

import numpy as np
import pandas as pd
from jaxtyping import Float, Int

logger = logging.getLogger(__name__)

ModelType = Literal[
    'temporal-mlp', 'temporalmlp',
    'tcn',
    'dtcn', 'dilatedtcn',
    'lstm', 'gru', 'rnn',
]


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
    batched_data = [np.zeros(batch_dims, dtype=data.dtype) for _ in range(n_batches)]
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


def compute_sequence_pad(
    model_type: ModelType,
    default: int | None = None,
    **model_params: Any,
) -> int:
    """Compute required sequence padding based on model architecture.

    Different model types require different amounts of padding to maintain
    temporal context and handle boundary effects.

    Args:
        model_type: type of model ('temporal-mlp', 'tcn', 'dtcn', 'dilatedtcn',
            'lstm', 'gru', 'rnn', etc.)
        default: if provided, return this value for unknown model types instead
            of raising ValueError. Use default=0 to silently handle unknown types.
        **model_params: model-specific parameters:
            - num_lags: number of temporal lags (required for mlp, tcn, dtcn)
            - num_layers: number of layers (required for tcn, dtcn)

    Returns:
        required padding in number of timesteps

    Raises:
        ValueError: if model_type is not recognized and default is None

    Examples:
        # Dilated TCN with 4 layers and 2 lags
        pad = compute_sequence_pad('dtcn', num_layers=4, num_lags=2)

        # Unknown model type with default fallback
        pad = compute_sequence_pad('transformer', default=0, num_layers=4)
    """
    model_type = model_type.lower()

    if model_type in ['temporal-mlp', 'temporalmlp']:
        return model_params['num_lags']

    elif model_type == 'tcn':
        num_layers = model_params['num_layers']
        num_lags = model_params['num_lags']
        return (2 ** num_layers) * num_lags

    elif model_type in ['dtcn', 'dilatedtcn']:
        # dilated TCN with more complex calculation
        # dilation of each dilation block is 2 ** layer_num
        # 2 conv layers per dilation block
        return sum(
            [2 * (2 ** n) * model_params['num_lags'] for n in range(model_params['num_layers'])]
        )

    elif model_type in ['lstm', 'gru', 'rnn']:
        # fixed warmup period for recurrent models
        return 4

    else:
        if default is not None:
            return default
        raise ValueError(
            f'Unknown model type: {model_type}. '
            f'Valid values: {", ".join(get_args(ModelType))}'
        )


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
        raise ValueError(f'Error loading marker CSV file {file_path}: {e}') from e


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
        raise ValueError(f'Error loading feature CSV file {file_path}: {e}') from e


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
        raise ValueError(f'Error loading label CSV file {file_path}: {e}') from e


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

    # assert that we're using all datapoints
    assert train_number + val_number == total_number

    return [train_number, val_number]


def compute_class_weights(
    labels: np.ndarray,
    num_classes: int | None = None,
    ignore_index: int = -100,
    sqrt_dampening: bool = False,
) -> list[float]:
    """Compute inverse frequency class weights for imbalanced datasets.

    Computes weights inversely proportional to class frequency. The most
    frequent class gets weight close to 1.0, while less frequent classes
    get higher weights to compensate for their underrepresentation.

    This function provides a unified implementation for class weight
    computation, usable by both the CSV pipeline and video pipeline.

    Args:
        labels: 1D array of integer class labels. Can contain ignore_index
            values which will be excluded from counting.
        num_classes: Total number of classes. If None, inferred from the
            maximum label value + 1.
        ignore_index: Label value to exclude from weight computation
            (typically -100 for padding/unlabeled frames).
        sqrt_dampening: If True, apply square root dampening to weights
            to avoid over-weighting very rare classes. Formula becomes
            sqrt(max_count / count) instead of max_count / count.

    Returns:
        List of class weights, one per class. Classes with zero samples
        get weight 0.0.

    Example:
        # From label files (video pipeline)
        all_labels = np.concatenate([np.load(f) for f in label_files])
        weights = compute_class_weights(all_labels, num_classes=3, sqrt_dampening=True)

        # From datamodule (CSV pipeline)
        all_labels = np.concatenate([batch['labels'].flatten() for batch in dataset])
        weights = compute_class_weights(all_labels, num_classes=4)
    """
    # Flatten in case input is not 1D
    labels = labels.flatten()

    # Filter out ignored labels
    valid_labels = labels[labels != ignore_index]

    if len(valid_labels) == 0:
        logger.warning("No valid labels found, returning uniform weights")
        if num_classes is None:
            num_classes = 1
        return [1.0] * num_classes

    # Count occurrences of each class
    unique_classes, counts = np.unique(valid_labels, return_counts=True)

    # Determine number of classes
    if num_classes is None:
        num_classes = int(max(unique_classes)) + 1

    # Build counts array for all classes
    totals = np.zeros(num_classes, dtype=np.float64)
    for cls, count in zip(unique_classes, counts, strict=True):
        cls_int = int(cls)
        if 0 <= cls_int < num_classes:
            totals[cls_int] = count

    # Handle edge case of all zeros
    max_count = np.max(totals)
    if max_count == 0:
        logger.warning("No labeled examples found, returning uniform weights")
        return [1.0] * num_classes

    # Compute inverse frequency weights
    weights = max_count / (totals + 1e-10)

    # Apply optional sqrt dampening
    if sqrt_dampening:
        weights = np.sqrt(weights)

    # Zero weight for classes with no samples
    weights[totals == 0] = 0.0

    logger.info(f"Class counts: {totals.astype(int).tolist()}")
    logger.info(f"Class weights: {[f'{w:.3f}' for w in weights]}")

    return weights.tolist()


def collect_labels_from_files(
    label_paths: list[str],
    ignore_index: int = -100,
    show_progress: bool = True,
) -> tuple[np.ndarray, int]:
    """Collect and concatenate labels from multiple .npy files.

    Helper function that loads labels from numpy files and returns
    a flattened array suitable for compute_class_weights().

    Handles both:
    - 1D arrays of class indices, shape (num_frames,)
    - 2D one-hot arrays, shape (num_frames, num_classes)

    Args:
        label_paths: List of paths to .npy label files.
        ignore_index: Value to use for ignored/padding frames.
        show_progress: Whether to show a progress bar.

    Returns:
        Tuple of:
        - Flattened 1D array of all labels
        - Number of classes (inferred from labels)

    Example:
        labels, num_classes = collect_labels_from_files(dataset.label_paths)
        weights = compute_class_weights(labels, num_classes, sqrt_dampening=True)
    """
    from tqdm import tqdm

    all_labels = []
    num_classes = 0

    iterator = tqdm(label_paths, desc="Collecting labels") if show_progress else label_paths

    for label_path in iterator:
        labels = np.load(label_path)

        # Handle one-hot encoded labels
        if labels.ndim > 1 and labels.shape[1] > 1:
            num_classes = max(num_classes, labels.shape[1])
            labels = np.argmax(labels, axis=1)
        elif labels.ndim > 1:
            labels = labels.squeeze()

        # Update num_classes from max label value
        valid = labels[labels != ignore_index]
        if len(valid) > 0:
            num_classes = max(num_classes, int(np.max(valid)) + 1)

        all_labels.append(labels.flatten())

    if not all_labels:
        return np.array([], dtype=np.int64), num_classes

    return np.concatenate(all_labels), num_classes


def collect_labels_from_datamodule(
    dataset: Any,
    ignore_index: int = -100,
    show_progress: bool = True,
) -> tuple[np.ndarray, int]:
    """Collect and concatenate labels from a PyTorch dataset.

    Helper function that iterates over a dataset and extracts labels,
    returning a flattened array suitable for compute_class_weights().

    Expects dataset items to be dicts with a 'labels' key.

    Handles both:
    - 1D arrays of class indices
    - 2D one-hot arrays

    Args:
        dataset: PyTorch dataset where __getitem__ returns a dict
            with 'labels' key.
        ignore_index: Value used for ignored/padding frames.
        show_progress: Whether to show a progress bar.

    Returns:
        Tuple of:
        - Flattened 1D array of all labels
        - Number of classes (inferred from labels)

    Example:
        datamodule.setup('fit')
        labels, num_classes = collect_labels_from_datamodule(datamodule.dataset_train)
        weights = compute_class_weights(labels, num_classes)
    """
    import torch
    from tqdm import tqdm

    all_labels = []
    num_classes = 0

    indices = range(len(dataset))
    iterator = tqdm(indices, desc="Collecting labels") if show_progress else indices

    for i in iterator:
        batch = dataset[i]

        if 'labels' not in batch:
            continue

        labels = batch['labels']

        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()

        # Handle one-hot encoded labels
        if labels.ndim > 1 and labels.shape[-1] > 1:
            num_classes = max(num_classes, labels.shape[-1])
            labels = np.argmax(labels, axis=-1)

        # Update num_classes from max label value
        valid = labels.flatten()
        valid = valid[valid != ignore_index]
        if len(valid) > 0:
            num_classes = max(num_classes, int(np.max(valid)) + 1)

        all_labels.append(labels.flatten())

    if not all_labels:
        # Try to get num_classes from dataset
        if hasattr(dataset, 'label_names'):
            num_classes = len(dataset.label_names)
        return np.array([], dtype=np.int64), num_classes

    return np.concatenate(all_labels), num_classes

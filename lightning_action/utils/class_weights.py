"""Utility functions for lightning-action.

This module provides common utility functions used across the package,
including class weight computation for imbalanced datasets.
"""

import logging
from typing import Optional

import numpy as np
from typeguard import typechecked

logger = logging.getLogger(__name__)


@typechecked
def compute_class_weights(
    labels: np.ndarray,
    num_classes: Optional[int] = None,
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
    for cls, count in zip(unique_classes, counts):
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


@typechecked
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


@typechecked  
def collect_labels_from_datamodule(
    dataset,
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

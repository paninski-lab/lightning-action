"""Evaluation functions for the lightning-action package."""

import itertools

import numpy as np
from sklearn.metrics import precision_score, recall_score
from typeguard import typechecked


@typechecked
def get_precision_recall(
    true_classes: np.ndarray,
    pred_classes: np.ndarray,
    background: int | None = 0,
    n_classes: int | None = None,
) -> dict[str, np.ndarray]:
    """Compute precision and recall for classifier.

    Args:
        true_classes: entries should be in [0, K-1] where K is the number of classes
        pred_classes: entries should be in [0, K-1] where K is the number of classes
        background: defines the background class that identifies points with no supervised
            label; these time points are omitted from the precision and recall calculations;
            if None, no background class is utilized
        n_classes: total number of non-background classes; if None, will be inferred from
            true classes

    Returns:
        dictionary containing:
            'precision': precision for each class (including background class)
            'recall': recall for each class (including background class)
            'f1': f1 score for each class
    """
    assert true_classes.shape[0] == pred_classes.shape[0]

    # find all data points that are not background
    if background is not None:
        assert background == 0  # need to generalize
        obs_idxs = np.where(true_classes != background)[0]
    else:
        obs_idxs = np.arange(true_classes.shape[0])

    if n_classes is None:
        n_classes = len(np.unique(true_classes[obs_idxs]))

    # set of labels to include in metric computations
    if background is not None:
        labels = np.arange(1, n_classes + 1)
    else:
        labels = np.arange(n_classes)

    precision = precision_score(
        true_classes[obs_idxs], pred_classes[obs_idxs],
        labels=labels, average=None, zero_division=0,
    )
    recall = recall_score(
        true_classes[obs_idxs], pred_classes[obs_idxs],
        labels=labels, average=None, zero_division=0,
    )

    # replace 0s with NaNs for classes with no ground truth
    # for n in range(precision.shape[0]):
    #     if precision[n] == 0 and recall[n] == 0:
    #         precision[n] = np.nan
    #         recall[n] = np.nan

    # compute f1
    p = precision
    r = recall
    f1 = 2 * p * r / (p + r + 1e-10)
    return {'precision': p, 'recall': r, 'f1': f1}


@typechecked
def int_over_union(array1: np.ndarray, array2: np.ndarray) -> dict[int, float]:
    """Compute intersection over union for two 1D arrays.

    Args:
        array1: integer array of shape (n,)
        array2: integer array of shape (n,)

    Returns:
        dictionary where keys are integer values in arrays and values are
        corresponding IoU (float)
    """
    vals = np.unique(np.concatenate([np.unique(array1), np.unique(array2)])).tolist()
    iou = {val: np.nan for val in vals}
    for val in vals:
        intersection = np.sum((array1 == val) & (array2 == val))
        union = np.sum((array1 == val) | (array2 == val))
        iou[val] = intersection / union
    return iou


@typechecked
def run_lengths(array: np.ndarray) -> dict[int, list[int]]:
    """Compute distribution of run lengths for an array with integer entries.

    Args:
        array: single-dimensional array

    Returns:
        dictionary where keys are integer values up to max value in array and
        values are lists of run lengths

    Example:
        >>> a = [1, 1, 1, 0, 0, 4, 4, 4, 4, 4, 4, 0, 1, 1, 1, 1]
        >>> run_lengths(a)
        {0: [2, 1], 1: [3, 4], 2: [], 3: [], 4: [6]}
    """
    seqs = {int(k): [] for k in np.arange(np.max(array) + 1)}
    for key, iterable in itertools.groupby(array):
        seqs[key].append(len(list(iterable)))
    return seqs

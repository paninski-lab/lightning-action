"""Data transforms for preprocessing behavioral data.

This module contains transform classes adapted from daart for preprocessing
marker data, labels, and features in action segmentation pipelines.
"""

from abc import ABC, abstractmethod

import numpy as np
from jaxtyping import Float
from typeguard import typechecked


class Transform(ABC):
    """Abstract base class for data transforms.
    
    All transforms should inherit from this class and implement the __call__ method
    to process data arrays.
    """

    @abstractmethod
    def __call__(
        self,
        data: Float[np.ndarray, 'time features']
    ) -> Float[np.ndarray, 'time features']:
        """Apply transform to input data.
        
        Args:
            data: input data array with shape (time, features)
            
        Returns:
            transformed data array with same shape
        """
        raise NotImplementedError

    @abstractmethod
    def __repr__(self) -> str:
        """Return string representation of transform."""
        raise NotImplementedError


class Compose:
    """Compose several transforms together.

    Adapted from PyTorch's transform composition pattern. Applies transforms
    sequentially in the order provided.

    Example:
        >>> transform = Compose([
        ...     ZScore(),
        ...     MotionEnergy(),
        ... ])
        >>> transformed_data = transform(data)
    """

    @typechecked
    def __init__(self, transforms: list[Transform]):
        """Initialize Compose transform.
        
        Args:
            transforms: list of transform objects to apply sequentially
        """
        self.transforms = transforms

    @typechecked
    def __call__(
        self,
        data: Float[np.ndarray, 'time features'],
    ) -> Float[np.ndarray, 'time features']:
        """Apply all transforms sequentially.
        
        Args:
            data: input data array with shape (time, features)
            
        Returns:
            transformed data after applying all transforms
        """
        result = data
        for transform in self.transforms:
            result = transform(result)
        return result

    def __repr__(self) -> str:
        """Return string representation of composed transforms."""
        format_string = f'{self.__class__.__name__}('
        for transform in self.transforms:
            format_string += f'{transform}, '
        # remove trailing comma and space, add closing paren
        format_string = format_string.rstrip(', ') + ')'
        return format_string


class MotionEnergy(Transform):
    """Compute motion energy across time dimension.
    
    Calculates the absolute difference between consecutive time points,
    providing a measure of movement or change in the signal.
    """

    def __init__(self):
        """Initialize MotionEnergy transform."""
        pass

    @typechecked
    def __call__(
        self,
        data: Float[np.ndarray, 'time features'],
    ) -> Float[np.ndarray, 'time features']:
        """Compute motion energy from input data.
        
        Args:
            data: input data array with shape (time, features)
            
        Returns:
            motion energy array with same shape as input.
            First time point is zeros, subsequent points are absolute
            differences from previous time point.
        """
        if data.shape[0] < 2:
            # if less than 2 time points, return zeros
            return np.zeros_like(data)
        
        # compute absolute difference between consecutive time points
        # duplicate first time point before diff to maintain shape
        result = np.abs(np.diff(data, axis=0, prepend=data[None, 0]))
        
        return result

    def __repr__(self) -> str:
        """Return string representation of transform."""
        return 'MotionEnergy()'


class ZScore(Transform):
    """Apply z-score normalization to data.
    
    Normalizes data by subtracting the mean and dividing by standard deviation
    across the time dimension for each feature. Handles zero-variance features
    by leaving them unchanged.
    """

    @typechecked
    def __init__(self, eps: float = 1e-8):
        """Initialize ZScore transform.
        
        Args:
            eps: small epsilon value to avoid division by zero
        """
        self.eps = eps
        self.mean = None
        self.std = None

    @typechecked
    def __call__(
        self,
        data: Float[np.ndarray, 'time features'],
    ) -> Float[np.ndarray, 'time features']:
        """Apply z-score normalization to input data.
        
        Args:
            data: input data array with shape (time, features)
            
        Returns:
            z-score normalized data with same shape as input
        """
        # make a copy to avoid modifying input data
        result = data.copy()
        
        # compute mean and std across time dimension
        mean = np.mean(result, axis=0)
        std = np.std(result, axis=0)
        self.mean = mean
        self.std = std

        # subtract mean
        result -= mean
        
        # divide by std only for features with non-zero variance
        nonzero_std = std > self.eps
        result[:, nonzero_std] = result[:, nonzero_std] / std[nonzero_std]
        
        return result

    def __repr__(self) -> str:
        """Return string representation of transform."""
        return f'ZScore(eps={self.eps})'


class VelocityConcat(Transform):
    """Compute velocity and concatenate with original signal.
    
    Computes the velocity (first derivative) of the signal along the time dimension
    using np.diff, pads the first timepoint with zeros, and concatenates the
    velocity to the original signal along the feature dimension.
    """

    def __init__(self):
        """Initialize VelocityConcat transform."""
        pass

    @typechecked
    def __call__(
        self,
        data: Float[np.ndarray, 'time features'],
    ) -> Float[np.ndarray, 'time features_times_two']:
        """Compute velocity and concatenate with original signal.
        
        Args:
            data: input data array with shape (time, features)
            
        Returns:
            concatenated array with shape (time, features*2) where the first
            'features' columns are the original signal and the second 'features'
            columns are the velocity
        """
        if data.shape[0] < 2:
            # if less than 2 time points, velocity is zero
            velocity = np.zeros_like(data)
        else:
            # compute velocity using diff along time dimension
            # pad first timepoint so velocity is zero for first row
            velocity = np.diff(data, axis=0, prepend=data[None, 0])
        
        # concatenate original signal with velocity along feature dimension
        result = np.concatenate([data, velocity], axis=1)
        
        return result

    def __repr__(self) -> str:
        """Return string representation of transform."""
        return 'VelocityConcat()'

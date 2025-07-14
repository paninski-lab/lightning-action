"""Backbone architectures for action segmentation models."""

from .rnn import RNN
from .temporalmlp import TemporalMLP

__all__ = [
    'RNN',
    'TemporalMLP',
]

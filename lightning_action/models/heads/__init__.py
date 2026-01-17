"""Head architectures for action segmentation models."""

from lightning_action.models.heads.rnn import RNN
from lightning_action.models.heads.tcn import DilatedTCN
from lightning_action.models.heads.temporalmlp import TemporalMLP
from lightning_action.models.necks.mha_pooling import MultiheadAttentionPooling

__all__ = [
    'DilatedTCN',
    'RNN',
    'TemporalMLP',
    'MultiheadAttentionPooling'
]

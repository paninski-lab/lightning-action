"""RNN (LSTM/GRU) backbone for action segmentation.

This module implements RNN-based backbones adapted from daart for use with
PyTorch Lightning. Supports both LSTM and GRU architectures with bidirectional
processing capability.
"""

import torch
from jaxtyping import Float
from torch import nn
from typeguard import typechecked


class RNN(nn.Module):
    """RNN backbone for temporal sequence modeling.
    
    Encoder-only implementation supporting LSTM and GRU architectures with
    optional bidirectional processing.
    """

    @typechecked
    def __init__(
        self,
        input_size: int,
        num_hid_units: int,
        num_layers: int,
        rnn_type: str = 'lstm',
        bidirectional: bool = False,
        dropout_rate: float = 0.0,
        seed: int = 42,
    ):
        """Initialize RNN backbone.
        
        Args:
            input_size: number of input features
            num_hid_units: number of hidden units per RNN layer
            num_layers: number of RNN layers
            rnn_type: 'lstm' or 'gru'
            bidirectional: True for bidirectional RNN
            dropout_rate: dropout probability (applied between RNN layers)
            seed: random seed for weight initialization
            
        Raises:
            ValueError: if rnn_type is not 'lstm' or 'gru'
        """
        super().__init__()
        
        self.input_size = input_size
        self.num_hid_units = num_hid_units
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional
        self.dropout_rate = dropout_rate
        self.seed = seed
        
        # validate rnn type
        if self.rnn_type not in ['lstm', 'gru']:
            raise ValueError(f'Invalid rnn_type "{rnn_type}"; must be "lstm" or "gru"')
        
        # set random seed
        torch.manual_seed(seed)
        
        # build model
        self._build_model()

    def _build_model(self):
        """Build the RNN model layers."""
        # create RNN layer
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=self.input_size,
                hidden_size=self.num_hid_units,
                num_layers=self.num_layers,
                batch_first=True,
                bidirectional=self.bidirectional,
                dropout=self.dropout_rate if self.num_layers > 1 else 0.0,
            )
        else:  # gru
            self.rnn = nn.GRU(
                input_size=self.input_size,
                hidden_size=self.num_hid_units,
                num_layers=self.num_layers,
                batch_first=True,
                bidirectional=self.bidirectional,
                dropout=self.dropout_rate if self.num_layers > 1 else 0.0,
            )
        
        # output projection layer
        # bidirectional doubles the hidden size
        rnn_output_size = self.num_hid_units * (2 if self.bidirectional else 1)
        self.output_projection = nn.Linear(rnn_output_size, self.num_hid_units)

    @typechecked
    def forward(
        self,
        x: Float[torch.Tensor, 'batch sequence features']
    ) -> Float[torch.Tensor, 'batch sequence n_hid_units']:
        """Forward pass through RNN backbone.
        
        Args:
            x: input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            output tensor of shape (batch_size, sequence_length, num_hid_units)
        """
        # pass through RNN
        # rnn_output: (batch, sequence, hidden_size * num_directions)
        # hidden states are not returned as we only need the outputs
        rnn_output, _ = self.rnn(x)
        
        # project to desired output size
        # output: (batch, sequence, num_hid_units)
        output = self.output_projection(rnn_output)
        
        return output

    def __repr__(self) -> str:
        """String representation of the model."""
        return (
            f'RNN(\n'
            f'  input_size={self.input_size},\n'
            f'  num_hid_units={self.num_hid_units},\n'
            f'  num_layers={self.num_layers},\n'
            f'  rnn_type={self.rnn_type},\n'
            f'  bidirectional={self.bidirectional},\n'
            f'  dropout_rate={self.dropout_rate}\n'
            f')'
        )

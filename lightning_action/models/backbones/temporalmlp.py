"""Temporal MLP backbone for action segmentation.

This module implements the TemporalMLP architecture adapted from daart,
which uses 1D convolution for temporal context followed by dense layers.
"""

import torch
import torch.nn as nn
from jaxtyping import Float
from typeguard import typechecked


class TemporalMLP(nn.Module):
    """Temporal Multi-Layer Perceptron for sequence encoding.
    
    This backbone uses 1D convolution to capture temporal dependencies
    followed by dense layers for feature extraction.
    
    Architecture:
    1. 1D Conv layer with temporal window (2*n_lags + 1)
    2. n_hid_layers dense layers with activations
    3. Final dense layer (no activation)
    
    Input shape: (batch, sequence, features)
    Output shape: (batch, sequence, n_hid_units)
    """

    @typechecked
    def __init__(
        self,
        input_size: int,
        num_hid_units: int,
        num_layers: int,
        num_lags: int = 5,
        activation: str = 'lrelu',
        dropout_rate: float = 0.0,
        seed: int = 42,
    ):
        """Initialize TemporalMLP backbone.
        
        Args:
            input_size: number of input features per timestep
            num_hid_units: number of hidden units in dense layers
            num_layers: number of hidden dense layers
            num_lags: number of temporal lags for 1D conv window (creates 2*n_lags + 1 kernel)
            activation: activation function ('relu', 'lrelu', 'sigmoid', 'tanh', 'linear')
            dropout_rate: dropout probability (0.0 = no dropout)
            seed: random seed for weight initialization
        """
        super().__init__()
        
        self.input_size = input_size
        self.num_hid_units = num_hid_units
        self.num_layers = num_layers
        self.num_lags = num_lags
        self.activation = activation
        self.dropout_rate = dropout_rate

        # set random seed
        torch.manual_seed(seed)

        # build model
        self.layers = nn.ModuleList()
        self._build_model()

    def _build_model(self):
        """Build the TemporalMLP model layers."""
        # initial 1D convolution layer for temporal context
        conv_kernel_size = 2 * self.num_lags + 1
        conv_layer = nn.Conv1d(
            in_channels=self.input_size,
            out_channels=self.num_hid_units,
            kernel_size=conv_kernel_size,
            padding=self.num_lags,  # maintains sequence length
        )
        self.layers.append(conv_layer)
        
        # add activation after conv layer
        if self.activation != 'linear':
            self.layers.append(self._get_activation())
        
        # add dropout if specified
        if self.dropout_rate > 0.0:
            self.layers.append(nn.Dropout(self.dropout_rate))
        
        # dense layers
        for i in range(self.num_layers):
            # linear layer
            linear_layer = nn.Linear(self.num_hid_units, self.num_hid_units)
            self.layers.append(linear_layer)
            
            # activation (except for final layer)
            if i < self.num_layers - 1 and self.activation != 'linear':
                self.layers.append(self._get_activation())
            
            # dropout (except for final layer)
            if i < self.num_layers - 1 and self.dropout_rate > 0.0:
                self.layers.append(nn.Dropout(self.dropout_rate))

    def _get_activation(self) -> nn.Module:
        """Get activation function module.
        
        Returns:
            activation function module
            
        Raises:
            ValueError: if activation type is not supported
        """
        if self.activation == 'relu':
            return nn.ReLU()
        elif self.activation == 'lrelu':
            return nn.LeakyReLU(negative_slope=0.01)
        elif self.activation == 'sigmoid':
            return nn.Sigmoid()
        elif self.activation == 'tanh':
            return nn.Tanh()
        elif self.activation == 'linear':
            return nn.Identity()
        else:
            raise ValueError(f'Unsupported activation: {self.activation}')

    @typechecked
    def forward(
        self,
        x: Float[torch.Tensor, 'batch sequence features'],
    ) -> Float[torch.Tensor, 'batch sequence n_hid_units']:
        """Forward pass through TemporalMLP.
        
        Args:
            x: input tensor with shape (batch, sequence, features)
            
        Returns:
            encoded features with shape (batch, sequence, n_hid_units)
        """
        batch_size, sequence_length, features = x.shape
        
        # start with input
        output = x
        
        # apply layers sequentially
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Conv1d):
                # conv1d expects (batch, channels, sequence)
                output = output.transpose(1, 2)  # (batch, features, sequence)
                output = layer(output)
                output = output.transpose(1, 2)  # (batch, sequence, hidden)
            else:
                # dense layers and activations work on last dimension
                output = layer(output)
        
        return output

    def __repr__(self) -> str:
        """Return string representation of the model."""
        return (
            f'TemporalMLP('
            f'  input_size={self.input_size}, '
            f'  num_hid_units={self.num_hid_units}, '
            f'  num_layers={self.num_layers}, '
            f'  num_lags={self.num_lags}, '
            f'  activation={self.activation}, '
            f'  dropout_rate={self.dropout_rate}, '
            ')'
        )

"""Temporal Convolution Network (TCN) backbone for action segmentation.

This module implements a temporal convolutional network with dilated convolutions and
residual connections for temporal modeling.
"""

import torch
from jaxtyping import Float
from torch import nn
from typeguard import typechecked


class DilatedTCN(nn.Module):
    """Dilated Temporal Convolutional Network backbone.
    
    Encoder-only implementation of a dilated TCN with residual connections.
    """

    @typechecked
    def __init__(
        self,
        input_size: int,
        num_hid_units: int,
        num_layers: int,
        num_lags: int = 1,
        activation: str = 'lrelu',
        dropout_rate: float = 0.2,
        seed: int = 42,
    ):
        """Initialize DilatedTCN backbone.
        
        Args:
            input_size: number of input features
            num_hid_units: number of hidden units per layer
            num_layers: number of TCN layers
            num_lags: kernel size parameter for dilated convolutions
            activation: activation function name
            dropout_rate: dropout probability
            seed: random seed for weight initialization
            
        Raises:
            ValueError: if activation function is not supported
        """
        super().__init__()
        
        self.input_size = input_size
        self.num_hid_units = num_hid_units
        self.num_layers = num_layers
        self.num_lags = num_lags
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.seed = seed
        
        # set random seed
        torch.manual_seed(seed)
        
        # build model
        self.model = nn.Sequential()
        self._build_model()

    def _build_model(self):
        """Build the TCN model layers."""
        for i_layer in range(self.num_layers):
            # dilation increases exponentially
            dilation = 2 ** i_layer
            
            # determine layer sizes
            in_size = self.input_size if i_layer == 0 else self.num_hid_units
            hid_size = self.num_hid_units
            
            if i_layer == (self.num_layers - 1):
                # final layer
                out_size = self.num_hid_units
            else:
                # intermediate layer
                out_size = self.num_hid_units
            
            # create TCN block
            tcn_block = DilationBlock(
                input_size=in_size,
                int_size=hid_size,
                output_size=out_size,
                kernel_size=self.num_lags,
                stride=1,
                dilation=dilation,
                activation=self.activation,
                dropout=self.dropout_rate,
            )
            
            # add to model
            block_name = f'tcn_block_{i_layer:02d}'
            self.model.add_module(block_name, tcn_block)

    @typechecked
    def forward(
        self,
        x: Float[torch.Tensor, 'batch sequence features']
    ) -> Float[torch.Tensor, 'batch sequence n_hid_units']:
        """Forward pass through TCN backbone.
        
        Args:
            x: input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            output tensor of shape (batch_size, sequence_length, num_hid_units)
        """
        # TCN expects (batch, channels, time) format
        # input: (batch, sequence, features) -> (batch, features, sequence)
        x_transposed = x.transpose(1, 2)
        
        # pass through TCN layers
        output_transposed = self.model(x_transposed)
        
        # convert back to (batch, sequence, features) format
        # output: (batch, features, sequence) -> (batch, sequence, features)
        output = output_transposed.transpose(1, 2)
        
        return output

    def __repr__(self) -> str:
        """String representation of the model."""
        return (
            f'DilatedTCN(\n'
            f'  input_size={self.input_size},\n'
            f'  num_hid_units={self.num_hid_units},\n'
            f'  num_layers={self.num_layers},\n'
            f'  num_lags={self.num_lags},\n'
            f'  activation={self.activation},\n'
            f'  dropout_rate={self.dropout_rate}\n'
            f')'
        )


class DilationBlock(nn.Module):
    """Residual temporal block module for use with DilatedTCN class.

    Implements a residual block with dilated convolutions for temporal modeling.
    """

    @typechecked
    def __init__(
        self,
        input_size: int,
        int_size: int,
        output_size: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 2,
        activation: str = 'lrelu',
        dropout: float = 0.2,
        final_activation: str | None = None,
    ):
        """Initialize DilationBlock.

        Args:
            input_size: number of input channels
            int_size: number of intermediate channels
            output_size: number of output channels
            kernel_size: size of convolutional kernel
            stride: convolution stride
            dilation: dilation factor
            activation: activation function name
            dropout: dropout probability
            final_activation: final activation function name (defaults to activation)
        """
        super().__init__()

        self.input_size = input_size
        self.int_size = int_size
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.activation_str = activation
        self.dropout_rate = dropout
        self.final_activation_str = final_activation or activation

        # build model

        # first convolution
        self.conv0 = nn.utils.parametrizations.weight_norm(nn.Conv1d(
            in_channels=input_size,
            out_channels=int_size,
            stride=stride,
            dilation=dilation,
            kernel_size=kernel_size * 2 + 1,  # window around t
            padding=kernel_size * dilation,  # same output
        ))

        # second convolution
        self.conv1 = nn.utils.parametrizations.weight_norm(nn.Conv1d(
            in_channels=int_size,
            out_channels=output_size,
            stride=stride,
            dilation=dilation,
            kernel_size=kernel_size * 2 + 1,  # window around t
            padding=kernel_size * dilation,  # same output
        ))

        # activation functions
        self.activation = self._get_activation_func(activation)
        self.final_activation = self._get_activation_func(self.final_activation_str)

        # dropout
        self.dropout = nn.Dropout1d(dropout)

        # build main block
        self.block = nn.Sequential()

        # first conv -> activation -> dropout
        self.block.add_module('conv1d_layer_0', self.conv0)
        if self.activation is not None:
            self.block.add_module(f'{activation}_0', self.activation)
        self.block.add_module('dropout_0', self.dropout)

        # second conv -> activation -> dropout
        self.block.add_module('conv1d_layer_1', self.conv1)
        if self.activation is not None:
            self.block.add_module(f'{activation}_1', self.activation)
        self.block.add_module('dropout_1', self.dropout)

        # residual connection projection if needed
        if input_size != output_size:
            self.downsample = nn.Conv1d(input_size, output_size, kernel_size=1)
        else:
            self.downsample = None

        # initialize weights
        self._init_weights()

    @staticmethod
    def _get_activation_func(activation) -> nn.Module:
        """Get activation function module.

        Returns:
            activation function module

        Raises:
            ValueError: if activation type is not supported
        """
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'lrelu':
            return nn.LeakyReLU(negative_slope=0.01)
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'linear':
            return nn.Identity()
        else:
            raise ValueError(f'Unsupported activation: {activation}')

    def _init_weights(self):
        """Initialize weights with normal distribution."""
        self.conv0.weight.data.normal_(0, 0.01)
        self.conv1.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through dilation block.

        Args:
            x: input tensor of shape (batch, channels, time)

        Returns:
            output tensor of shape (batch, output_size, time)
        """
        # main path
        out = self.block(x)

        # residual connection
        res = x if self.downsample is None else self.downsample(x)

        # combine and apply final activation
        combined = self.final_activation(out + res)

        return combined

    def __repr__(self) -> str:
        """String representation of the block."""
        return (
            f'DilationBlock(\n'
            f'  input_size={self.input_size}, int_size={self.int_size}, '
            f'  output_size={self.output_size},\n'
            f'  kernel_size={self.kernel_size}, dilation={self.dilation},\n'
            f'  activation={self.activation_str}, dropout={self.dropout_rate}\n'
            f')'
        )

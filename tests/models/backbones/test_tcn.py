"""Tests for DilatedTCN backbone model."""

import pytest
import torch

from lightning_action.models.backbones import DilatedTCN
from lightning_action.models.backbones.tcn import DilationBlock


class TestDilationBlock:
    """Test the DilationBlock class."""

    @pytest.fixture
    def basic_block_config(self):
        """Basic configuration for DilationBlock."""
        return {
            'input_size': 6,
            'int_size': 32,
            'output_size': 32,
            'kernel_size': 3,
            'stride': 1,
            'dilation': 2,
            'activation': 'relu',
            'dropout': 0.1,
        }

    def test_initialization(self, basic_block_config):
        """Test DilationBlock initialization."""
        block = DilationBlock(**basic_block_config)
        
        # check attributes
        assert block.input_size == basic_block_config['input_size']
        assert block.int_size == basic_block_config['int_size']
        assert block.output_size == basic_block_config['output_size']
        assert block.kernel_size == basic_block_config['kernel_size']
        assert block.dilation == basic_block_config['dilation']
        assert block.activation_str == basic_block_config['activation']
        assert block.dropout_rate == basic_block_config['dropout']
        
        # check components exist
        assert hasattr(block, 'conv0')
        assert hasattr(block, 'conv1')
        assert hasattr(block, 'activation')
        assert hasattr(block, 'final_activation')
        assert hasattr(block, 'dropout')
        assert hasattr(block, 'block')

    def test_downsample_creation(self):
        """Test that downsample layer is created when input_size != output_size."""
        # same size - no downsample
        block1 = DilationBlock(
            input_size=32, int_size=32, output_size=32, kernel_size=3, dilation=2,
        )
        assert block1.downsample is None

        # different size - downsample needed
        block2 = DilationBlock(
            input_size=16, int_size=32, output_size=32, kernel_size=3, dilation=2
        )
        assert block2.downsample is not None
        assert isinstance(block2.downsample, torch.nn.Conv1d)
        assert block2.downsample.in_channels == 16
        assert block2.downsample.out_channels == 32

    def test_forward_pass_shape(self, basic_block_config):
        """Test forward pass produces correct output shapes."""
        block = DilationBlock(**basic_block_config)
        
        batch_size, channels, time_steps = 2, basic_block_config['input_size'], 100
        x = torch.randn(batch_size, channels, time_steps)
        
        output = block(x)
        
        expected_shape = (batch_size, basic_block_config['output_size'], time_steps)
        assert output.shape == expected_shape
        assert torch.isfinite(output).all()

    def test_different_activations(self):
        """Test block with different activation functions."""
        base_config = {
            'input_size': 6,
            'int_size': 32,
            'output_size': 32,
            'kernel_size': 3,
            'dilation': 2,
            'dropout': 0.1,
        }
        
        activations = ['relu', 'lrelu', 'sigmoid', 'tanh', 'linear']
        
        for activation in activations:
            block = DilationBlock(activation=activation, **base_config)
            x = torch.randn(2, 6, 100)
            output = block(x)
            
            assert output.shape == (2, 32, 100)
            assert torch.isfinite(output).all()

    def test_final_activation_override(self):
        """Test final activation can be different from main activation."""
        block = DilationBlock(
            input_size=6, int_size=32, output_size=32,
            kernel_size=3, dilation=2,
            activation='relu', final_activation='tanh'
        )
        
        x = torch.randn(2, 6, 100)
        output = block(x)
        
        assert output.shape == (2, 32, 100)
        assert torch.isfinite(output).all()
        # tanh output should be in [-1, 1] range
        assert output.min() >= -1.0
        assert output.max() <= 1.0

    def test_gradient_flow(self, basic_block_config):
        """Test gradient flow through the block."""
        block = DilationBlock(**basic_block_config)
        
        x = torch.randn(2, 6, 100, requires_grad=True)
        output = block(x)
        
        loss = output.sum()
        loss.backward()
        
        # check gradients exist
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        
        # check block parameters have gradients
        for name, param in block.named_parameters():
            assert param.grad is not None, f"No gradient for parameter {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for parameter {name}"

    def test_repr(self, basic_block_config):
        """Test string representation of DilationBlock."""
        block = DilationBlock(**basic_block_config)
        repr_str = repr(block)
        
        assert 'DilationBlock' in repr_str
        assert str(basic_block_config['input_size']) in repr_str
        assert str(basic_block_config['output_size']) in repr_str
        assert str(basic_block_config['kernel_size']) in repr_str
        assert str(basic_block_config['dilation']) in repr_str
        assert basic_block_config['activation'] in repr_str


class TestDilatedTCN:
    """Test the DilatedTCN backbone class."""

    @pytest.fixture
    def tcn_configs(self):
        """Fixture providing different TCN configurations for testing."""
        return [
            {
                'input_size': 6,
                'num_hid_units': 32,
                'num_layers': 2,
                'num_lags': 1,
                'activation': 'relu',
                'dropout_rate': 0.1,
                'seed': 42,
            },
            {
                'input_size': 10,
                'num_hid_units': 64,
                'num_layers': 4,
                'num_lags': 3,
                'activation': 'lrelu',
                'dropout_rate': 0.2,
                'seed': 123,
            },
            {
                'input_size': 8,
                'num_hid_units': 48,
                'num_layers': 3,
                'num_lags': 2,
                'activation': 'tanh',
                'dropout_rate': 0.0,
                'seed': 456,
            },
        ]

    def test_basic_initialization(self, tcn_configs):
        """Test DilatedTCN initialization with different configurations."""
        for config in tcn_configs:
            model = DilatedTCN(**config)
            
            # check basic attributes
            assert model.input_size == config['input_size']
            assert model.num_hid_units == config['num_hid_units']
            assert model.num_layers == config['num_layers']
            assert model.num_lags == config['num_lags']
            assert model.activation == config['activation']
            assert model.dropout_rate == config['dropout_rate']
            assert model.seed == config['seed']
            
            # check model structure
            assert hasattr(model, 'model')
            assert isinstance(model.model, torch.nn.Sequential)
            
            # check number of blocks
            assert len(model.model) == config['num_layers']
            
            # check each block is a DilationBlock
            for i, block in enumerate(model.model):
                assert isinstance(block, DilationBlock)
                # check dilation increases exponentially
                expected_dilation = 2 ** i
                assert block.dilation == expected_dilation

    def test_forward_pass_shape(self, tcn_configs):
        """Test forward pass produces correct output shapes."""
        for config in tcn_configs:
            model = DilatedTCN(**config)
            
            batch_size, sequence_length = 2, 100
            x = torch.randn(batch_size, sequence_length, config['input_size'])
            
            # forward pass
            output = model(x)
            
            # check output shape
            expected_shape = (batch_size, sequence_length, config['num_hid_units'])
            assert output.shape == expected_shape
            
            # check output is finite
            assert torch.isfinite(output).all()

    def test_layer_sizes(self, tcn_configs):
        """Test that layer input/output sizes are correct."""
        config = tcn_configs[0]
        model = DilatedTCN(**config)

        for i, block in enumerate(model.model):
            if i == 0:
                # first layer takes input_size
                assert block.input_size == config['input_size']
            else:
                # subsequent layers take num_hid_units
                assert block.input_size == config['num_hid_units']

            # all layers output num_hid_units
            assert block.output_size == config['num_hid_units']

    def test_different_activations(self):
        """Test TCN with different activation functions."""
        base_config = {
            'input_size': 6,
            'num_hid_units': 32,
            'num_layers': 2,
            'num_lags': 1,
            'dropout_rate': 0.1,
            'seed': 42,
        }

        activations = ['relu', 'lrelu', 'sigmoid', 'tanh', 'linear']

        for activation in activations:
            config = base_config.copy()
            config['activation'] = activation

            model = DilatedTCN(**config)
            x = torch.randn(2, 100, config['input_size'])
            output = model(x)

            expected_shape = (2, 100, config['num_hid_units'])
            assert output.shape == expected_shape
            assert torch.isfinite(output).all()

    def test_invalid_activation(self):
        """Test that invalid activation raises ValueError."""
        config = {
            'input_size': 6,
            'num_hid_units': 32,
            'num_layers': 2,
            'activation': 'invalid_activation',
        }

        with pytest.raises(
                ValueError,
                match='Unsupported activation'
        ):
            DilatedTCN(**config)

    def test_dilation_pattern(self, tcn_configs):
        """Test that dilation pattern is correct."""
        config = tcn_configs[1]  # use config with multiple layers
        model = DilatedTCN(**config)

        # check dilation pattern: 2^0, 2^1, 2^2, ...
        for i, block in enumerate(model.model):
            expected_dilation = 2 ** i
            assert block.dilation == expected_dilation

    def test_deterministic_output_with_seed(self):
        """Test that same seed produces deterministic outputs."""
        config = {
            'input_size': 6,
            'num_hid_units': 32,
            'num_layers': 2,
            'num_lags': 1,
            'activation': 'relu',
            'dropout_rate': 0.0,  # no dropout for deterministic test
            'seed': 42,
        }

        x = torch.randn(2, 100, config['input_size'])

        # create two models with same seed
        model1 = DilatedTCN(**config)
        model2 = DilatedTCN(**config)

        # set to eval mode
        model1.eval()
        model2.eval()

        with torch.no_grad():
            output1 = model1(x)
            output2 = model2(x)

        # outputs should be identical due to same seed
        assert torch.allclose(output1, output2, atol=1e-6)

    def test_gradient_flow(self, tcn_configs):
        """Test gradient flow through the TCN."""
        config = tcn_configs[0]
        model = DilatedTCN(**config)

        x = torch.randn(2, 100, config['input_size'], requires_grad=True)
        output = model(x)

        # compute dummy loss and backpropagate
        loss = output.sum()
        loss.backward()

        # check that input has gradients
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

        # check that model parameters have gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for parameter {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for parameter {name}"

    def test_different_batch_sizes(self, tcn_configs):
        """Test TCN with different batch sizes."""
        config = tcn_configs[0]  # use first config
        model = DilatedTCN(**config)

        sequence_length = 100
        input_size = config['input_size']

        for batch_size in [1, 2, 4, 8]:
            x = torch.randn(batch_size, sequence_length, input_size)
            output = model(x)

            expected_shape = (batch_size, sequence_length, config['num_hid_units'])
            assert output.shape == expected_shape

    def test_different_sequence_lengths(self, tcn_configs):
        """Test TCN with different sequence lengths."""
        config = tcn_configs[0]  # use first config
        model = DilatedTCN(**config)
        
        batch_size = 2
        input_size = config['input_size']
        
        for seq_len in [10, 50, 100, 200, 500]:
            x = torch.randn(batch_size, seq_len, input_size)
            output = model(x)
            
            expected_shape = (batch_size, seq_len, config['num_hid_units'])
            assert output.shape == expected_shape

    def test_repr(self, tcn_configs):
        """Test string representation of the TCN."""
        config = tcn_configs[0]
        model = DilatedTCN(**config)
        repr_str = repr(model)
        
        assert 'DilatedTCN' in repr_str
        assert str(config['input_size']) in repr_str
        assert str(config['num_hid_units']) in repr_str
        assert str(config['num_layers']) in repr_str
        assert str(config['num_lags']) in repr_str
        assert config['activation'] in repr_str
        assert str(config['dropout_rate']) in repr_str

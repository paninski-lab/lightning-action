"""Tests for RNN backbone model."""

import pytest
import torch

from lightning_action.models.backbones import RNN


class TestRNN:
    """Test the RNN backbone class."""

    @pytest.fixture
    def rnn_configs(self):
        """Fixture providing different RNN configurations for testing."""
        return [
            {
                'input_size': 6,
                'num_hid_units': 32,
                'num_layers': 1,
                'rnn_type': 'lstm',
                'bidirectional': False,
                'dropout_rate': 0.0,
                'seed': 42,
            },
            {
                'input_size': 10,
                'num_hid_units': 64,
                'num_layers': 2,
                'rnn_type': 'gru',
                'bidirectional': True,
                'dropout_rate': 0.1,
                'seed': 123,
            },
            {
                'input_size': 8,
                'num_hid_units': 48,
                'num_layers': 3,
                'rnn_type': 'lstm',
                'bidirectional': True,
                'dropout_rate': 0.2,
                'seed': 456,
            },
        ]

    def test_basic_initialization(self, rnn_configs):
        """Test RNN initialization with different configurations."""
        for config in rnn_configs:
            model = RNN(**config)
            
            # check basic attributes
            assert model.input_size == config['input_size']
            assert model.num_hid_units == config['num_hid_units']
            assert model.num_layers == config['num_layers']
            assert model.rnn_type == config['rnn_type'].lower()
            assert model.bidirectional == config['bidirectional']
            assert model.dropout_rate == config['dropout_rate']
            assert model.seed == config['seed']
            
            # check that RNN layer is created correctly
            assert hasattr(model, 'rnn')
            assert hasattr(model, 'output_projection')
            
            # check RNN type
            if config['rnn_type'].lower() == 'lstm':
                assert isinstance(model.rnn, torch.nn.LSTM)
            else:
                assert isinstance(model.rnn, torch.nn.GRU)
            
            # check output projection layer
            expected_rnn_output_size = (
                config['num_hid_units'] *
                (2 if config['bidirectional'] else 1)
            )
            assert model.output_projection.in_features == expected_rnn_output_size
            assert model.output_projection.out_features == config['num_hid_units']

    def test_invalid_rnn_type(self):
        """Test that invalid RNN type raises ValueError."""
        with pytest.raises(ValueError, match='Invalid rnn_type "invalid"'):
            RNN(
                input_size=6,
                num_hid_units=32,
                num_layers=1,
                rnn_type='invalid',
            )

    def test_forward_pass_shape(self, rnn_configs):
        """Test forward pass produces correct output shapes."""
        for config in rnn_configs:
            model = RNN(**config)
            
            # create input with correct input size
            batch_size, sequence_length = 2, 100
            x = torch.randn(batch_size, sequence_length, config['input_size'])
            
            # forward pass
            output = model(x)
            
            # check output shape
            expected_shape = (batch_size, sequence_length, config['num_hid_units'])
            assert output.shape == expected_shape
            
            # check output is finite
            assert torch.isfinite(output).all()

    def test_bidirectional_vs_unidirectional(self):
        """Test bidirectional vs unidirectional RNNs."""
        input_size, num_hid_units, num_layers = 6, 32, 1
        x = torch.randn(2, 50, input_size)
        
        # create unidirectional and bidirectional models
        uni_model = RNN(
            input_size=input_size,
            num_hid_units=num_hid_units,
            num_layers=num_layers,
            rnn_type='lstm',
            bidirectional=False,
            seed=42,
        )
        
        bi_model = RNN(
            input_size=input_size,
            num_hid_units=num_hid_units,
            num_layers=num_layers,
            rnn_type='lstm',
            bidirectional=True,
            seed=42,
        )
        
        # forward pass
        uni_output = uni_model(x)
        bi_output = bi_model(x)
        
        # both should have same output shape (projection layer normalizes)
        assert uni_output.shape == bi_output.shape
        
        # but outputs should be different
        assert not torch.allclose(uni_output, bi_output, atol=1e-5)

    def test_dropout_behavior(self):
        """Test that dropout affects outputs in training mode."""
        config = {
            'input_size': 6,
            'num_hid_units': 32,
            'num_layers': 3,  # need multiple layers for dropout
            'rnn_type': 'lstm',
            'bidirectional': False,
            'dropout_rate': 0.5,
            'seed': 42,
        }

        model = RNN(**config)
        x = torch.randn(2, 50, config['input_size'])

        # set to training mode
        model.train()

        # multiple forward passes should give different results due to dropout
        output1 = model(x)
        output2 = model(x)

        # outputs should be different due to dropout (in most cases)
        # note: this test might occasionally fail due to randomness
        if config['dropout_rate'] > 0 and config['num_layers'] > 1:
            # allow some tolerance since dropout might not always change outputs significantly
            difference = torch.abs(output1 - output2).mean()
            assert difference > 1e-6, "Dropout should cause some variation in outputs"

    def test_single_layer_no_dropout(self):
        """Test that single layer RNN has no dropout regardless of dropout_rate."""
        config = {
            'input_size': 6,
            'num_hid_units': 32,
            'num_layers': 1,
            'rnn_type': 'lstm',
            'bidirectional': False,
            'dropout_rate': 0.5,  # should be ignored for single layer
            'seed': 42,
        }

        model = RNN(**config)

        # check that RNN layer has no dropout
        assert model.rnn.dropout == 0.0

    def test_deterministic_output_with_seed(self):
        """Test that same seed produces deterministic outputs."""
        config = {
            'input_size': 6,
            'num_hid_units': 32,
            'num_layers': 1,
            'rnn_type': 'lstm',
            'bidirectional': False,
            'dropout_rate': 0.0,
            'seed': 42,
        }

        x = torch.randn(2, 50, config['input_size'])

        # create two models with same seed
        model1 = RNN(**config)
        model2 = RNN(**config)

        # set to eval mode to avoid any randomness
        model1.eval()
        model2.eval()

        with torch.no_grad():
            output1 = model1(x)
            output2 = model2(x)

        # outputs should be identical due to same seed
        assert torch.allclose(output1, output2, atol=1e-6)

    def test_gradient_flow(self, rnn_configs):
        """Test that gradients flow through the model."""
        config = rnn_configs[0]
        model = RNN(**config)

        x = torch.randn(2, 50, config['input_size'], requires_grad=True)
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

    def test_different_batch_sizes(self, rnn_configs):
        """Test RNN with different batch sizes."""
        config = rnn_configs[0]  # use first config
        model = RNN(**config)

        sequence_length = 100
        input_size = config['input_size']

        for batch_size in [1, 2, 4, 8]:
            x = torch.randn(batch_size, sequence_length, input_size)
            output = model(x)

            expected_shape = (batch_size, sequence_length, config['num_hid_units'])
            assert output.shape == expected_shape

    def test_different_sequence_lengths(self, rnn_configs):
        """Test RNN with different sequence lengths."""
        config = rnn_configs[0]  # use first config
        model = RNN(**config)
        
        batch_size = 2
        input_size = config['input_size']
        
        for seq_len in [10, 50, 100, 200]:
            x = torch.randn(batch_size, seq_len, input_size)
            output = model(x)
            
            expected_shape = (batch_size, seq_len, config['num_hid_units'])
            assert output.shape == expected_shape

    def test_model_repr(self, rnn_configs):
        """Test string representation of the model."""
        for config in rnn_configs:
            model = RNN(**config)
            repr_str = repr(model)
            
            # check that key parameters are in the string representation
            assert str(config['input_size']) in repr_str
            assert str(config['num_hid_units']) in repr_str
            assert str(config['num_layers']) in repr_str
            assert config['rnn_type'] in repr_str
            assert str(config['bidirectional']) in repr_str
            assert str(config['dropout_rate']) in repr_str

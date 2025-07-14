"""Tests for TemporalMLP backbone."""

import pytest
import torch

from lightning_action.models import TemporalMLP


class TestTemporalMLP:
    """Test the TemporalMLP backbone class."""

    def test_basic_initialization(self):
        """Test basic model initialization."""
        model = TemporalMLP(
            input_size=6,
            num_hid_units=32,
            num_layers=2,
            num_lags=3,
        )
        
        assert model.input_size == 6
        assert model.num_hid_units == 32
        assert model.num_layers == 2
        assert model.num_lags == 3
        assert model.activation == 'lrelu'  # default
        assert model.dropout_rate == 0.0  # default

    def test_forward_pass_shape(self):
        """Test that forward pass preserves correct shapes."""
        batch_size, sequence_length, features = 2, 100, 6
        
        model = TemporalMLP(
            input_size=features,
            num_hid_units=32,
            num_layers=2,
            num_lags=5,
        )
        
        # create input tensor
        x = torch.randn(batch_size, sequence_length, features)
        
        # forward pass
        output = model(x)
        
        # check output shape
        expected_shape = (batch_size, sequence_length, 32)
        assert output.shape == expected_shape

    def test_different_activations(self):
        """Test different activation functions."""
        input_size, n_hid_units = 4, 16
        x = torch.randn(1, 50, input_size)
        
        activations = ['relu', 'lrelu', 'sigmoid', 'tanh', 'linear']
        
        for activation in activations:
            model = TemporalMLP(
                input_size=input_size,
                num_hid_units=n_hid_units,
                num_layers=1,
                activation=activation,
            )
            
            # should not raise error
            output = model(x)
            assert output.shape == (1, 50, n_hid_units)

    def test_invalid_activation(self):
        """Test that invalid activation raises error."""
        with pytest.raises(ValueError, match='Unsupported activation'):
            TemporalMLP(
                input_size=4,
                num_hid_units=16,
                num_layers=1,
                activation='invalid_activation',
            )

    def test_layer_count(self):
        """Test that correct number of layers are created."""
        n_hid_layers = 3
        model = TemporalMLP(
            input_size=4,
            num_hid_units=16,
            num_layers=n_hid_layers,
            dropout_rate=0.1,
        )

        # count different layer types
        conv_layers = sum(1 for layer in model.layers if isinstance(layer, torch.nn.Conv1d))
        linear_layers = sum(1 for layer in model.layers if isinstance(layer, torch.nn.Linear))
        dropout_layers = sum(1 for layer in model.layers if isinstance(layer, torch.nn.Dropout))

        assert conv_layers == 1  # one conv layer
        assert linear_layers == n_hid_layers  # n_hid_layers linear layers
        # dropout after conv + between hidden layers (but not after final)
        assert dropout_layers == n_hid_layers

    def test_zero_hidden_layers(self):
        """Test model with zero hidden layers (only conv layer)."""
        model = TemporalMLP(
            input_size=6,
            num_hid_units=32,
            num_layers=0,
            num_lags=2,
        )

        x = torch.randn(2, 50, 6)
        output = model(x)

        # should still work, only conv layer + activation
        assert output.shape == (2, 50, 32)

    def test_different_num_lags(self):
        """Test different temporal window sizes."""
        input_size, num_hid_units = 4, 16
        sequence_length = 100
        
        for num_lags in [1, 3, 5, 10]:
            model = TemporalMLP(
                input_size=input_size,
                num_hid_units=num_hid_units,
                num_layers=1,
                num_lags=num_lags,
            )
            
            x = torch.randn(1, sequence_length, input_size)
            output = model(x)
            
            # sequence length should be preserved regardless of num_lags
            assert output.shape == (1, sequence_length, num_hid_units)

    def test_dropout_behavior(self):
        """Test dropout functionality."""
        model = TemporalMLP(
            input_size=6,
            num_hid_units=32,
            num_layers=2,
            dropout_rate=0.5,
        )
        
        x = torch.randn(2, 50, 6)
        
        # test training mode (dropout active)
        model.train()
        output_train1 = model(x)
        output_train2 = model(x)
        
        # outputs should be different due to dropout
        assert not torch.allclose(output_train1, output_train2, atol=1e-5)
        
        # test eval mode (dropout inactive)
        model.eval()
        output_eval1 = model(x)
        output_eval2 = model(x)
        
        # outputs should be identical in eval mode
        assert torch.allclose(output_eval1, output_eval2)

    def test_no_dropout(self):
        """Test model without dropout."""
        model = TemporalMLP(
            input_size=6,
            num_hid_units=32,
            num_layers=2,
            dropout_rate=0.0,
        )
        
        x = torch.randn(2, 50, 6)
        
        # outputs should be identical even in training mode
        model.train()
        output1 = model(x)
        output2 = model(x)
        
        assert torch.allclose(output1, output2)

    def test_gradient_flow(self):
        """Test that gradients flow properly through the model."""
        model = TemporalMLP(
            input_size=6,
            num_hid_units=32,
            num_layers=2,
        )
        
        x = torch.randn(2, 50, 6, requires_grad=True)
        output = model(x)
        
        # compute dummy loss and backward pass
        loss = output.sum()
        loss.backward()
        
        # check that input gradients exist
        assert x.grad is not None
        assert x.grad.shape == x.shape
        
        # check that model parameters have gradients
        for param in model.parameters():
            assert param.grad is not None

    def test_different_batch_sizes(self):
        """Test model with different batch sizes."""
        model = TemporalMLP(
            input_size=4,
            num_hid_units=16,
            num_layers=1,
        )
        
        sequence_length, features = 50, 4
        
        for batch_size in [1, 2, 8, 16]:
            x = torch.randn(batch_size, sequence_length, features)
            output = model(x)
            
            expected_shape = (batch_size, sequence_length, 16)
            assert output.shape == expected_shape

    def test_different_sequence_lengths(self):
        """Test model with different sequence lengths."""
        model = TemporalMLP(
            input_size=4,
            num_hid_units=16,
            num_layers=1,
            num_lags=3,
        )
        
        batch_size, features = 2, 4
        
        for seq_len in [10, 50, 100, 500]:
            x = torch.randn(batch_size, seq_len, features)
            output = model(x)
            
            expected_shape = (batch_size, seq_len, 16)
            assert output.shape == expected_shape

    def test_repr(self):
        """Test string representation."""
        model = TemporalMLP(
            input_size=6,
            num_hid_units=32,
            num_layers=2,
            num_lags=5,
            activation='relu',
            dropout_rate=0.1,
        )
        
        repr_str = repr(model)
        
        # check that key parameters are in representation
        assert 'TemporalMLP(' in repr_str
        assert 'input_size=6' in repr_str
        assert 'num_hid_units=32' in repr_str
        assert 'num_layers=2' in repr_str
        assert 'num_lags=5' in repr_str
        assert 'activation=relu' in repr_str
        assert 'dropout_rate=0.1' in repr_str

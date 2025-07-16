"""Tests for Segmenter model with backbone integration."""

import pytest
import torch

from lightning_action.models import Segmenter


class TestSegmenter:
    """Test the Segmenter model class."""

    @pytest.fixture
    def backbone_configs(self):
        """Fixture providing different backbone configurations for testing."""
        return [
            {
                'backbone_type': 'temporalmlp',
                'config': {
                    'model': {
                        'input_size': 6,
                        'output_size': 4,
                        'sequence_length': 100,
                        'backbone': 'temporalmlp',
                        'num_hid_units': 32,
                        'num_layers': 2,
                        'n_lags': 3,
                        'activation': 'lrelu',
                        'dropout_rate': 0.1,
                        'seed': 42,
                    },
                    'optimizer': {
                        'type': 'Adam',
                        'lr': 1e-3,
                        'wd': 1e-4,
                    }
                }
            },
            {
                'backbone_type': 'rnn',
                'config': {
                    'model': {
                        'input_size': 6,
                        'output_size': 4,
                        'sequence_length': 100,
                        'backbone': 'rnn',
                        'num_hid_units': 32,
                        'num_layers': 1,
                        'rnn_type': 'lstm',
                        'bidirectional': False,
                        'dropout_rate': 0.1,
                        'seed': 42,
                    },
                    'optimizer': {
                        'type': 'Adam',
                        'lr': 1e-3,
                        'wd': 1e-4,
                    }
                }
            },
            {
                'backbone_type': 'rnn',
                'config': {
                    'model': {
                        'input_size': 6,
                        'output_size': 4,
                        'sequence_length': 150,
                        'backbone': 'rnn',
                        'num_hid_units': 48,
                        'num_layers': 2,
                        'rnn_type': 'gru',
                        'bidirectional': True,
                        'dropout_rate': 0.2,
                        'seed': 123,
                    },
                    'optimizer': {
                        'type': 'AdamW',
                        'lr': 2e-3,
                        'wd': 1e-3,
                    }
                }
            },
            {
                'backbone_type': 'dilatedtcn',
                'config': {
                    'model': {
                        'input_size': 6,
                        'output_size': 4,
                        'sequence_length': 100,
                        'backbone': 'dilatedtcn',
                        'num_hid_units': 32,
                        'num_layers': 3,
                        'n_lags': 2,
                        'activation': 'relu',
                        'dropout_rate': 0.2,
                        'seed': 42,
                    },
                    'optimizer': {
                        'type': 'Adam',
                        'lr': 1e-3,
                        'wd': 1e-4,
                    }
                }
            }
        ]

    @pytest.fixture
    def sample_batch(self):
        """Fixture providing sample batch data."""
        # these values match those in backbone_configs fixture
        batch_size, sequence_length, features, output_size = 2, 100, 6, 4
        return {
            'input': torch.randn(batch_size, sequence_length, features),
            'labels': torch.randint(0, 4, (batch_size, sequence_length, output_size)).double(),
            'dataset_id': ['test_dataset'] * batch_size,
            'batch_idx': torch.arange(batch_size),
        }

    def test_initialization(self, backbone_configs):
        """Test model initialization with different backbones."""
        for backbone_config in backbone_configs:
            config = backbone_config['config']
            
            # create model
            model = Segmenter(config)
            
            # check basic attributes
            assert model.input_size == config['model']['input_size']
            assert model.output_size == config['model']['output_size']
            assert model.sequence_length == config['model']['sequence_length']
            
            # check backbone exists
            assert hasattr(model, 'backbone')
            assert hasattr(model, 'classifier')
            
            # check metrics are initialized
            assert hasattr(model, 'train_accuracy')
            assert hasattr(model, 'train_f1')
            assert hasattr(model, 'val_accuracy')
            assert hasattr(model, 'val_f1')

    def test_forward_pass(self, backbone_configs, sample_batch):
        """Test forward pass with different backbones."""
        for backbone_config in backbone_configs:
            config = backbone_config['config']
            model = Segmenter(config)
            
            x = sample_batch['input']
            batch_size, sequence_length, features = x.shape
            
            # forward pass
            outputs = model(x)
            
            # check output dictionary structure
            assert isinstance(outputs, dict)
            assert 'logits' in outputs
            assert 'probabilities' in outputs
            assert 'features' in outputs
            
            # check output shapes
            expected_logits_shape = (
                batch_size, sequence_length, config['model']['output_size']
            )
            expected_probs_shape = (
                batch_size, sequence_length, config['model']['output_size']
            )
            expected_features_shape = (
                batch_size, sequence_length, config['model']['num_hid_units']
            )
            
            assert outputs['logits'].shape == expected_logits_shape
            assert outputs['probabilities'].shape == expected_probs_shape
            assert outputs['features'].shape == expected_features_shape
            
            # check probabilities sum to 1
            prob_sums = outputs['probabilities'].sum(dim=-1)
            expected_sums = torch.ones_like(prob_sums)
            assert torch.allclose(prob_sums, expected_sums, atol=1e-6)

    def test_compute_loss(self, backbone_configs, sample_batch):
        """Test loss computation with different backbones."""
        for backbone_config in backbone_configs:
            config = backbone_config['config']
            model = Segmenter(config)
            
            x = sample_batch['input']
            targets = sample_batch['labels']
            
            # forward pass
            outputs = model(x)
            
            # compute loss
            loss, metrics = model.compute_loss(outputs, targets, stage='train')
            
            # check loss
            assert isinstance(loss, torch.Tensor)
            assert loss.ndim == 0  # scalar
            assert loss.item() >= 0  # cross entropy is non-negative
            
            # check metrics
            assert isinstance(metrics, dict)
            expected_metrics = ['train_loss', 'train_accuracy', 'train_f1']
            for metric in expected_metrics:
                assert metric in metrics
                assert isinstance(metrics[metric], float)
                
            # check metric ranges
            assert 0.0 <= metrics['train_accuracy'] <= 1.0
            assert 0.0 <= metrics['train_f1'] <= 1.0

    def test_training_step(self, backbone_configs, sample_batch):
        """Test training step with different backbones."""
        for backbone_config in backbone_configs:
            config = backbone_config['config']
            model = Segmenter(config)
            
            # training step
            loss = model.training_step(sample_batch, batch_idx=0)
            
            # check loss
            assert isinstance(loss, torch.Tensor)
            assert loss.ndim == 0  # scalar
            assert loss.item() >= 0

    def test_validation_step(self, backbone_configs, sample_batch):
        """Test validation step with different backbones."""
        for backbone_config in backbone_configs:
            config = backbone_config['config']
            model = Segmenter(config)
            
            # validation step (should not raise error)
            result = model.validation_step(sample_batch, batch_idx=0)

            # validation step returns None
            assert result is None

    def test_predict_step(self, backbone_configs, sample_batch):
        """Test prediction step with different backbones."""
        for backbone_config in backbone_configs:
            config = backbone_config['config']
            model = Segmenter(config)
            
            # predict step
            predictions = model.predict_step(sample_batch, batch_idx=0)
            
            # check predictions structure
            assert isinstance(predictions, dict)
            assert 'logits' in predictions
            assert 'probabilities' in predictions
            assert 'predictions' in predictions
            
            x = sample_batch['input']
            batch_size, sequence_length = x.shape[:2]
            output_size = config['model']['output_size']
            
            # check prediction shapes
            assert predictions['logits'].shape == (batch_size, sequence_length, output_size)
            assert predictions['probabilities'].shape == (batch_size, sequence_length, output_size)
            assert predictions['predictions'].shape == (batch_size, sequence_length)
            
            # check prediction values are valid class indices
            assert predictions['predictions'].min() >= 0
            assert predictions['predictions'].max() < output_size

    def test_configure_optimizers(self, backbone_configs):
        """Test optimizer configuration with different backbones."""
        for backbone_config in backbone_configs:
            config = backbone_config['config']
            model = Segmenter(config)
            
            # test basic optimizer
            optimizer_config = model.configure_optimizers()
            
            if isinstance(optimizer_config, dict):
                # with scheduler
                assert 'optimizer' in optimizer_config
                assert 'lr_scheduler' in optimizer_config
                optimizer = optimizer_config['optimizer']
            else:
                # just optimizer
                optimizer = optimizer_config
            
            # check optimizer type
            expected_type = config['optimizer']['type']
            if expected_type.lower() == 'adam':
                assert isinstance(optimizer, torch.optim.Adam)
            elif expected_type.lower() == 'adamw':
                assert isinstance(optimizer, torch.optim.AdamW)
            
            # check learning rate
            expected_lr = config['optimizer']['lr']
            assert optimizer.param_groups[0]['lr'] == expected_lr

    def test_different_optimizer_types(self, backbone_configs):
        """Test different optimizer configurations."""
        base_config = backbone_configs[0]['config'].copy()
        
        optimizer_types = ['Adam', 'AdamW']
        
        for opt_type in optimizer_types:
            config = base_config.copy()
            config['optimizer']['type'] = opt_type
            
            model = Segmenter(config)
            optimizer_config = model.configure_optimizers()
            
            if isinstance(optimizer_config, dict):
                optimizer = optimizer_config['optimizer']
            else:
                optimizer = optimizer_config
            
            if opt_type.lower() == 'adam':
                assert isinstance(optimizer, torch.optim.Adam)
            elif opt_type.lower() == 'adamw':
                assert isinstance(optimizer, torch.optim.AdamW)

    def test_invalid_optimizer_type(self, backbone_configs):
        """Test invalid optimizer type raises error."""
        config = backbone_configs[0]['config'].copy()
        config['optimizer']['type'] = 'invalid_optimizer'
        
        model = Segmenter(config)
        
        with pytest.raises(ValueError, match='Unsupported optimizer type'):
            model.configure_optimizers()

    def test_gradient_flow(self, backbone_configs, sample_batch):
        """Test gradient flow through model with different backbones."""
        for backbone_config in backbone_configs:
            config = backbone_config['config']
            model = Segmenter(config)
            
            x = sample_batch['input']
            targets = sample_batch['labels']
            
            # forward pass
            outputs = model(x)
            loss, _ = model.compute_loss(outputs, targets)
            
            # backward pass
            loss.backward()
            
            # check that model parameters have gradients
            for name, param in model.named_parameters():
                assert param.grad is not None, f'No gradient for parameter {name}'
                assert not torch.isnan(param.grad).any(), f'NaN gradient for parameter {name}'

    def test_different_batch_sizes(self, backbone_configs):
        """Test model with different batch sizes."""
        for backbone_config in backbone_configs:
            config = backbone_config['config']
            model = Segmenter(config)
            
            sequence_length = config['model']['sequence_length']
            input_size = config['model']['input_size']
            output_size = config['model']['output_size']
            
            for batch_size in [1, 2, 4, 8]:
                x = torch.randn(batch_size, sequence_length, input_size)
                outputs = model(x)
                
                expected_shape = (batch_size, sequence_length, output_size)
                assert outputs['logits'].shape == expected_shape
                assert outputs['probabilities'].shape == expected_shape

    def test_different_sequence_lengths(self, backbone_configs):
        """Test model with different sequence lengths."""
        for backbone_config in backbone_configs:
            config = backbone_config['config']
            model = Segmenter(config)
            
            batch_size = 2
            input_size = config['model']['input_size']
            output_size = config['model']['output_size']
            
            for seq_len in [50, 100, 200, 500]:
                x = torch.randn(batch_size, seq_len, input_size)
                outputs = model(x)
                
                expected_shape = (batch_size, seq_len, output_size)
                assert outputs['logits'].shape == expected_shape
                assert outputs['probabilities'].shape == expected_shape

    def test_model_eval_mode(self, backbone_configs, sample_batch):
        """Test model behavior in eval mode."""
        for backbone_config in backbone_configs:
            config = backbone_config['config']
            model = Segmenter(config)
            
            x = sample_batch['input']
            
            # test in eval mode
            model.eval()
            
            with torch.no_grad():
                outputs1 = model(x)
                outputs2 = model(x)
            
            # outputs should be identical in eval mode (no dropout)
            assert torch.allclose(outputs1['logits'], outputs2['logits'])
            assert torch.allclose(outputs1['probabilities'], outputs2['probabilities'])

    def test_model_train_mode(self, backbone_configs, sample_batch):
        """Test model behavior in train mode with dropout."""
        for backbone_config in backbone_configs:
            config = backbone_config['config']
            # ensure dropout is enabled
            if 'dropout_rate' in config['model']:
                config['model']['dropout_rate'] = 0.5
            
            model = Segmenter(config)
            x = sample_batch['input']
            
            # test in train mode
            model.train()
            
            outputs1 = model(x)
            outputs2 = model(x)
            
            # outputs might be different due to dropout (depending on implementation)
            # just check that they have the right shape and are finite
            assert torch.isfinite(outputs1['logits']).all()
            assert torch.isfinite(outputs2['logits']).all()

    def test_unsupported_backbone_type(self):
        """Test that unsupported backbone type raises error."""
        config = {
            'model': {
                'input_size': 6,
                'output_size': 4,
                'backbone': 'unsupported_backbone',
                'num_hid_units': 32,
                'num_layers': 2,
            },
            'optimizer': {
                'type': 'Adam',
                'lr': 1e-3,
            }
        }
        
        with pytest.raises(ValueError, match='Unsupported backbone type'):
            Segmenter(config)

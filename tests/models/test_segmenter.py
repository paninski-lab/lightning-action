"""Tests for Segmenter model with head integration."""

import copy
from unittest.mock import patch

import pytest
import torch

from lightning_action.models import Segmenter


class TestSegmenter:
    """Test the Segmenter model class."""

    @pytest.fixture
    def head_configs(self):
        """Fixture providing different head configurations for testing."""
        return [
            {
                'head_type': 'temporalmlp',
                'config': {
                    'model': {
                        'input_size': 6,
                        'output_size': 4,
                        'sequence_length': 100,
                        'head': 'temporalmlp',
                        'num_hid_units': 32,
                        'num_layers': 2,
                        'num_lags': 3,
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
                'head_type': 'rnn',
                'config': {
                    'model': {
                        'input_size': 6,
                        'output_size': 4,
                        'sequence_length': 100,
                        'head': 'rnn',
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
                'head_type': 'rnn',
                'config': {
                    'model': {
                        'input_size': 6,
                        'output_size': 4,
                        'sequence_length': 150,
                        'head': 'rnn',
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
                'head_type': 'dilatedtcn',
                'config': {
                    'model': {
                        'input_size': 6,
                        'output_size': 4,
                        'sequence_length': 100,
                        'head': 'dilatedtcn',
                        'num_hid_units': 32,
                        'num_layers': 3,
                        'num_lags': 2,
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
        # these values match those in head_configs fixture
        batch_size, sequence_length, features, output_size = 2, 100, 6, 4
        return {
            'input': torch.randn(batch_size, sequence_length, features),
            'labels': torch.randint(0, 4, (batch_size, sequence_length, output_size)).double(),
            'dataset_id': ['test_dataset'] * batch_size,
            'batch_idx': torch.arange(batch_size),
        }

    def test_initialization(self, head_configs):
        """Test model initialization with different heads."""
        for head_config in head_configs:
            config = head_config['config']

            # create model
            model = Segmenter(config)

            # check basic attributes
            assert model.input_size == config['model']['input_size']
            assert model.output_size == config['model']['output_size']
            assert model.sequence_length == config['model']['sequence_length']

            # check head exists
            assert hasattr(model, 'head')
            assert hasattr(model, 'classifier')

            # check metrics are initialized
            assert hasattr(model, 'train_accuracy')
            assert hasattr(model, 'train_f1')
            assert hasattr(model, 'val_accuracy')
            assert hasattr(model, 'val_f1')

    def test_remove_padding(self, head_configs, sample_batch):
        """Test the removal of padding from batch"""
        # remove non-data arrays
        del sample_batch['dataset_id']
        del sample_batch['batch_idx']

        for head_config in head_configs:
            config = head_config['config']
            model = Segmenter(config)
            unpadded_len = 100 - 2 * model.sequence_pad
            # test dict operation
            batch_no_pad = model._remove_padding(copy.copy(sample_batch))
            for _, val in batch_no_pad.items():
                assert val.shape[1] == unpadded_len
            # test array operation
            array_no_pad = model._remove_padding(torch.randn(2, 100, 6))
            assert array_no_pad.shape[1] == unpadded_len

    def test_forward_pass(self, head_configs, sample_batch):
        """Test forward pass with different heads."""
        for head_config in head_configs:
            config = head_config['config']
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

    def test_compute_loss(self, head_configs, sample_batch):
        """Test loss computation with different heads."""
        for head_config in head_configs:
            config = head_config['config']
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

    def test_compute_loss_with_one_hot_targets(self, head_configs):
        """Test compute_loss accepts one-hot encoded targets."""
        config = head_configs[0]['config']
        model = Segmenter(config)

        batch_size = 2
        sequence_length = 100
        output_size = config['model']['output_size']

        # Create input and run forward pass
        x = torch.randn(batch_size, sequence_length, config['model']['input_size'])
        outputs = model(x)

        # Create one-hot encoded targets: shape (batch, sequence, num_classes)
        class_indices = torch.randint(0, output_size, (batch_size, sequence_length))
        one_hot_targets = torch.zeros(batch_size, sequence_length, output_size)
        one_hot_targets.scatter_(2, class_indices.unsqueeze(-1), 1.0)

        # Compute loss with one-hot targets
        loss, metrics = model.compute_loss(outputs, one_hot_targets, stage='train')

        # Verify loss is valid
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() >= 0
        assert not torch.isnan(loss)

        # Verify metrics
        assert 'train_loss' in metrics
        assert 'train_accuracy' in metrics
        assert 'train_f1' in metrics

    def test_compute_loss_with_class_index_targets(self, head_configs):
        """Test compute_loss accepts class index targets."""
        config = head_configs[0]['config']
        model = Segmenter(config)

        batch_size = 2
        sequence_length = 100
        output_size = config['model']['output_size']

        # Create input and run forward pass
        x = torch.randn(batch_size, sequence_length, config['model']['input_size'])
        outputs = model(x)

        # Create class index targets: shape (batch, sequence)
        class_index_targets = torch.randint(0, output_size, (batch_size, sequence_length))

        # Compute loss with class index targets
        loss, metrics = model.compute_loss(outputs, class_index_targets, stage='train')

        # Verify loss is valid
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() >= 0
        assert not torch.isnan(loss)

        # Verify metrics
        assert 'train_loss' in metrics
        assert 'train_accuracy' in metrics
        assert 'train_f1' in metrics

    def test_compute_loss_one_hot_vs_class_index_equivalence(self, head_configs):
        """Test that one-hot and class index targets produce the same loss."""
        config = head_configs[0]['config']
        model = Segmenter(config)
        model.eval()  # Ensure deterministic behavior

        batch_size = 2
        sequence_length = 100
        output_size = config['model']['output_size']

        # Create input and run forward pass
        torch.manual_seed(42)
        x = torch.randn(batch_size, sequence_length, config['model']['input_size'])
        outputs = model(x)

        # Create class index targets
        class_indices = torch.randint(0, output_size, (batch_size, sequence_length))

        # Create equivalent one-hot targets
        one_hot_targets = torch.zeros(batch_size, sequence_length, output_size)
        one_hot_targets.scatter_(2, class_indices.unsqueeze(-1), 1.0)

        # Compute loss with both target formats
        loss_class_idx, _ = model.compute_loss(outputs, class_indices, stage='train')

        # Reset metrics to avoid accumulation effects
        model.train_accuracy.reset()
        model.train_f1.reset()

        loss_one_hot, _ = model.compute_loss(outputs, one_hot_targets, stage='train')

        # Losses should be identical
        assert torch.allclose(loss_class_idx, loss_one_hot, atol=1e-6)

    def test_compute_loss_all_targets_ignore_index(self, head_configs):
        """Test compute_loss when all targets are ignore_index."""
        config = head_configs[0]['config']
        config['data'] = {'ignore_index': -100}
        model = Segmenter(config)

        batch_size = 2
        sequence_length = 100

        # Create input and run forward pass
        x = torch.randn(batch_size, sequence_length, config['model']['input_size'])
        outputs = model(x)

        # Create targets where ALL values are ignore_index
        ignore_index = model.ignore_index
        all_ignored_targets = torch.full(
            (batch_size, sequence_length),
            ignore_index,
            dtype=torch.long
        )

        # Compute loss
        loss, metrics = model.compute_loss(outputs, all_ignored_targets, stage='train')

        # Loss should be 0 (no valid targets to compute loss on)
        assert loss.item() == 0.0

        # Metrics should indicate the special case
        assert metrics['train_loss'] == 0.0
        # Accuracy and F1 should be NaN when all ignored
        # NaN check
        assert 'train_accuracy' not in metrics or metrics['train_accuracy'] != metrics['train_accuracy']
        assert 'train_f1' not in metrics or metrics['train_f1'] != metrics['train_f1']  # NaN check

    def test_compute_loss_partial_ignore_index(self, head_configs):
        """Test compute_loss with some targets as ignore_index."""
        config = head_configs[0]['config']
        config['data'] = {'ignore_index': -100}
        model = Segmenter(config)

        batch_size = 2
        sequence_length = 100
        output_size = config['model']['output_size']

        # Create input and run forward pass
        x = torch.randn(batch_size, sequence_length, config['model']['input_size'])
        outputs = model(x)

        # Create targets with mix of valid and ignored values
        targets = torch.randint(0, output_size, (batch_size, sequence_length))
        # Set first 10 and last 10 positions to ignore_index
        targets[:, :10] = model.ignore_index
        targets[:, -10:] = model.ignore_index

        # Compute loss
        loss, metrics = model.compute_loss(outputs, targets, stage='train')

        # Loss should be valid (computed on non-ignored targets)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
        assert not torch.isnan(loss)

        # Metrics should be present
        assert 'train_loss' in metrics
        assert 'train_accuracy' in metrics
        assert 'train_f1' in metrics

    def test_compute_loss_with_class_weights(self, head_configs):
        """Test compute_loss with class weights."""
        config = head_configs[0]['config']
        output_size = config['model']['output_size']

        # Add class weights to config
        class_weights = [1.0, 2.0, 0.5, 1.5]  # 4 classes
        config['model']['class_weights'] = class_weights

        model = Segmenter(config)

        batch_size = 2
        sequence_length = 100

        # Create input and run forward pass
        x = torch.randn(batch_size, sequence_length, config['model']['input_size'])
        outputs = model(x)

        # Create targets
        targets = torch.randint(0, output_size, (batch_size, sequence_length))

        # Compute loss with class weights
        loss, metrics = model.compute_loss(outputs, targets, stage='train')

        # Loss should be valid
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
        assert not torch.isnan(loss)

    def test_compute_loss_validation_stage(self, head_configs):
        """Test compute_loss with validation stage."""
        config = head_configs[0]['config']
        model = Segmenter(config)

        batch_size = 2
        sequence_length = 100
        output_size = config['model']['output_size']

        # Create input and run forward pass
        x = torch.randn(batch_size, sequence_length, config['model']['input_size'])
        outputs = model(x)
        targets = torch.randint(0, output_size, (batch_size, sequence_length))

        # Compute loss with val stage
        loss, metrics = model.compute_loss(outputs, targets, stage='val')

        # Check metrics have val_ prefix
        assert 'val_loss' in metrics
        assert 'val_accuracy' in metrics
        assert 'val_f1' in metrics

        # Ensure train_ metrics are not present
        assert 'train_loss' not in metrics
        assert 'train_accuracy' not in metrics
        assert 'train_f1' not in metrics

    def test_training_step(self, head_configs, sample_batch):
        """Test training step with different heads."""
        for head_config in head_configs:
            config = head_config['config']
            model = Segmenter(config)

            # training step
            loss = model.training_step(sample_batch, batch_idx=0)

            # check loss
            assert isinstance(loss, torch.Tensor)
            assert loss.ndim == 0  # scalar
            assert loss.item() >= 0

    def test_validation_step(self, head_configs, sample_batch):
        """Test validation step with different heads."""
        for head_config in head_configs:
            config = head_config['config']
            model = Segmenter(config)

            # validation step (should not raise error)
            result = model.validation_step(sample_batch, batch_idx=0)

            # validation step returns None
            assert result is None

    def test_padding_removal_in_training_step(self, head_configs):
        """Test that padding is correctly removed before computing loss in training_step.

        This test verifies the bug fix where _remove_padding() is called before
        compute_loss() in both training_step() and validation_step(). We use
        Python's unittest.mock.patch to "spy" on the compute_loss method and
        capture the tensor shapes it receives.
        """
        for head_config in head_configs:
            config = head_config['config']
            model = Segmenter(config)

            # Create a batch with known dimensions
            sequence_length = 100
            batch_size = 2
            batch = {
                'input': torch.randn(batch_size, sequence_length, config['model']['input_size']),
                'labels': torch.randint(
                    0, 4, (batch_size, sequence_length, config['model']['output_size'])
                ).double(),
            }

            # --- PATCHING EXPLANATION ---
            # We want to verify that compute_loss receives tensors with padding removed.
            # To do this without modifying the actual method, we use Python's "patch" mechanism.
            #
            # Step 1: Save a reference to the original compute_loss method
            original_compute_loss = model.compute_loss

            # Step 2: Create a dictionary to store captured information
            # This will be populated by our "spy" function below
            captured_shapes = {}

            # Step 3: Create a "spy" function that wraps the original method
            # This function will be called INSTEAD of compute_loss during the test
            def spy_compute_loss(outputs, targets, stage='train'):
                """A wrapper function that captures tensor shapes then calls the original method.

                This is called a "spy" because it observes what's happening without
                changing the behavior - it records information and then delegates to
                the real implementation.
                """
                # Capture the shapes of the tensors passed to compute_loss
                # The key thing we're checking: do these shapes reflect padding removal?
                captured_shapes['outputs_shape'] = outputs['logits'].shape
                captured_shapes['targets_shape'] = targets.shape

                # Call the original compute_loss method to maintain normal behavior
                # This ensures training_step still returns a valid loss
                return original_compute_loss(outputs, targets, stage)

            # Step 4: Use patch.object as a context manager
            # - patch.object(model, 'compute_loss', ...): temporarily replace model.compute_loss
            # - side_effect=spy_compute_loss: when compute_loss is called, run spy function instead
            # - The 'with' block ensures the patch is only active during this scope
            #   and automatically restores the original method when the block exits
            with patch.object(model, 'compute_loss', side_effect=spy_compute_loss):
                # Inside this block, any call to model.compute_loss will trigger spy_compute_loss
                # Our spy will capture the shapes at the compute_loss call
                model.training_step(batch, batch_idx=0)

            # Now the patch is deactivated and captured_shapes contains the recorded information

            # --- VERIFICATION ---
            # If _remove_padding() was called correctly, the sequence dimension should be reduced
            # Original sequence_length: 100
            # After padding removal: 100 - 2*sequence_pad (remove from both start and end)
            expected_seq_len = sequence_length - 2 * model.sequence_pad

            # Check that both outputs and targets had padding removed
            assert captured_shapes['outputs_shape'][1] == expected_seq_len, \
                f"Expected sequence length {expected_seq_len} after padding removal, " \
                f"but got {captured_shapes['outputs_shape'][1]}"

            assert captured_shapes['targets_shape'][1] == expected_seq_len, \
                f"Expected sequence length {expected_seq_len} after padding removal, " \
                f"but got {captured_shapes['targets_shape'][1]}"

    def test_padding_removal_in_validation_step(self, head_configs):
        """Test that padding is correctly removed before computing loss in validation_step.

        This is the same test as above but for validation_step. We separate them
        because they are different code paths that both need to handle padding correctly.
        """
        for head_config in head_configs:
            config = head_config['config']
            model = Segmenter(config)

            # Skip heads without padding
            if model.sequence_pad == 0:
                continue

            # Create test batch
            sequence_length = 100
            batch_size = 2
            batch = {
                'input': torch.randn(batch_size, sequence_length, config['model']['input_size']),
                'labels': torch.randint(
                    0, 4, (batch_size, sequence_length, config['model']['output_size'])
                ).double(),
            }

            # Save original method
            original_compute_loss = model.compute_loss

            # Dictionary to capture information
            captured_shapes = {}

            # Spy function to capture shapes
            def spy_compute_loss(outputs, targets, stage='val'):
                captured_shapes['outputs_shape'] = outputs['logits'].shape
                captured_shapes['targets_shape'] = targets.shape
                return original_compute_loss(outputs, targets, stage)

            # Temporarily replace compute_loss with our spy
            with patch.object(model, 'compute_loss', side_effect=spy_compute_loss):
                model.validation_step(batch, batch_idx=0)

            # Verify padding was removed
            expected_seq_len = sequence_length - 2 * model.sequence_pad

            assert captured_shapes['outputs_shape'][1] == expected_seq_len, \
                f"Expected sequence length {expected_seq_len} after padding removal, " \
                f"but got {captured_shapes['outputs_shape'][1]}"

            assert captured_shapes['targets_shape'][1] == expected_seq_len, \
                f"Expected sequence length {expected_seq_len} after padding removal, " \
                f"but got {captured_shapes['targets_shape'][1]}"

    def test_predict_step(self, head_configs, sample_batch):
        """Test prediction step with different heads."""
        for head_config in head_configs:
            config = head_config['config']
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
            seq_len_no_pad = sequence_length - 2 * model.sequence_pad
            assert predictions['logits'].shape == (batch_size, seq_len_no_pad, output_size)
            assert predictions['probabilities'].shape == (batch_size, seq_len_no_pad, output_size)
            assert predictions['predictions'].shape == (batch_size, seq_len_no_pad)

            # check prediction values are valid class indices
            assert predictions['predictions'].min() >= 0
            assert predictions['predictions'].max() < output_size

    def test_configure_optimizers(self, head_configs):
        """Test optimizer configuration with different heads."""
        for head_config in head_configs:
            config = head_config['config']
            model = Segmenter(config)

            # test basic optimizer
            optimizer_config = model.configure_optimizers()

            if isinstance(optimizer_config, dict):
                assert 'optimizer' in optimizer_config
                # lr_scheduler is optional - only check if present
                if 'lr_scheduler' in optimizer_config:
                    assert isinstance(optimizer_config['lr_scheduler'], dict)
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

    def test_different_optimizer_types(self, head_configs):
        """Test different optimizer configurations."""
        base_config = copy.deepcopy(head_configs[0]['config'])

        optimizer_types = [
            ('Adam', torch.optim.Adam),
            ('AdamW', torch.optim.AdamW),
            ('SGD', torch.optim.SGD),
        ]

        for opt_type, expected_class in optimizer_types:
            config = copy.deepcopy(base_config)
            config['optimizer']['type'] = opt_type

            # Add momentum for SGD
            if opt_type == 'SGD':
                config['optimizer']['momentum'] = 0.9

            model = Segmenter(config)
            optimizer_config = model.configure_optimizers()

            if isinstance(optimizer_config, dict):
                optimizer = optimizer_config['optimizer']
            else:
                optimizer = optimizer_config

            assert isinstance(optimizer, expected_class), \
                f"Expected {expected_class.__name__} but got {type(optimizer).__name__}"

    def test_sgd_optimizer_with_momentum(self, head_configs):
        """Test SGD optimizer configuration with momentum."""
        config = copy.deepcopy(head_configs[0]['config'])
        config['optimizer']['type'] = 'SGD'
        config['optimizer']['momentum'] = 0.95
        config['optimizer']['lr'] = 0.01

        model = Segmenter(config)
        optimizer_config = model.configure_optimizers()
        optimizer = optimizer_config['optimizer']

        assert isinstance(optimizer, torch.optim.SGD)
        # Check momentum is set correctly
        assert optimizer.param_groups[0]['momentum'] == 0.95
        assert optimizer.param_groups[0]['lr'] == 0.01

    def test_sgd_default_momentum(self, head_configs):
        """Test SGD optimizer uses default momentum when not specified."""
        config = copy.deepcopy(head_configs[0]['config'])
        config['optimizer']['type'] = 'SGD'
        # Don't specify momentum - should use default of 0.9

        model = Segmenter(config)
        optimizer_config = model.configure_optimizers()
        optimizer = optimizer_config['optimizer']

        assert isinstance(optimizer, torch.optim.SGD)
        assert optimizer.param_groups[0]['momentum'] == 0.9  # default value

    def test_invalid_optimizer_type(self, head_configs):
        """Test invalid optimizer type raises error."""
        config = copy.deepcopy(head_configs[0]['config'])
        config['optimizer']['type'] = 'invalid_optimizer'

        model = Segmenter(config)

        with pytest.raises(ValueError, match='Unsupported optimizer type'):
            model.configure_optimizers()

    # =========================================================================
    # Scheduler Tests
    # =========================================================================

    def test_scheduler_step_lr(self, head_configs):
        """Test StepLR scheduler configuration."""
        config = copy.deepcopy(head_configs[0]['config'])
        config['optimizer']['scheduler'] = {
            'use_scheduler': True,
            'type': 'step',
            'step_size': 10,
            'gamma': 0.5,
        }

        model = Segmenter(config)
        optimizer_config = model.configure_optimizers()

        assert 'lr_scheduler' in optimizer_config
        scheduler_config = optimizer_config['lr_scheduler']
        scheduler = scheduler_config['scheduler']

        assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)
        assert scheduler.step_size == 10
        assert scheduler.gamma == 0.5

    def test_scheduler_cosine_annealing(self, head_configs):
        """Test CosineAnnealingLR scheduler configuration."""
        config = copy.deepcopy(head_configs[0]['config'])
        config['optimizer']['scheduler'] = {
            'use_scheduler': True,
            'type': 'cosine',
            'T_max': 50,
            'eta_min_factor': 10,
        }

        model = Segmenter(config)
        optimizer_config = model.configure_optimizers()

        assert 'lr_scheduler' in optimizer_config
        scheduler = optimizer_config['lr_scheduler']['scheduler']

        assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
        assert scheduler.T_max == 50
        # eta_min should be lr / eta_min_factor
        expected_eta_min = config['optimizer']['lr'] / 10
        assert scheduler.eta_min == expected_eta_min

    def test_scheduler_cosine_warm_restarts(self, head_configs):
        """Test CosineAnnealingWarmRestarts scheduler configuration."""
        config = copy.deepcopy(head_configs[0]['config'])
        config['optimizer']['scheduler'] = {
            'use_scheduler': True,
            'type': 'cosine_warm_restarts',
            'T_0': 20,
            'T_mult': 2,
            'eta_min_factor': 20,
        }

        model = Segmenter(config)
        optimizer_config = model.configure_optimizers()

        assert 'lr_scheduler' in optimizer_config
        scheduler = optimizer_config['lr_scheduler']['scheduler']

        assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts)
        assert scheduler.T_0 == 20
        assert scheduler.T_mult == 2

    def test_scheduler_cosine_warm_restarts_alternate_name(self, head_configs):
        """Test CosineAnnealingWarmRestarts with alternate naming."""
        config = copy.deepcopy(head_configs[0]['config'])
        config['optimizer']['scheduler'] = {
            'use_scheduler': True,
            'type': 'cosineannealingwarmrestarts',  # alternate naming
            'T_0': 15,
            'T_mult': 3,
        }

        model = Segmenter(config)
        optimizer_config = model.configure_optimizers()

        scheduler = optimizer_config['lr_scheduler']['scheduler']
        assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts)

    def test_scheduler_reduce_on_plateau(self, head_configs):
        """Test ReduceLROnPlateau scheduler configuration."""
        config = copy.deepcopy(head_configs[0]['config'])
        config['optimizer']['scheduler'] = {
            'use_scheduler': True,
            'type': 'reduce_on_plateau',
            'factor': 0.25,
            'patience': 5,
        }

        model = Segmenter(config)
        optimizer_config = model.configure_optimizers()

        assert 'lr_scheduler' in optimizer_config
        scheduler_config = optimizer_config['lr_scheduler']
        scheduler = scheduler_config['scheduler']

        assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
        assert scheduler.factor == 0.25
        assert scheduler.patience == 5
        # ReduceLROnPlateau should have monitor set
        assert scheduler_config['monitor'] == 'val_loss'

    def test_scheduler_reduce_on_plateau_alternate_name(self, head_configs):
        """Test ReduceLROnPlateau with alternate naming."""
        config = copy.deepcopy(head_configs[0]['config'])
        config['optimizer']['scheduler'] = {
            'use_scheduler': True,
            'type': 'reducelronplateau',  # alternate naming
            'factor': 0.5,
            'patience': 10,
        }

        model = Segmenter(config)
        optimizer_config = model.configure_optimizers()

        scheduler = optimizer_config['lr_scheduler']['scheduler']
        assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

    def test_scheduler_disabled(self, head_configs):
        """Test scheduler is disabled when use_scheduler is False."""
        config = copy.deepcopy(head_configs[0]['config'])
        config['optimizer']['scheduler'] = {
            'use_scheduler': False,
            'type': 'cosine',
        }

        model = Segmenter(config)
        optimizer_config = model.configure_optimizers()

        # Should only have optimizer, no scheduler
        assert 'optimizer' in optimizer_config
        assert 'lr_scheduler' not in optimizer_config

    def test_scheduler_none(self, head_configs):
        """Test no scheduler when scheduler config is None."""
        config = copy.deepcopy(head_configs[0]['config'])
        # Ensure no scheduler key
        if 'scheduler' in config['optimizer']:
            del config['optimizer']['scheduler']

        model = Segmenter(config)
        optimizer_config = model.configure_optimizers()

        assert 'optimizer' in optimizer_config
        assert 'lr_scheduler' not in optimizer_config

    def test_scheduler_flat_specification(self, head_configs):
        """Test scheduler with flat string specification (scheduler: 'cosine')."""
        config = copy.deepcopy(head_configs[0]['config'])
        # Use flat specification instead of nested dict
        config['optimizer']['scheduler'] = 'cosine'

        model = Segmenter(config)
        optimizer_config = model.configure_optimizers()

        assert 'lr_scheduler' in optimizer_config
        scheduler = optimizer_config['lr_scheduler']['scheduler']
        assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_scheduler_flat_specification_step(self, head_configs):
        """Test scheduler with flat string specification for StepLR."""
        config = copy.deepcopy(head_configs[0]['config'])
        config['optimizer']['scheduler'] = 'step'

        model = Segmenter(config)
        optimizer_config = model.configure_optimizers()

        scheduler = optimizer_config['lr_scheduler']['scheduler']
        assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)
        # Should use default values
        assert scheduler.step_size == 30  # default
        assert scheduler.gamma == 0.1  # default

    def test_scheduler_invalid_type(self, head_configs):
        """Test invalid scheduler type raises error."""
        config = copy.deepcopy(head_configs[0]['config'])
        config['optimizer']['scheduler'] = {
            'use_scheduler': True,
            'type': 'invalid_scheduler',
        }

        model = Segmenter(config)

        with pytest.raises(ValueError, match='Unsupported scheduler type'):
            model.configure_optimizers()

    def test_scheduler_T_max_from_optimizer_config(self, head_configs):
        """Test T_max can be specified at optimizer level for cosine scheduler."""
        config = copy.deepcopy(head_configs[0]['config'])
        config['optimizer']['T_max'] = 200  # At optimizer level, not scheduler level
        config['optimizer']['scheduler'] = {
            'use_scheduler': True,
            'type': 'cosine',
            # T_max not specified here, should use from optimizer level
        }

        model = Segmenter(config)
        optimizer_config = model.configure_optimizers()

        scheduler = optimizer_config['lr_scheduler']['scheduler']
        assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
        assert scheduler.T_max == 200

    def test_scheduler_defaults(self, head_configs):
        """Test scheduler uses sensible defaults when minimal config provided."""
        config = copy.deepcopy(head_configs[0]['config'])
        config['optimizer']['scheduler'] = {
            'use_scheduler': True,
            'type': 'step',
            # No step_size or gamma specified
        }

        model = Segmenter(config)
        optimizer_config = model.configure_optimizers()

        scheduler = optimizer_config['lr_scheduler']['scheduler']
        # Should use defaults
        assert scheduler.step_size == 30
        assert scheduler.gamma == 0.1

    def test_scheduler_config_structure(self, head_configs):
        """Test scheduler config returns proper Lightning structure."""
        config = copy.deepcopy(head_configs[0]['config'])
        config['optimizer']['scheduler'] = {
            'use_scheduler': True,
            'type': 'cosine',
            'T_max': 100,
        }

        model = Segmenter(config)
        optimizer_config = model.configure_optimizers()

        # Check structure expected by Lightning
        assert 'optimizer' in optimizer_config
        assert 'lr_scheduler' in optimizer_config

        lr_scheduler_config = optimizer_config['lr_scheduler']
        assert 'scheduler' in lr_scheduler_config
        assert 'monitor' in lr_scheduler_config
        assert 'interval' in lr_scheduler_config
        assert 'frequency' in lr_scheduler_config

        assert lr_scheduler_config['interval'] == 'epoch'
        assert lr_scheduler_config['frequency'] == 1

    def test_gradient_flow(self, head_configs, sample_batch):
        """Test gradient flow through model with different heads."""
        for head_config in head_configs:
            config = head_config['config']
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

    def test_different_batch_sizes(self, head_configs):
        """Test model with different batch sizes."""
        for head_config in head_configs:
            config = head_config['config']
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

    def test_different_sequence_lengths(self, head_configs):
        """Test model with different sequence lengths."""
        for head_config in head_configs:
            config = head_config['config']
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

    def test_model_eval_mode(self, head_configs, sample_batch):
        """Test model behavior in eval mode."""
        for head_config in head_configs:
            config = head_config['config']
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

    def test_model_train_mode(self, head_configs, sample_batch):
        """Test model behavior in train mode with dropout."""
        for head_config in head_configs:
            config = head_config['config']
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

    def test_unsupported_head_type(self):
        """Test that unsupported head type raises error."""
        config = {
            'model': {
                'input_size': 6,
                'output_size': 4,
                'head': 'unsupported_head',
                'num_hid_units': 32,
                'num_layers': 2,
            },
            'optimizer': {
                'type': 'Adam',
                'lr': 1e-3,
            }
        }

        with pytest.raises(ValueError, match='Unsupported head type'):
            Segmenter(config)

    def test_train_accuracy_metric(self, head_configs):
        """Test train_accuracy metric computation."""
        config = head_configs[0]['config']
        model = Segmenter(config)

        # create sample predictions and targets
        predictions = torch.tensor([
            [0, 1, 2, 3, 0, 1, 2, 3, 0, 1],  # batch 1
            [1, 2, 3, 0, 1, 2, 3, 0, 1, 2],  # batch 2
        ])

        targets = torch.tensor([
            [0, 1, 2, 3, 0, 1, 1, 3, 0, 1],  # batch 1 - 1 wrong (index 6)
            [1, 2, 3, 0, 1, 2, 3, 0, 1, 2],  # batch 2 - all correct
        ])

        # compute accuracy
        accuracy = model.train_accuracy(predictions, targets)

        # manually calculate expected accuracy
        # batch 1: 9/10 correct, batch 2: 10/10 correct
        # total: 19/20 = 0.95
        expected_accuracy = 19.0 / 20.0

        assert torch.allclose(accuracy, torch.tensor(expected_accuracy))

    def test_train_f1_metric(self, head_configs):
        """Test train_f1 metric computation."""
        config = head_configs[0]['config']
        model = Segmenter(config)

        # create sample predictions and targets with known F1 characteristics
        predictions = torch.tensor([
            [0, 0, 1, 1, 2, 2, 3, 3, 0, 1, 2, 3],  # batch 1
            [0, 1, 1, 2, 2, 3, 3, 0, 1, 2, 3, 0],  # batch 2
        ])

        targets = torch.tensor([
            [0, 0, 1, 1, 2, 2, 3, 3, 0, 1, 2, 3],  # batch 1 - all correct
            [0, 1, 2, 2, 2, 3, 3, 0, 1, 2, 3, 0],  # batch 2 - some wrong
        ])

        # compute F1 score
        f1 = model.train_f1(predictions, targets)

        # F1 should be between 0 and 1, and should be high since most predictions are correct
        assert 0.0 <= f1 <= 1.0
        assert f1 > 0.8  # should be quite high given the mostly correct predictions

    def test_train_accuracy_with_ignore_index(self, head_configs):
        """Test train_accuracy metric with ignore_index functionality."""
        config = copy.deepcopy(head_configs[0]['config'])
        # set ignore_index in data config
        config['data'] = {'ignore_index': 0}
        model = Segmenter(config)

        # create predictions and targets where class 0 should be ignored
        predictions = torch.tensor([
            [0, 1, 2, 3, 0, 1, 2, 3],  # batch 1
            [1, 2, 3, 0, 1, 2, 3, 0],  # batch 2
        ])

        targets = torch.tensor([
            [0, 1, 2, 3, 0, 1, 1, 3],  # batch 1 - class 0 ignored, 1 wrong at index 6
            [1, 2, 3, 0, 1, 2, 3, 0],  # batch 2 - class 0 ignored, all non-ignored correct
        ])

        # compute accuracy
        accuracy = model.train_accuracy(predictions, targets)

        # class 0 should be ignored, so we only count non-zero predictions
        # batch 1: positions 1,2,3,5,6,7 -> 5/6 correct (position 6 is wrong)
        # batch 2: positions 0,1,2,4,5,6 -> 6/6 correct
        # total: 11/12 ≈ 0.9167
        expected_accuracy = 11.0 / 12.0

        assert torch.allclose(accuracy, torch.tensor(expected_accuracy), atol=1e-3)

    def test_train_f1_with_ignore_index(self, head_configs):
        """Test train_f1 metric with ignore_index functionality."""
        config = copy.deepcopy(head_configs[0]['config'])
        # set ignore_index in data config
        config['data'] = {'ignore_index': 0}
        model = Segmenter(config)

        # create predictions and targets where class 0 should be ignored
        predictions = torch.tensor([
            [0, 1, 2, 3, 0, 1, 2, 3],  # batch 1
            [1, 2, 3, 0, 1, 2, 3, 0],  # batch 2
        ])

        targets = torch.tensor([
            [0, 1, 2, 3, 0, 1, 1, 3],  # batch 1 - class 0 ignored, 1 wrong at index 6
            [1, 2, 3, 0, 1, 2, 3, 0],  # batch 2 - class 0 ignored, all non-ignored correct
        ])

        # compute F1 score
        f1 = model.train_f1(predictions, targets)

        # F1 should be in [0, 1], should be high since most non-ignored predictions are correct
        assert 0.0 <= f1 <= 1.0
        assert f1 > 0.8  # should be quite high given the mostly correct predictions

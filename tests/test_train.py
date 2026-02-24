"""Tests for training functionality.

This module tests both the CSV pipeline training (train.py) and the shared
training utilities (train_utils.py).
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from lightning_action.data import DataModule
from lightning_action.models.segmenter import Segmenter

# Also test that train.py re-exports work for backward compatibility
from lightning_action.train import (
    build_data_config_from_path,
    compute_class_weights,
    train,
)

# Import shared utilities (now in train_utils.py, re-exported from train.py)
from lightning_action.train_utils import (
    get_callbacks,
    get_callbacks_from_config,
    pretty_print_config,
    reset_seeds,
    save_config,
    update_config_with_class_weights,
    update_config_with_label_names,
    validate_config,
)

# =============================================================================
# Tests for Shared Utilities (train_utils.py)
# =============================================================================


class TestResetSeeds:
    """Test the reset_seeds function."""

    def test_reset_seeds_with_default(self):
        """Test reset_seeds with default seed value."""
        reset_seeds()

        # Check that seeds are set
        assert os.environ.get("PYTHONHASHSEED") == "0"

        # Check torch backends settings
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False

    def test_reset_seeds_with_custom_seed(self):
        """Test reset_seeds with custom seed value."""
        custom_seed = 42
        reset_seeds(seed=custom_seed)

        # Check environment variable
        assert os.environ.get("PYTHONHASHSEED") == str(custom_seed)

    def test_deterministic_behavior(self):
        """Test that reset_seeds provides deterministic behavior."""
        seed = 123

        # Reset seeds and generate random numbers
        reset_seeds(seed=seed)
        torch_val1 = torch.randn(1).item()
        np_val1 = np.random.random()

        # Reset seeds again and generate random numbers
        reset_seeds(seed=seed)
        torch_val2 = torch.randn(1).item()
        np_val2 = np.random.random()

        # Values should be identical
        assert torch_val1 == torch_val2
        assert np_val1 == np_val2


class TestValidateConfig:
    """Test the validate_config function."""

    def test_valid_config(self):
        """Test validation with all required sections present."""
        config = {'data': {}, 'training': {}, 'model': {}}
        # Should not raise
        validate_config(config, required_sections=['data', 'training'])

    def test_missing_data_section(self):
        """Test validation with missing data section."""
        config = {'training': {}, 'model': {}}
        with pytest.raises(ValueError, match="Configuration must contain 'data' section"):
            validate_config(config, required_sections=['data', 'training'])

    def test_missing_training_section(self):
        """Test validation with missing training section."""
        config = {'data': {}, 'model': {}}
        with pytest.raises(ValueError, match="Configuration must contain 'training' section"):
            validate_config(config, required_sections=['data', 'training'])

    def test_custom_required_sections(self):
        """Test validation with custom required sections."""
        config = {'custom': {}}
        with pytest.raises(ValueError, match="Configuration must contain 'other' section"):
            validate_config(config, required_sections=['custom', 'other'])

    def test_empty_required_sections(self):
        """Test validation with empty required sections list."""
        config = {}
        # Should not raise
        validate_config(config, required_sections=[])


class TestUpdateConfigWithClassWeights:
    """Test the update_config_with_class_weights function."""

    def test_adds_weights_to_config(self):
        """Test that class weights are added to config."""
        config = {}
        model = MagicMock()
        model.config = {}
        weights = [1.0, 2.0, 1.5]

        update_config_with_class_weights(config, model, weights)

        assert config['model']['class_weights'] == weights

    def test_adds_weights_to_model_config(self):
        """Test that class weights are added to model config."""
        config = {}
        model = MagicMock()
        model.config = {}
        weights = [1.0, 2.0, 1.5]

        update_config_with_class_weights(config, model, weights)

        assert model.config['model']['class_weights'] == weights

    def test_handles_none_weights(self):
        """Test handling of None class weights."""
        config = {}
        model = MagicMock()
        model.config = {}

        update_config_with_class_weights(config, model, None)

        assert config['model']['class_weights'] is None
        assert model.config['model']['class_weights'] is None

    def test_handles_model_without_config(self):
        """Test handling of model without config attribute."""
        config = {}
        model = MagicMock(spec=[])  # No config attribute
        weights = [1.0, 2.0]

        # Should not raise
        update_config_with_class_weights(config, model, weights)

        assert config['model']['class_weights'] == weights

    def test_preserves_existing_model_config(self):
        """Test that existing model config is preserved."""
        config = {'model': {'num_layers': 4}}
        model = MagicMock()
        model.config = {'model': {'head': 'tcn'}}
        weights = [1.0, 1.5]

        update_config_with_class_weights(config, model, weights)

        assert config['model']['num_layers'] == 4
        assert config['model']['class_weights'] == weights
        assert model.config['model']['head'] == 'tcn'
        assert model.config['model']['class_weights'] == weights


class TestUpdateConfigWithLabelNames:
    """Test the update_config_with_label_names function."""

    def test_adds_label_names_to_config(self):
        """Test that label names are added to config."""
        config = {'data': {}}
        model = MagicMock()
        model.config = {'data': {}}
        label_names = ['class_a', 'class_b', 'class_c']

        update_config_with_label_names(config, model, label_names)

        assert config['data']['label_names'] == label_names
        assert model.config['data']['label_names'] == label_names

    def test_empty_label_names_not_added(self):
        """Test that empty label names list is not added."""
        config = {'data': {}}
        model = MagicMock()
        model.config = {'data': {}}

        update_config_with_label_names(config, model, [])

        assert 'label_names' not in config['data']

    def test_creates_data_section_if_missing(self):
        """Test that data section is created if missing."""
        config = {}
        model = MagicMock()
        model.config = {}
        label_names = ['class_a', 'class_b']

        update_config_with_label_names(config, model, label_names)

        assert config['data']['label_names'] == label_names

    def test_handles_model_without_config(self):
        """Test handling of model without config attribute."""
        config = {}
        model = MagicMock(spec=[])  # No config attribute
        label_names = ['class_a', 'class_b']

        # Should not raise
        update_config_with_label_names(config, model, label_names)

        assert config['data']['label_names'] == label_names


class TestSaveConfig:
    """Test the save_config function."""

    def test_saves_config_to_file(self):
        """Test that config is saved to YAML file."""
        config = {'data': {'path': '/test'}, 'training': {'epochs': 10}}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            result = save_config(config, output_dir)

            assert result == output_dir / 'config.yaml'
            assert result.exists()

            # Verify content
            import yaml
            with open(result) as f:
                loaded = yaml.safe_load(f)
            assert loaded == config

    def test_creates_parent_directories(self):
        """Test that parent directories are created."""
        config = {'test': 'value'}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'nested' / 'dirs'
            result = save_config(config, output_dir)

            assert result.exists()

    def test_custom_filename(self):
        """Test saving with custom filename."""
        config = {'test': 'value'}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            result = save_config(config, output_dir, filename='custom.yaml')

            assert result == output_dir / 'custom.yaml'
            assert result.exists()

    def test_overwrites_existing_file(self):
        """Test that existing config file is overwritten."""
        config1 = {'version': 1}
        config2 = {'version': 2}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            save_config(config1, output_dir)
            save_config(config2, output_dir)

            import yaml
            with open(output_dir / 'config.yaml') as f:
                loaded = yaml.safe_load(f)
            assert loaded == config2


class TestPrettyPrintConfig:
    """Test the pretty_print_config function."""

    def test_pretty_print_simple_config(self, capsys):
        """Test pretty printing a simple configuration."""
        config = {
            'model': {'layers': 2, 'units': 64},
            'training': {'epochs': 100, 'lr': 0.001},
            'data': {'batch_size': 32}
        }

        pretty_print_config(config)
        captured = capsys.readouterr()

        # Check that all sections are printed
        assert 'Configuration:' in captured.out
        assert 'model parameters' in captured.out
        assert 'training parameters' in captured.out
        assert 'data parameters' in captured.out

        # Check that values are printed
        assert 'layers: 2' in captured.out
        assert 'epochs: 100' in captured.out
        assert 'batch_size: 32' in captured.out

    def test_pretty_print_nested_config(self, capsys):
        """Test pretty printing nested configuration."""
        config = {
            'model': {
                'head': 'temporalmlp',
                'params': {'units': 128}
            },
            'simple_value': 'test'
        }

        pretty_print_config(config)
        captured = capsys.readouterr()

        # Check nested dict is handled
        assert 'head: temporalmlp' in captured.out
        assert 'params: {' in captured.out

        # Check simple value is handled
        assert 'test' in captured.out

    def test_pretty_print_empty_config(self, capsys):
        """Test pretty printing empty configuration."""
        config = {}

        pretty_print_config(config)
        captured = capsys.readouterr()

        assert 'Configuration:' in captured.out


class TestGetCallbacks:
    """Test the get_callbacks function."""

    def test_get_callbacks_default(self):
        """Test get_callbacks with default parameters."""
        callbacks = get_callbacks()

        assert len(callbacks) == 2  # lr_monitor and checkpointing

        # Check types
        callback_types = [type(cb).__name__ for cb in callbacks]
        assert 'LearningRateMonitor' in callback_types
        assert 'ModelCheckpoint' in callback_types

    def test_get_callbacks_all_disabled(self):
        """Test get_callbacks with all features disabled."""
        callbacks = get_callbacks(
            checkpointing=False,
            lr_monitor=False,
            early_stopping=False,
        )

        assert len(callbacks) == 0

    def test_get_callbacks_early_stopping(self):
        """Test get_callbacks with early stopping enabled."""
        callbacks = get_callbacks(
            early_stopping=True,
            early_stopping_patience=5,
        )

        callback_types = [type(cb).__name__ for cb in callbacks]
        assert 'EarlyStopping' in callback_types

        # Find early stopping callback and check patience
        early_stop_cb = next(cb for cb in callbacks if type(cb).__name__ == 'EarlyStopping')
        assert early_stop_cb.patience == 5

    def test_get_callbacks_periodic_checkpointing(self):
        """Test get_callbacks with periodic checkpointing."""
        callbacks = get_callbacks(ckpt_every_n_epochs=10)

        # Should have 3 callbacks: lr_monitor, best checkpoint, periodic checkpoint
        assert len(callbacks) == 3

        callback_types = [type(cb).__name__ for cb in callbacks]
        assert callback_types.count('ModelCheckpoint') == 2

    def test_get_callbacks_no_lr_monitor(self):
        """Test get_callbacks without learning rate monitoring."""
        callbacks = get_callbacks(lr_monitor=False)

        callback_types = [type(cb).__name__ for cb in callbacks]
        assert 'LearningRateMonitor' not in callback_types
        assert 'ModelCheckpoint' in callback_types  # checkpointing still enabled

    def test_get_callbacks_comprehensive(self):
        """Test get_callbacks with all features enabled."""
        callbacks = get_callbacks(
            checkpointing=True,
            lr_monitor=True,
            ckpt_every_n_epochs=5,
            early_stopping=True,
            early_stopping_patience=15,
        )

        # Should have 4 callbacks: lr_monitor, best checkpoint, periodic checkpoint, early stopping
        assert len(callbacks) == 4

        callback_types = [type(cb).__name__ for cb in callbacks]
        assert 'LearningRateMonitor' in callback_types
        assert 'EarlyStopping' in callback_types
        assert callback_types.count('ModelCheckpoint') == 2

        # Check early stopping patience
        early_stop_cb = next(cb for cb in callbacks if type(cb).__name__ == 'EarlyStopping')
        assert early_stop_cb.patience == 15

    def test_get_callbacks_custom_monitor(self):
        """Test get_callbacks with custom monitor metric."""
        callbacks = get_callbacks(monitor='train_loss')

        # Find the checkpoint callback and verify monitor
        ckpt_cb = next(cb for cb in callbacks if type(cb).__name__ == 'ModelCheckpoint')
        assert ckpt_cb.monitor == 'train_loss'

    def test_get_callbacks_early_stopping_custom_monitor(self):
        """Test get_callbacks early stopping uses custom monitor."""
        callbacks = get_callbacks(
            early_stopping=True,
            early_stopping_patience=5,
            monitor='train_loss',
        )

        early_stop_cb = next(cb for cb in callbacks if type(cb).__name__ == 'EarlyStopping')
        assert early_stop_cb.monitor == 'train_loss'


class TestGetCallbacksFromConfig:
    """Test the get_callbacks_from_config convenience function."""

    def test_extracts_callback_params(self):
        """Test that callback params are extracted from config."""
        training_config = {
            'checkpointing': True,
            'lr_monitor': True,
            'early_stopping': True,
            'early_stopping_patience': 7,
        }

        callbacks = get_callbacks_from_config(training_config)

        callback_types = [type(cb).__name__ for cb in callbacks]
        assert 'LearningRateMonitor' in callback_types
        assert 'ModelCheckpoint' in callback_types
        assert 'EarlyStopping' in callback_types

        # Check patience
        early_stop_cb = next(cb for cb in callbacks if type(cb).__name__ == 'EarlyStopping')
        assert early_stop_cb.patience == 7

    def test_uses_defaults_for_missing_keys(self):
        """Test that defaults are used for missing config keys."""
        training_config = {}

        callbacks = get_callbacks_from_config(training_config)

        # Default is checkpointing and lr_monitor enabled
        assert len(callbacks) == 2
        callback_types = [type(cb).__name__ for cb in callbacks]
        assert 'LearningRateMonitor' in callback_types
        assert 'ModelCheckpoint' in callback_types

    def test_periodic_checkpointing_from_config(self):
        """Test periodic checkpointing from config."""
        training_config = {
            'ckpt_every_n_epochs': 10,
        }

        callbacks = get_callbacks_from_config(training_config)

        callback_types = [type(cb).__name__ for cb in callbacks]
        assert callback_types.count('ModelCheckpoint') == 2

    def test_get_callbacks_from_config_custom_monitor(self):
        """Test that custom monitor is passed through from config helper."""
        training_config = {'checkpointing': True}

        callbacks = get_callbacks_from_config(training_config, monitor='train_loss')

        ckpt_cb = next(cb for cb in callbacks if type(cb).__name__ == 'ModelCheckpoint')
        assert ckpt_cb.monitor == 'train_loss'


# =============================================================================
# Tests for CSV Pipeline (train.py)
# =============================================================================

class TestComputeClassWeights:
    """Test the compute_class_weights function."""

    @pytest.fixture
    def mock_datamodule(self):
        """Create mock DataModule for testing."""
        datamodule = MagicMock(spec=DataModule)

        # Create mock dataset with sample data
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 3
        mock_dataset.label_names = ['class0', 'class1', 'class2', 'class3']

        # Mock batch data with one-hot encoded labels
        # batch1: 3 timepoints, first is class 0, second is class 1, third is class 1
        batch1 = {'labels': torch.tensor([
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]],  # sequence 1
            [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0]]   # sequence 2
        ])}  # counts: class 0: 1, class 1: 3, class 2: 2, class 3: 0

        # batch2: 3 timepoints
        batch2 = {'labels': torch.tensor([
            [[1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]],  # sequence 1
            [[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1]]   # sequence 2
        ])}  # counts: class 0: 2, class 1: 1, class 2: 2, class 3: 1

        # batch3: 3 timepoints
        batch3 = {'labels': torch.tensor([
            [[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]],  # sequence 1
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]   # sequence 2
        ])}  # counts: class 0: 1, class 1: 1, class 2: 1, class 3: 3

        # Total counts: class 0: 4, class 1: 5, class 2: 5, class 3: 4

        mock_dataset.__getitem__.side_effect = [batch1, batch2, batch3]
        datamodule.dataset_train = mock_dataset

        return datamodule

    def test_compute_class_weights_basic(self, mock_datamodule):
        """Test basic class weight computation."""
        weights = compute_class_weights(mock_datamodule, ignore_index=-100)

        # Expected counts: class 0: 4, class 1: 5, class 2: 5, class 3: 4
        # Max count is 5, so weights should be: [5/4, 5/5, 5/5, 5/4] = [1.25, 1.0, 1.0, 1.25]
        expected_weights = [1.25, 1.0, 1.0, 1.25]

        assert len(weights) == 4
        for i, expected in enumerate(expected_weights):
            assert abs(weights[i] - expected) < 1e-6

    def test_compute_class_weights_ignore_index(self, mock_datamodule):
        """Test class weight computation with ignored class."""
        weights = compute_class_weights(mock_datamodule, ignore_index=0)

        # Class 0 should be ignored (weight 0)
        # Expected counts: class 1: 5, class 2: 5, class 3: 4
        # Max count is 5, so weights should be: [0.0, 1.0, 1.0, 1.25]
        expected_weights = [0.0, 1.0, 1.0, 1.25]

        assert len(weights) == 4
        for i, expected in enumerate(expected_weights):
            assert abs(weights[i] - expected) < 1e-6

    def test_compute_class_weights_no_labels(self):
        """Test class weight computation when no labels are found."""
        datamodule = MagicMock(spec=DataModule)
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 1
        mock_dataset.label_names = ['class0', 'class1', 'class2', 'class3']

        # Batch with no labels key
        batch = {'features': torch.randn(2, 10)}
        mock_dataset.__getitem__.return_value = batch

        datamodule.dataset_train = mock_dataset

        weights = compute_class_weights(datamodule)

        # Should return uniform weights
        expected_weights = [1.0, 1.0, 1.0, 1.0]
        assert weights == expected_weights

    def test_compute_class_weights_calls_setup(self):
        """Test that compute_class_weights calls setup if needed."""
        datamodule = MagicMock(spec=DataModule)
        datamodule.dataset_train = None  # Not set up yet

        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 1
        mock_dataset.label_names = ['class0', 'class1']
        mock_dataset.__getitem__.return_value = {'labels': torch.tensor([[0, 1]])}

        # After setup is called, dataset_train will be set
        def setup_side_effect(stage):
            datamodule.dataset_train = mock_dataset

        datamodule.setup.side_effect = setup_side_effect

        compute_class_weights(datamodule)

        datamodule.setup.assert_called_once_with('fit')


class TestBuildDataConfigFromPath:
    """Test the build_data_config_from_path function."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory with sample structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir)

            # Create signal directories
            markers_dir = data_path / 'markers'
            labels_dir = data_path / 'labels'
            features_dir = data_path / 'features_0'

            markers_dir.mkdir()
            labels_dir.mkdir()
            features_dir.mkdir()

            # Create sample CSV files with realistic DLC/label formats
            experiments = ['exp1', 'exp2', 'exp3']
            for exp in experiments:
                # DLC markers format with multi-level headers
                markers_content = (
                    'scorer,model,model,model,model,model,model\n'
                    'bodyparts,body1,body1,body1,body2,body2,body2\n'
                    'coords,x,y,likelihood,x,y,likelihood\n'
                    '0,10.5,20.3,0.9,15.2,25.1,0.8\n'
                    '1,11.1,21.0,0.9,16.0,26.3,0.85\n'
                    '2,12.3,22.1,0.88,17.1,27.0,0.9\n'
                )
                (markers_dir / f'{exp}.csv').write_text(markers_content)

                # One-hot encoded labels with index column
                labels_content = (
                    ',background,still,walk,groom\n'
                    '0,1,0,0,0\n'
                    '1,0,1,0,0\n'
                    '2,0,0,1,0\n'
                )
                (labels_dir / f'{exp}.csv').write_text(labels_content)

                # Optional features format (similar to markers but simpler)
                features_content = (
                    ',feature1,feature2,feature3\n'
                    '0,0.1,0.2,0.3\n'
                    '1,0.4,0.5,0.6\n'
                    '2,0.7,0.8,0.9\n'
                )
                (features_dir / f'{exp}.csv').write_text(features_content)

            yield data_path

    def test_build_config_auto_detect_all(self, temp_data_dir):
        """Test building config with auto-detection of experiments and signals."""
        config = build_data_config_from_path(temp_data_dir)

        # Should find all 3 experiments
        assert len(config['ids']) == 3
        assert set(config['ids']) == {'exp1', 'exp2', 'exp3'}

        # Should have signals, transforms, paths for each experiment
        assert len(config['signals']) == 3
        assert len(config['transforms']) == 3
        assert len(config['paths']) == 3

    def test_build_config_specified_experiments(self, temp_data_dir):
        """Test building config with specified experiment IDs."""
        config = build_data_config_from_path(
            temp_data_dir,
            expt_ids=['exp1', 'exp2']
        )

        assert len(config['ids']) == 2
        assert set(config['ids']) == {'exp1', 'exp2'}

    def test_build_config_missing_experiments(self, temp_data_dir):
        """Test building config with missing experiment IDs."""
        with pytest.raises(FileNotFoundError, match="Did not find expt_id="):
            build_data_config_from_path(
                temp_data_dir,
                expt_ids=['exp1', 'exp2', 'exp4']
            )

    def test_build_config_specified_signal_types(self, temp_data_dir):
        """Test building config with specified signal types."""
        config = build_data_config_from_path(
            temp_data_dir,
            signal_types=['markers', 'labels']
        )

        # Check only specified signal types
        assert len(config['signals'][0]) == 2
        signal_types = set(config['signals'][0])
        assert signal_types == {'markers', 'labels'}

    def test_build_config_default_transforms(self, temp_data_dir):
        """Test that default transforms are applied when none specified."""
        config = build_data_config_from_path(temp_data_dir, expt_ids=['exp1'])

        # Check transforms for first experiment
        transforms = config['transforms'][0]
        signals = config['signals'][0]

        for i, signal_type in enumerate(signals):
            if signal_type.startswith(('markers', 'features')):
                # Should have ZScore transform
                assert transforms[i] is not None
                assert len(transforms[i]) == 1
                assert transforms[i][0].__class__.__name__ == 'ZScore'
            else:
                # Labels should have no transform
                assert transforms[i] is None

    def test_build_config_custom_transforms_single(self, temp_data_dir):
        """Test building config with single custom transform."""
        config = build_data_config_from_path(
            temp_data_dir,
            signal_types=['markers', 'labels'],
            transforms=['MotionEnergy']
        )

        # Check transforms for first experiment
        transforms = config['transforms'][0]
        signals = config['signals'][0]

        for i, signal_type in enumerate(signals):
            if signal_type.startswith(('markers', 'features')):
                # Should have MotionEnergy transform
                assert transforms[i] is not None
                assert len(transforms[i]) == 1
                assert transforms[i][0].__class__.__name__ == 'MotionEnergy'
            else:
                # Labels should have no transform
                assert transforms[i] is None

    def test_build_config_custom_transforms_multiple(self, temp_data_dir):
        """Test building config with multiple custom transforms."""
        config = build_data_config_from_path(
            temp_data_dir,
            signal_types=['markers', 'labels'],
            transforms=['ZScore', 'MotionEnergy']
        )

        # Check transforms for first experiment
        transforms = config['transforms'][0]
        signals = config['signals'][0]

        # Markers should get both transforms
        markers_idx = signals.index('markers')
        assert transforms[markers_idx][0].__class__.__name__ == 'ZScore'
        assert transforms[markers_idx][1].__class__.__name__ == 'MotionEnergy'

        # Labels should have no transform
        labels_idx = signals.index('labels')
        assert transforms[labels_idx] is None

    def test_build_config_nonexistent_path(self):
        """Test error handling for nonexistent data path."""
        with pytest.raises(FileNotFoundError, match="Data path does not exist"):
            build_data_config_from_path('/nonexistent/path')

    def test_build_config_no_signal_dirs(self):
        """Test error handling when no signal directories are found."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir)
            # Empty directory

            with pytest.raises(ValueError, match="No signal directories found"):
                build_data_config_from_path(data_path)

    def test_build_config_invalid_transform(self, temp_data_dir):
        """Test error handling for invalid transform class name."""
        with pytest.raises(ValueError, match="Unknown transform class: InvalidTransform"):
            build_data_config_from_path(
                temp_data_dir,
                transforms=['InvalidTransform']
            )

    def test_build_config_no_signal_dir(self):
        """Test error handling when signal directory is missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir)

            with pytest.raises(NotADirectoryError, match="Signal directory not found"):
                build_data_config_from_path(data_path, signal_types=['markers'])

            # Create signal directory but no CSV files
            markers_dir = data_path / 'markers'
            markers_dir.mkdir()
            markers_dir.joinpath('session1.csv').touch()

            with pytest.raises(NotADirectoryError, match="Signal directory not found"):
                build_data_config_from_path(data_path, signal_types=['markers', 'labels'])

    def test_build_config_no_experiments(self):
        """Test error handling when no experiment CSV files are found."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir)

            # Create signal directory but no CSV files
            markers_dir = data_path / 'markers'
            markers_dir.mkdir()

            with pytest.raises(ValueError, match="No CSV files found"):
                build_data_config_from_path(data_path)

    def test_build_config_paths_are_correct(self, temp_data_dir):
        """Test that generated paths are correct and absolute."""
        config = build_data_config_from_path(
            temp_data_dir,
            expt_ids=['exp1'],
            signal_types=['markers', 'labels']
        )

        paths = config['paths'][0]
        signals = config['signals'][0]

        for i, sig_type in enumerate(signals):
            expected_path = temp_data_dir / sig_type / 'exp1.csv'
            assert paths[i] == str(expected_path)


class TestTrain:
    """Test the train function."""

    @pytest.fixture
    def basic_config(self):
        """Basic configuration for training tests."""
        return {
            'data': {
                'data_path': '/tmp/test_data',
                'weight_classes': True,
            },
            'model': {
                'input_size': 10,
                'output_size': 4,
                'head': 'temporalmlp',
                'num_hid_units': 32,
                'num_layers': 2,
            },
            'optimizer': {
                'type': 'Adam',
                'lr': 1e-3,
            },
            'training': {
                'num_epochs': 2,
                'batch_size': 4,
                'sequence_length': 50,
                'num_workers': 0,
                'device': 'cpu',
                'checkpointing': False,
                'lr_monitor': False,
                'seed': 42,
            }
        }

    @pytest.fixture
    def mock_model(self):
        """Mock Lightning model for testing."""
        model = MagicMock(spec=Segmenter)
        model.config = {}
        return model

    @pytest.fixture
    def temp_output_dir(self):
        """Temporary output directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @patch('lightning_action.train.DataModule')
    @patch('lightning_action.train.pl.Trainer')
    @patch('lightning_action.train.build_data_config_from_path')
    def test_train_basic_flow(
        self,
        mock_build_config,
        mock_trainer_class,
        mock_datamodule_class,
        basic_config,
        mock_model,
        temp_output_dir,
    ):
        """Test basic training flow."""
        # Setup mocks
        mock_build_config.return_value = {
            'ids': ['exp1'], 'signals': [['markers', 'labels']],
            'transforms': [None, None], 'paths': [['path1', 'path2']]
        }

        mock_datamodule = MagicMock()
        mock_datamodule.dataset_train = MagicMock()
        mock_datamodule.dataset_train.__len__.return_value = 100
        mock_datamodule_class.return_value = mock_datamodule

        mock_trainer = MagicMock()
        mock_trainer_class.return_value = mock_trainer

        # Run training
        result = train(basic_config, mock_model, temp_output_dir)

        # Check that components were created
        mock_datamodule_class.assert_called_once()
        mock_trainer_class.assert_called_once()

        # Check that trainer.fit was called
        mock_trainer.fit.assert_called_once_with(model=mock_model, datamodule=mock_datamodule)

        # Check that config was saved
        config_file = temp_output_dir / 'config.yaml'
        assert config_file.exists()

        # Check that final model was saved
        mock_trainer.save_checkpoint.assert_called_once()

        # Check return value
        assert result == mock_model

    @patch('lightning_action.train.DataModule')
    @patch('lightning_action.train.build_data_config_from_path')
    def test_train_with_class_weights(
        self,
        mock_build_config,
        mock_datamodule_class,
        basic_config,
        mock_model,
        temp_output_dir
    ):
        """Test training with class weight computation."""
        # Enable class weighting
        basic_config['data']['weight_classes'] = True

        # Setup mocks
        mock_build_config.return_value = {
            'ids': ['exp1'], 'signals': [['markers', 'labels']],
            'transforms': [None, None], 'paths': [['path1', 'path2']]
        }

        mock_datamodule = MagicMock()
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 10
        mock_dataset.__getitem__.return_value = {'labels': torch.tensor([[0, 1, 1]])}
        mock_datamodule.dataset_train = mock_dataset
        mock_datamodule_class.return_value = mock_datamodule

        with patch('lightning_action.train.pl.Trainer') as mock_trainer_class:
            mock_trainer = MagicMock()
            mock_trainer_class.return_value = mock_trainer

            train(basic_config, mock_model, temp_output_dir)

            # Check that class weights were computed and set
            assert 'class_weights' in basic_config['model']

    @patch('lightning_action.train.DataModule')
    @patch('lightning_action.train.build_data_config_from_path')
    def test_train_without_class_weights(
        self,
        mock_build_config,
        mock_datamodule_class,
        basic_config,
        mock_model,
        temp_output_dir
    ):
        """Test training with class weighting disabled."""
        # Disable class weighting
        basic_config['data']['weight_classes'] = False

        # Setup mocks
        mock_build_config.return_value = {
            'ids': ['exp1'], 'signals': [['markers', 'labels']],
            'transforms': [None, None], 'paths': [['path1', 'path2']]
        }

        mock_datamodule = MagicMock()
        mock_datamodule.dataset_train = MagicMock()
        mock_datamodule.dataset_train.__len__.return_value = 100
        mock_datamodule_class.return_value = mock_datamodule

        with patch('lightning_action.train.pl.Trainer') as mock_trainer_class:
            mock_trainer = MagicMock()
            mock_trainer_class.return_value = mock_trainer

            train(basic_config, mock_model, temp_output_dir)

            # Check that class weights are None
            assert basic_config['model']['class_weights'] is None

    def test_train_missing_config_sections(self, mock_model, temp_output_dir):
        """Test training with missing required configuration sections."""
        # Missing data section
        config_no_data = {'model': {}, 'training': {}}
        with pytest.raises(ValueError, match="Configuration must contain 'data' section"):
            train(config_no_data, mock_model, temp_output_dir)

        # Missing training section
        config_no_training = {'data': {}, 'model': {}}
        with pytest.raises(ValueError, match="Configuration must contain 'training' section"):
            train(config_no_training, mock_model, temp_output_dir)

    @patch('lightning_action.train.DataModule')
    @patch('lightning_action.train.build_data_config_from_path')
    def test_train_gpu_config(
        self,
        mock_build_config,
        mock_datamodule_class,
        basic_config,
        mock_model,
        temp_output_dir,
    ):
        """Test training with GPU configuration."""
        # Set GPU device
        basic_config['training']['device'] = 'gpu'

        # Setup mocks
        mock_build_config.return_value = {
            'ids': ['exp1'], 'signals': [['markers', 'labels']],
            'transforms': [None, None], 'paths': [['path1', 'path2']]
        }

        mock_datamodule = MagicMock()
        mock_datamodule.dataset_train = MagicMock()
        mock_datamodule.dataset_train.__len__.return_value = 100
        mock_datamodule_class.return_value = mock_datamodule

        with patch('lightning_action.train.torch.cuda.is_available', return_value=True):
            with patch('lightning_action.train.pl.Trainer') as mock_trainer_class:
                mock_trainer = MagicMock()
                mock_trainer_class.return_value = mock_trainer

                train(basic_config, mock_model, temp_output_dir)

                # Check trainer was configured for GPU
                trainer_kwargs = mock_trainer_class.call_args[1]
                assert trainer_kwargs['accelerator'] == 'gpu'
                assert trainer_kwargs['devices'] == 1

    @patch('lightning_action.train.DataModule')
    @patch('lightning_action.train.build_data_config_from_path')
    def test_train_cpu_fallback_when_gpu_unavailable(
        self,
        mock_build_config,
        mock_datamodule_class,
        basic_config,
        mock_model,
        temp_output_dir,
    ):
        """Test training falls back to CPU when GPU is unavailable."""
        # Request GPU
        basic_config['training']['device'] = 'gpu'

        # Setup mocks
        mock_build_config.return_value = {
            'ids': ['exp1'], 'signals': [['markers', 'labels']],
            'transforms': [None, None], 'paths': [['path1', 'path2']]
        }

        mock_datamodule = MagicMock()
        mock_datamodule.dataset_train = MagicMock()
        mock_datamodule.dataset_train.__len__.return_value = 100
        mock_datamodule_class.return_value = mock_datamodule

        with patch('lightning_action.train.torch.cuda.is_available', return_value=False):
            with patch('lightning_action.train.pl.Trainer') as mock_trainer_class:
                mock_trainer = MagicMock()
                mock_trainer_class.return_value = mock_trainer

                train(basic_config, mock_model, temp_output_dir)

                # Check trainer was configured for CPU
                trainer_kwargs = mock_trainer_class.call_args[1]
                assert trainer_kwargs['accelerator'] == 'cpu'

    @patch('lightning_action.train.DataModule')
    def test_train_existing_data_config(self, mock_datamodule_class, mock_model, temp_output_dir):
        """Test training with existing full data configuration (no data_path)."""
        config = {
            'data': {
                'ids': ['exp1'],
                'signals': [['markers', 'labels']],
                'transforms': [None, None],
                'paths': [['path1', 'path2']],
                'weight_classes': False,
            },
            'model': {
                'input_size': 10,
                'output_size': 4,
            },
            'training': {
                'num_epochs': 1,
                'batch_size': 4,
                'device': 'cpu',
                'checkpointing': False,
                'lr_monitor': False,
            }
        }

        mock_datamodule = MagicMock()
        mock_datamodule.dataset_train = MagicMock()
        mock_datamodule.dataset_train.__len__.return_value = 100
        mock_datamodule_class.return_value = mock_datamodule

        with patch('lightning_action.train.pl.Trainer') as mock_trainer_class:
            mock_trainer = MagicMock()
            mock_trainer_class.return_value = mock_trainer

            train(config, mock_model, temp_output_dir)

            # Should not have called build_data_config_from_path
            # DataModule should be called with original config
            call_args = mock_datamodule_class.call_args
            assert call_args[1]['data_config'] == config['data']

    @patch('lightning_action.train.DataModule')
    @patch('lightning_action.train.build_data_config_from_path')
    def test_train_saves_version(
        self,
        mock_build_config,
        mock_datamodule_class,
        basic_config,
        mock_model,
        temp_output_dir,
    ):
        """Test that training saves package version to config."""
        # Setup mocks
        mock_build_config.return_value = {
            'ids': ['exp1'], 'signals': [['markers', 'labels']],
            'transforms': [None, None], 'paths': [['path1', 'path2']]
        }

        mock_datamodule = MagicMock()
        mock_datamodule.dataset_train = MagicMock()
        mock_datamodule.dataset_train.__len__.return_value = 100
        mock_datamodule_class.return_value = mock_datamodule

        with patch('lightning_action.train.pl.Trainer') as mock_trainer_class:
            mock_trainer = MagicMock()
            mock_trainer_class.return_value = mock_trainer

            train(basic_config, mock_model, temp_output_dir)

            # Check that version was added to config
            assert 'version' in basic_config

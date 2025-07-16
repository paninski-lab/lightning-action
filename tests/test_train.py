"""Tests for training functionality."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from lightning_action.data import DataModule
from lightning_action.models.segmenter import Segmenter
from lightning_action.train import (
    build_data_config_from_path,
    compute_class_weights,
    get_callbacks,
    pretty_print_config,
    reset_seeds,
    train,
)


class TestResetSeeds:
    """Test the reset_seeds function."""

    def test_reset_seeds_with_default(self):
        """Test reset_seeds with default seed value."""
        reset_seeds()
        
        # check that seeds are set
        assert os.environ.get("PYTHONHASHSEED") == "0"
        
        # check torch backends settings
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False

    def test_reset_seeds_with_custom_seed(self):
        """Test reset_seeds with custom seed value."""
        custom_seed = 42
        reset_seeds(seed=custom_seed)
        
        # check environment variable
        assert os.environ.get("PYTHONHASHSEED") == str(custom_seed)

    def test_deterministic_behavior(self):
        """Test that reset_seeds provides deterministic behavior."""
        seed = 123
        
        # reset seeds and generate random numbers
        reset_seeds(seed=seed)
        torch_val1 = torch.randn(1).item()
        np_val1 = np.random.random()
        
        # reset seeds again and generate random numbers
        reset_seeds(seed=seed)
        torch_val2 = torch.randn(1).item()
        np_val2 = np.random.random()
        
        # values should be identical
        assert torch_val1 == torch_val2
        assert np_val1 == np_val2


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
        
        # check that all sections are printed
        assert 'Configuration:' in captured.out
        assert 'model parameters' in captured.out
        assert 'training parameters' in captured.out
        assert 'data parameters' in captured.out
        
        # check that values are printed
        assert 'layers: 2' in captured.out
        assert 'epochs: 100' in captured.out
        assert 'batch_size: 32' in captured.out

    def test_pretty_print_nested_config(self, capsys):
        """Test pretty printing nested configuration."""
        config = {
            'model': {
                'backbone': 'temporalmlp',
                'params': {'units': 128}
            },
            'simple_value': 'test'
        }
        
        pretty_print_config(config)
        captured = capsys.readouterr()
        
        # check nested dict is handled
        assert 'backbone: temporalmlp' in captured.out
        assert 'params: {' in captured.out
        
        # check simple value is handled
        assert 'test' in captured.out

    def test_pretty_print_empty_config(self, capsys):
        """Test pretty printing empty configuration."""
        config = {}
        
        pretty_print_config(config)
        captured = capsys.readouterr()
        
        assert 'Configuration:' in captured.out


class TestBuildDataConfigFromPath:
    """Test the build_data_config_from_path function."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory with sample structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir)
            
            # create signal directories
            markers_dir = data_path / 'markers'
            labels_dir = data_path / 'labels'
            features_dir = data_path / 'features_0'
            
            markers_dir.mkdir()
            labels_dir.mkdir()
            features_dir.mkdir()
            
            # create sample CSV files with realistic DLC/label formats
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

    def test_build_config_specified_experiments(self, temp_data_dir):
        """Test building config with specified experiment IDs."""
        config = build_data_config_from_path(
            temp_data_dir,
            expt_ids=['exp1', 'exp2']
        )
        
        assert len(config['ids']) == 2
        assert set(config['ids']) == {'exp1', 'exp2'}

    def test_build_config_specified_signal_types(self, temp_data_dir):
        """Test building config with specified signal types."""
        config = build_data_config_from_path(
            temp_data_dir,
            signal_types=['markers', 'labels']
        )
        
        # check only specified signal types
        assert len(config['signals'][0]) == 2
        signal_types = set(config['signals'][0])
        assert signal_types == {'markers', 'labels'}

    def test_build_config_nonexistent_path(self):
        """Test error handling for nonexistent data path."""
        with pytest.raises(FileNotFoundError, match="Data path does not exist"):
            build_data_config_from_path('/nonexistent/path')

    def test_build_config_no_experiments(self):
        """Test error handling when no experiments are found."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir)
            
            # create signal directory but no CSV files
            markers_dir = data_path / 'markers'
            markers_dir.mkdir()
            
            with pytest.raises(ValueError, match="No experiment CSV files found"):
                build_data_config_from_path(data_path)

    def test_build_config_no_signal_dirs(self):
        """Test error handling when no signal directories are found."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir)
            # empty directory
            
            with pytest.raises(ValueError, match="No signal directories found"):
                build_data_config_from_path(data_path)

    def test_build_config_default_transforms(self, temp_data_dir):
        """Test that default transforms are applied when none specified."""
        config = build_data_config_from_path(temp_data_dir, expt_ids=['exp1'])
        
        # check transforms for first experiment
        transforms = config['transforms'][0]
        signals = config['signals'][0]
        
        for i, signal_type in enumerate(signals):
            if signal_type.startswith(('markers', 'features')):
                # should have ZScore transform
                assert transforms[i] is not None
                assert len(transforms[i]) == 1
                assert transforms[i][0].__class__.__name__ == 'ZScore'
            else:
                # labels should have no transform
                assert transforms[i] is None

    def test_build_config_custom_transforms_single(self, temp_data_dir):
        """Test building config with single custom transform."""
        config = build_data_config_from_path(
            temp_data_dir,
            signal_types=['markers', 'labels'],
            transforms=['MotionEnergy']
        )
        
        # check transforms for first experiment
        transforms = config['transforms'][0]
        signals = config['signals'][0]
        
        for i, signal_type in enumerate(signals):
            if signal_type.startswith(('markers', 'features')):
                # should have MotionEnergy transform
                assert transforms[i] is not None
                assert len(transforms[i]) == 1
                assert transforms[i][0].__class__.__name__ == 'MotionEnergy'
            else:
                # labels should have no transform
                assert transforms[i] is None

    def test_build_config_custom_transforms_multiple(self, temp_data_dir):
        """Test building config with multiple custom transforms."""
        config = build_data_config_from_path(
            temp_data_dir,
            signal_types=['markers', 'labels'],
            transforms=['ZScore', 'MotionEnergy']
        )
        
        # check transforms for first experiment
        transforms = config['transforms'][0]
        signals = config['signals'][0]
        print(transforms)
        # markers should get ZScore (first transform)
        markers_idx = signals.index('markers')
        assert transforms[markers_idx][0].__class__.__name__ == 'ZScore'
        assert transforms[markers_idx][1].__class__.__name__ == 'MotionEnergy'
        
        # labels should have no transform
        labels_idx = signals.index('labels')
        assert transforms[labels_idx] is None

    def test_build_config_invalid_transform(self, temp_data_dir):
        """Test error handling for invalid transform class name."""
        with pytest.raises(ValueError, match="Unknown transform class: InvalidTransform"):
            build_data_config_from_path(
                temp_data_dir,
                transforms=['InvalidTransform']
            )


class TestComputeClassWeights:
    """Test the compute_class_weights function."""

    @pytest.fixture
    def mock_datamodule(self):
        """Create mock DataModule for testing."""
        datamodule = MagicMock(spec=DataModule)
        
        # create mock dataset with sample data
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 3
        
        # mock batch data with one-hot encoded labels
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
        weights = compute_class_weights(mock_datamodule, ignore_class=-100)
        print(weights)
        # expected counts: class 0: 4, class 1: 5, class 2: 5, class 3: 4
        # max count is 5, so weights should be: [5/4, 5/5, 5/5, 5/4] = [1.25, 1.0, 1.0, 1.25]
        expected_weights = [1.25, 1.0, 1.0, 1.25]
        
        assert len(weights) == 4
        for i, expected in enumerate(expected_weights):
            assert abs(weights[i] - expected) < 1e-6

    def test_compute_class_weights_ignore_class(self, mock_datamodule):
        """Test class weight computation with ignored class."""
        weights = compute_class_weights(mock_datamodule, ignore_class=0)
        
        # class 0 should be ignored (weight 0)
        # expected counts: class 1: 5, class 2: 5, class 3: 4
        # max count is 5, so weights should be: [0.0, 1.0, 1.0, 1.25]
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
        
        # batch with no labels key
        batch = {'features': torch.randn(2, 10)}
        mock_dataset.__getitem__.return_value = batch
        
        datamodule.dataset_train = mock_dataset
        
        weights = compute_class_weights(datamodule)
        
        # should return uniform weights
        expected_weights = [1.0, 1.0, 1.0, 1.0]
        assert weights == expected_weights


class TestGetCallbacks:
    """Test the get_callbacks function."""

    def test_get_callbacks_default(self):
        """Test get_callbacks with default parameters."""
        callbacks = get_callbacks()
        
        assert len(callbacks) == 2  # lr_monitor and checkpointing
        
        # check types
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
        
        # find early stopping callback and check patience
        early_stop_cb = next(cb for cb in callbacks if type(cb).__name__ == 'EarlyStopping')
        assert early_stop_cb.patience == 5

    def test_get_callbacks_periodic_checkpointing(self):
        """Test get_callbacks with periodic checkpointing."""
        callbacks = get_callbacks(ckpt_every_n_epochs=10)
        
        # should have 3 callbacks: lr_monitor, best checkpoint, periodic checkpoint
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
        
        # should have 4 callbacks: lr_monitor, best checkpoint, periodic checkpoint, early stopping
        assert len(callbacks) == 4
        
        callback_types = [type(cb).__name__ for cb in callbacks]
        assert 'LearningRateMonitor' in callback_types
        assert 'EarlyStopping' in callback_types
        assert callback_types.count('ModelCheckpoint') == 2


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
                'backbone': 'temporalmlp',
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
    def test_train_basic_flow(self, mock_build_config, mock_trainer_class, mock_datamodule_class, 
                             basic_config, mock_model, temp_output_dir):
        """Test basic training flow."""
        # setup mocks
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
        
        # run training
        result = train(basic_config, mock_model, temp_output_dir)
        
        # check that components were created
        mock_datamodule_class.assert_called_once()
        mock_trainer_class.assert_called_once()
        
        # check that trainer.fit was called
        mock_trainer.fit.assert_called_once_with(model=mock_model, datamodule=mock_datamodule)
        
        # check that config was saved
        config_file = temp_output_dir / 'config.yaml'
        assert config_file.exists()
        
        # check that final model was saved
        mock_trainer.save_checkpoint.assert_called_once()
        
        # check return value
        assert result == mock_model

    @patch('lightning_action.train.DataModule')
    @patch('lightning_action.train.build_data_config_from_path')
    def test_train_with_class_weights(self, mock_build_config, mock_datamodule_class,
                                     basic_config, mock_model, temp_output_dir):
        """Test training with class weight computation."""
        # enable class weighting
        basic_config['data']['weight_classes'] = True
        
        # setup mocks
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
            
            # check that class weights were computed and set
            assert 'class_weights' in basic_config['model']

    def test_train_missing_config_sections(self, mock_model, temp_output_dir):
        """Test training with missing required configuration sections."""
        # missing data section
        config_no_data = {'model': {}, 'training': {}}
        with pytest.raises(ValueError, match="Configuration must contain 'data' section"):
            train(config_no_data, mock_model, temp_output_dir)
        
        # missing training section
        config_no_training = {'data': {}, 'model': {}}
        with pytest.raises(ValueError, match="Configuration must contain 'training' section"):
            train(config_no_training, mock_model, temp_output_dir)

    @patch('lightning_action.train.DataModule')
    @patch('lightning_action.train.build_data_config_from_path')
    def test_train_gpu_config(self, mock_build_config, mock_datamodule_class,
                             basic_config, mock_model, temp_output_dir):
        """Test training with GPU configuration."""
        # set GPU device
        basic_config['training']['device'] = 'gpu'
        
        # setup mocks
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
                
                # check trainer was configured for GPU
                trainer_kwargs = mock_trainer_class.call_args[1]
                assert trainer_kwargs['accelerator'] == 'gpu'
                assert trainer_kwargs['devices'] == 1

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
            
            # should not have called build_data_config_from_path
            # DataModule should be called with original config
            call_args = mock_datamodule_class.call_args
            assert call_args[1]['data_config'] == config['data']

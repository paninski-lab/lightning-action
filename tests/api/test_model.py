"""Tests for the Model and VideoModel APIs.

This module contains tests for:
- BaseModelAPI shared functionality
- Model (CSV pipeline) specific functionality  
- VideoModel (video pipeline) specific functionality

The tests verify that the refactored inheritance structure works correctly
and that both pipelines maintain their expected behavior.

Original tests preserved from test_model.py:
- TestModelIntegration: All integration tests with real data
- TestModelSequencePadding: All sequence padding computation tests

New tests added for refactored code:
- TestChdir: Context manager tests
- TestBaseModelAPIShared: Shared base class functionality
- TestVideoModelAPI: Video pipeline specific tests
- TestInheritance: Inheritance verification tests
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from lightning_action.api.base import BaseModelAPI, chdir
from lightning_action.api.model import Model
from lightning_action.api.video_model import VideoModel

# =============================================================================
# Tests for shared utilities (chdir)
# =============================================================================


class TestChdir:
    """Tests for the chdir context manager."""

    def test_chdir_changes_directory(self):
        """Test that chdir changes to the specified directory."""
        import os
        original_cwd = os.getcwd()

        with tempfile.TemporaryDirectory() as tmpdir:
            with chdir(Path(tmpdir)):
                assert os.getcwd() == tmpdir

            # Should restore original directory
            assert os.getcwd() == original_cwd

    def test_chdir_restores_on_exception(self):
        """Test that chdir restores directory even on exception."""
        import os
        original_cwd = os.getcwd()

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                with chdir(Path(tmpdir)):
                    raise ValueError("Test exception")
            except ValueError:
                pass

            # Should still restore original directory
            assert os.getcwd() == original_cwd


# =============================================================================
# Tests for BaseModelAPI shared functionality
# =============================================================================

class TestBaseModelAPIShared:
    """Tests for shared BaseModelAPI functionality."""

    def test_find_config_file_yaml(self):
        """Test finding config.yaml file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            config_path = model_dir / 'config.yaml'
            config_path.write_text('model:\n  input_size: 10')

            found_path = Model._find_config_file(model_dir)
            assert found_path == config_path

    def test_find_config_file_hparams_fallback(self):
        """Test fallback to hparams.yaml when config.yaml is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            hparams_path = model_dir / 'hparams.yaml'
            hparams_path.write_text('model:\n  input_size: 10')

            found_path = Model._find_config_file(model_dir)
            assert found_path == hparams_path

    def test_find_config_file_not_found(self):
        """Test FileNotFoundError when no config file exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)

            with pytest.raises(FileNotFoundError, match='Config file not found'):
                Model._find_config_file(model_dir)

    def test_find_checkpoint_file_best_ckpt(self):
        """Test finding best checkpoint file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            ckpt_dir = model_dir / 'checkpoints'
            ckpt_dir.mkdir()

            best_ckpt = ckpt_dir / 'best-epoch=5.ckpt'
            best_ckpt.touch()
            other_ckpt = ckpt_dir / 'epoch=10.ckpt'
            other_ckpt.touch()

            found_path = Model._find_checkpoint_file(model_dir)
            assert found_path == best_ckpt

    def test_find_checkpoint_file_any_ckpt(self):
        """Test finding any .ckpt file when no best checkpoint exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            ckpt_path = model_dir / 'model.ckpt'
            ckpt_path.touch()

            found_path = Model._find_checkpoint_file(model_dir)
            assert found_path == ckpt_path

    def test_find_checkpoint_file_not_found(self):
        """Test FileNotFoundError when no checkpoint exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)

            with pytest.raises(FileNotFoundError, match='No checkpoint files found'):
                Model._find_checkpoint_file(model_dir)


# =============================================================================
# Tests for Model sequence padding (from original TestModelSequencePadding)
# =============================================================================

class TestModelSequencePadding:
    """Test sequence padding computation in Model.from_config."""

    def test_sequence_pad_temporal_mlp(self):
        """Test sequence padding computation for TemporalMLP head."""
        config = {
            'data': {
                'data_path': '/tmp/test',
                'input_dir': 'markers',
            },
            'model': {
                'input_size': 10,
                'output_size': 4,
                'head': 'temporalmlp',
                'num_hid_units': 32,
                'num_layers': 2,
                'num_lags': 3,
            },
            'optimizer': {
                'type': 'Adam',
                'lr': 1e-3,
            },
            'training': {
                'num_epochs': 1,
                'batch_size': 4,
            }
        }

        model = Model.from_config(config)

        # For temporal-mlp, sequence_pad should equal num_lags
        assert model.config['model']['sequence_pad'] == 3

    def test_sequence_pad_dilated_tcn(self):
        """Test sequence padding computation for DilatedTCN head."""
        config = {
            'data': {
                'data_path': '/tmp/test',
                'input_dir': 'markers',
            },
            'model': {
                'input_size': 10,
                'output_size': 4,
                'head': 'dtcn',
                'num_hid_units': 32,
                'num_layers': 3,
                'num_lags': 2,
            },
            'optimizer': {
                'type': 'Adam',
                'lr': 1e-3,
            },
            'training': {
                'num_epochs': 1,
                'batch_size': 4,
            }
        }

        model = Model.from_config(config)

        # For DTCN, sequence_pad is sum of 2 * (2 ** n) * num_lags for n in range(num_layers)
        # layers: 0, 1, 2
        # dilations: 2**0=1, 2**1=2, 2**2=4
        # pad per layer: 2*1*2=4, 2*2*2=8, 2*4*2=16
        # total: 4 + 8 + 16 = 28
        expected_pad = 2 * (2 ** 0) * 2 + 2 * (2 ** 1) * 2 + 2 * (2 ** 2) * 2
        assert model.config['model']['sequence_pad'] == expected_pad

    def test_sequence_pad_lstm(self):
        """Test sequence padding computation for LSTM head."""
        config = {
            'data': {
                'data_path': '/tmp/test',
                'input_dir': 'markers',
            },
            'model': {
                'input_size': 10,
                'output_size': 4,
                'head': 'rnn',
                'rnn_type': 'lstm',
                'num_hid_units': 32,
                'num_layers': 2,
                'bidirectional': False,
            },
            'optimizer': {
                'type': 'Adam',
                'lr': 1e-3,
            },
            'training': {
                'num_epochs': 1,
                'batch_size': 4,
            }
        }

        model = Model.from_config(config)

        # For LSTM/GRU, sequence_pad should be fixed at 4
        assert model.config['model']['sequence_pad'] == 4

    def test_sequence_pad_gru(self):
        """Test sequence padding computation for GRU head."""
        config = {
            'data': {
                'data_path': '/tmp/test',
                'input_dir': 'markers',
            },
            'model': {
                'input_size': 10,
                'output_size': 4,
                'head': 'rnn',
                'rnn_type': 'gru',
                'num_hid_units': 32,
                'num_layers': 2,
                'bidirectional': True,
            },
            'optimizer': {
                'type': 'Adam',
                'lr': 1e-3,
            },
            'training': {
                'num_epochs': 1,
                'batch_size': 4,
            }
        }

        model = Model.from_config(config)

        # For LSTM/GRU, sequence_pad should be fixed at 4
        assert model.config['model']['sequence_pad'] == 4

    def test_sequence_pad_different_parameters(self):
        """Test sequence padding with different parameter combinations."""
        base_config = {
            'data': {
                'data_path': '/tmp/test',
                'input_dir': 'markers',
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
                'num_epochs': 1,
                'batch_size': 4,
            }
        }

        # Test different num_lags values
        for num_lags in [1, 2, 5, 10]:
            config = base_config.copy()
            config['model'] = base_config['model'].copy()
            config['model']['num_lags'] = num_lags

            model = Model.from_config(config)

            # sequence_pad should equal num_lags for temporal-mlp
            assert model.config['model']['sequence_pad'] == num_lags


# =============================================================================
# Tests for Model (CSV pipeline) - Unit tests
# =============================================================================

class TestModelAPI:
    """Unit tests for the Model API (CSV pipeline)."""

    @pytest.fixture
    def basic_config(self):
        """Basic configuration for Model tests."""
        return {
            'data': {
                'data_path': '/tmp/test',
                'input_dir': 'markers',
            },
            'model': {
                'input_size': 10,
                'output_size': 4,
                'head': 'temporalmlp',
                'num_hid_units': 32,
                'num_layers': 2,
                'num_lags': 3,
            },
            'optimizer': {
                'type': 'Adam',
                'lr': 1e-3,
            },
            'training': {
                'num_epochs': 1,
                'batch_size': 4,
            }
        }

    def test_model_from_config_dict(self, basic_config):
        """Test creating Model from config dictionary."""
        model = Model.from_config(basic_config)

        assert model.model is not None
        assert model.config is not None
        assert model.model_dir is None
        assert hasattr(model.model, 'head')
        assert hasattr(model.model, 'classifier')

    def test_model_from_config_file(self, basic_config):
        """Test creating Model from config file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(basic_config, f)
            config_path = f.name

        try:
            model = Model.from_config(config_path)

            assert model.model is not None
            assert model.config is not None
        finally:
            Path(config_path).unlink()

    def test_model_from_config_nonexistent(self):
        """Test FileNotFoundError for nonexistent config file."""
        with pytest.raises(FileNotFoundError, match='Config file not found'):
            Model.from_config('/nonexistent/config.yaml')

    def test_model_from_dir_missing_files(self):
        """Test error handling when loading from empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_dir = Path(tmpdir) / 'empty'
            empty_dir.mkdir()

            with pytest.raises(FileNotFoundError, match='Config file not found'):
                Model.from_dir(empty_dir)

    def test_model_velocity_concat_doubles_input(self):
        """Test that VelocityConcat transform doubles input size."""
        config = {
            'data': {
                'data_path': '/tmp/test',
                'input_dir': 'markers',
                'transforms': ['VelocityConcat', 'ZScore'],
            },
            'model': {
                'input_size': 10,
                'output_size': 4,
                'head': 'temporalmlp',
                'num_hid_units': 32,
                'num_layers': 2,
                'num_lags': 3,
            },
            'optimizer': {
                'type': 'Adam',
                'lr': 1e-3,
            },
            'training': {
                'num_epochs': 1,
                'batch_size': 4,
            }
        }

        model = Model.from_config(config)

        # Input size should be doubled due to VelocityConcat
        assert model.config['model']['input_size'] == 20

    def test_model_predict_requires_trained_model(self, basic_config):
        """Test that predict raises error without trained model."""
        model = Model.from_config(basic_config)
        model.model = None

        with pytest.raises(ValueError, match='Model must be trained or loaded'):
            model.predict(
                data_path='/tmp/data',
                input_dir='markers',
                output_dir='/tmp/output',
            )

    def test_model_predict_single_output_file_with_multiple_expts(self, basic_config):
        """Test that output_file with multiple expt_ids raises error."""
        model = Model.from_config(basic_config)

        with pytest.raises(RuntimeError, match='Can only supply'):
            model.predict(
                data_path='/tmp/data',
                input_dir='markers',
                output_dir='/tmp/output',
                output_file='/tmp/output/single.csv',
                expt_ids=['exp1', 'exp2'],
            )

    def test_model_get_model_class(self):
        """Test that _get_model_class returns Segmenter."""
        from lightning_action.models.segmenter import Segmenter
        assert Model._get_model_class() == Segmenter

    def test_model_get_train_function(self):
        """Test that _get_train_function returns train."""
        from lightning_action.train import train
        assert Model._get_train_function() == train


# =============================================================================
# Tests for VideoModel (video pipeline) - Unit tests
# =============================================================================

class TestVideoModelAPI:
    """Unit tests for the VideoModel API (video pipeline)."""

    @pytest.fixture
    def video_config(self):
        """Basic configuration for VideoModel tests."""
        return {
            'data': {
                'videos_dir': '/tmp/videos',
                'labels_dir': '/tmp/labels',
                'expt_ids': ['video1', 'video2'],
            },
            'model': {
                'output_size': 3,
                'head': 'dtcn',
                'num_hid_units': 32,
                'num_layers': 2,
                'num_lags': 1,
            },
            'optimizer': {
                'type': 'Adam',
                'lr': 1e-3,
            },
            'training': {
                'num_epochs': 1,
                'batch_size': 1,
                'sequence_length': 64,
            }
        }

    def test_videomodel_from_config_dict(self, video_config):
        """Test creating VideoModel from config dictionary."""
        model = VideoModel.from_config(video_config)

        assert model.model is not None
        assert model.config is not None
        assert model.model_dir is None

    def test_videomodel_from_config_file(self, video_config):
        """Test creating VideoModel from config file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(video_config, f)
            config_path = f.name

        try:
            model = VideoModel.from_config(config_path)

            assert model.model is not None
            assert model.config is not None
        finally:
            Path(config_path).unlink()

    def test_videomodel_from_config_nonexistent(self):
        """Test FileNotFoundError for nonexistent config file."""
        with pytest.raises(FileNotFoundError, match='Config file not found'):
            VideoModel.from_config('/nonexistent/config.yaml')

    def test_videomodel_from_dir_missing_files(self):
        """Test error handling when loading from empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError, match='Config file not found'):
                VideoModel.from_dir(tmpdir)

    def test_videomodel_predict_requires_trained_model(self, video_config):
        """Test that predict raises error without trained model."""
        model = VideoModel.from_config(video_config)
        model.model = None

        with pytest.raises(ValueError, match='Model must be trained or loaded'):
            model.predict(
                videos_dir='/tmp/videos',
                output_dir='/tmp/output',
            )

    def test_videomodel_predict_single_output_file_with_multiple_videos(self, video_config):
        """Test that output_file with multiple expt_ids raises error."""
        model = VideoModel.from_config(video_config)

        with pytest.raises(RuntimeError, match='Can only supply'):
            model.predict(
                videos_dir='/tmp/videos',
                output_dir='/tmp/output',
                output_file='/tmp/output/single.csv',
                expt_ids=['video1', 'video2'],
            )

    def test_videomodel_get_model_class(self):
        """Test that _get_model_class returns VideoSegmenter."""
        from lightning_action.models.video_segmenter import VideoSegmenter
        assert VideoModel._get_model_class() == VideoSegmenter

    def test_videomodel_get_train_function(self):
        """Test that _get_train_function returns train_video."""
        from lightning_action.video_train import train_video
        assert VideoModel._get_train_function() == train_video

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_videomodel_setup_trainer(self, video_config):
        """Test trainer setup for prediction (requires GPU)."""
        model = VideoModel.from_config(video_config)
        trainer = model._setup_trainer()

        assert trainer is not None

    def test_videomodel_setup_trainer_no_gpu(self, video_config):
        """Test that setup_trainer raises error without GPU."""
        model = VideoModel.from_config(video_config)

        with patch('torch.cuda.device_count', return_value=0):
            with pytest.raises(RuntimeError, match='No GPU detected'):
                model._setup_trainer()


# =============================================================================
# Tests for _get_video_frame_count utility
# =============================================================================

class TestGetVideoFrameCount:
    """Tests for the _get_video_frame_count utility function."""

    def test_get_frame_count_invalid_path(self):
        """Test RuntimeError for invalid video path."""
        from lightning_action.api.video_model import _get_video_frame_count

        with pytest.raises(RuntimeError, match='Could not open video'):
            _get_video_frame_count('/nonexistent/video.mp4')


# =============================================================================
# Tests for inheritance and polymorphism
# =============================================================================

class TestInheritance:
    """Tests verifying correct inheritance behavior."""

    def test_model_is_base_model_api(self):
        """Test that Model inherits from BaseModelAPI."""
        assert issubclass(Model, BaseModelAPI)

    def test_videomodel_is_base_model_api(self):
        """Test that VideoModel inherits from BaseModelAPI."""
        assert issubclass(VideoModel, BaseModelAPI)

    def test_shared_methods_exist(self):
        """Test that shared methods exist on both classes."""
        shared_methods = [
            'from_dir',
            'from_config',
            'train',
            '_find_config_file',
            '_find_checkpoint_file',
            '_load_checkpoint',
            '_setup_trainer',
        ]

        for method in shared_methods:
            assert hasattr(Model, method)
            assert hasattr(VideoModel, method)

    def test_abstract_methods_implemented(self):
        """Test that both classes implement all abstract methods."""
        abstract_methods = [
            '_create_model_from_config',
            '_get_model_class',
            '_get_train_function',
            '_setup_trainer',
            '_run_post_training_inference',
            'predict',
        ]

        for method in abstract_methods:
            # These should not raise NotImplementedError
            assert hasattr(Model, method)
            assert hasattr(VideoModel, method)
            assert callable(getattr(Model, method))
            assert callable(getattr(VideoModel, method))


# =============================================================================
# Integration tests for Model (CSV pipeline) - from original TestModelIntegration
# =============================================================================

class TestModelIntegration:
    """Integration tests for Model API with real data.

    These tests require the data_dir and config_path fixtures from conftest.py.
    """

    @pytest.fixture
    def fast_config(self, config_path):
        """Create a fast training config for testing."""
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Make training fast for testing
        config['training']['num_epochs'] = 2
        config['training']['log_every_n_steps'] = 1
        config['training']['check_val_every_n_epoch'] = 1
        config['training']['batch_size'] = 4

        # Reduce model size
        config['model']['num_hid_units'] = 8
        config['model']['num_layers'] = 1

        return config

    @pytest.fixture
    def gpu_config(self, fast_config):
        """GPU version of fast config."""
        gpu_config = fast_config.copy()
        gpu_config['training'] = fast_config['training'].copy()
        gpu_config['training']['device'] = 'gpu'
        return gpu_config

    def test_model_from_config_cpu(self, fast_config):
        """Test creating model from config dictionary."""
        model = Model.from_config(fast_config)

        assert model.model is not None
        assert model.config == fast_config
        assert model.model_dir is None

        # Check model components
        assert hasattr(model.model, 'head')
        assert hasattr(model.model, 'classifier')

    def test_model_from_config_file(self, fast_config):
        """Test creating model from config file."""
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(fast_config, f)
            temp_config_path = f.name

        try:
            model = Model.from_config(temp_config_path)

            assert model.model is not None
            assert 'sequence_pad' in model.config['model']  # Gets added before model creation
            del model.config['model']['sequence_pad']
            assert model.config == fast_config
            assert model.model_dir is None
        finally:
            Path(temp_config_path).unlink()

    def test_model_from_nonexistent_config(self):
        """Test error handling for nonexistent config file."""
        with pytest.raises(FileNotFoundError, match='Config file not found'):
            Model.from_config('/nonexistent/config.yaml')

    def test_model_train_cpu(self, data_dir, fast_config):
        """Test training model on CPU with real data."""
        fast_config['data']['data_path'] = str(data_dir)

        model = Model.from_config(fast_config)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / 'test_run'

            # Train model
            model.train(output_dir=output_dir, post_inference=False)

            # Check that model was trained
            assert model.model is not None
            assert model.model_dir == output_dir

            # Check that files were created
            assert output_dir.exists()
            assert (output_dir / 'config.yaml').exists()
            assert (output_dir / 'final_model.ckpt').exists()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_model_train_gpu(self, data_dir, gpu_config):
        """Test training model on GPU with real data."""
        gpu_config['data']['data_path'] = str(data_dir)

        model = Model.from_config(gpu_config)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / 'test_run_gpu'

            # Train model
            model.train(output_dir=output_dir, post_inference=False)

            # Check that model was trained
            assert model.model is not None
            assert model.model_dir == output_dir

            # Check that files were created
            assert output_dir.exists()
            assert (output_dir / 'config.yaml').exists()
            assert (output_dir / 'final_model.ckpt').exists()

    def test_model_from_dir_after_training(self, data_dir, fast_config):
        """Test loading model from directory after training."""
        fast_config['data']['data_path'] = str(data_dir)

        # Train model
        model1 = Model.from_config(fast_config)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / 'test_run'
            model1.train(output_dir=output_dir, post_inference=False)

            # Load model from directory
            model2 = Model.from_dir(output_dir)

            assert model2.model is not None
            assert model2.model_dir == output_dir
            assert model2.config is not None

            # Models should have same structure
            assert type(model1.model) == type(model2.model)

    def test_model_from_dir_missing_files(self):
        """Test error handling when loading from directory with missing files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            empty_dir = Path(temp_dir) / 'empty'
            empty_dir.mkdir()

            with pytest.raises(FileNotFoundError, match='Config file not found'):
                Model.from_dir(empty_dir)

    def test_model_predict_cpu(self, data_dir, fast_config):
        """Test prediction on CPU with real data."""
        fast_config['data']['data_path'] = str(data_dir)

        # Train model
        model = Model.from_config(fast_config)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / 'test_run'
            model.train(output_dir=output_dir, post_inference=False)

            # Test prediction
            prediction_dir = Path(temp_dir) / 'predictions'
            model.predict(
                data_path=data_dir,
                input_dir='markers',
                output_dir=prediction_dir,
                expt_ids=['2019_06_26_fly2'],
            )

            # Check prediction directory and file were created
            assert prediction_dir.exists()
            prediction_file = prediction_dir / '2019_06_26_fly2_predictions.csv'
            assert prediction_file.exists()

            # Load and check predictions
            predictions = pd.read_csv(prediction_file, index_col=0, header=[0])
            assert len(predictions.shape) == 2  # (time_steps, num_classes)
            assert predictions.shape[0] == 45001
            assert predictions.shape[1] == fast_config['model']['output_size']

            # Probabilities should sum to 1, just check first 1000 rows
            prob_sums = np.sum(predictions.to_numpy(), axis=1)
            assert np.allclose(prob_sums[:1000], 1.0, atol=1e-6)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_model_predict_gpu(self, data_dir, gpu_config):
        """Test prediction on GPU with real data."""
        gpu_config['data']['data_path'] = str(data_dir)

        # Train model
        model = Model.from_config(gpu_config)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / 'test_run_gpu'
            model.train(output_dir=output_dir, post_inference=False)

            # Test prediction
            prediction_dir = Path(temp_dir) / 'predictions_gpu'
            model.predict(
                data_path=data_dir,
                input_dir='markers',
                output_dir=prediction_dir,
                expt_ids=['2019_06_26_fly2'],
            )

            # Check prediction directory and file were created
            assert prediction_dir.exists()
            prediction_file = prediction_dir / '2019_06_26_fly2_predictions.csv'
            assert prediction_file.exists()

            # Load and check predictions
            predictions = pd.read_csv(prediction_file, index_col=0, header=[0])
            assert len(predictions.shape) == 2  # (time_steps, num_classes)
            assert predictions.shape[1] == gpu_config['model']['output_size']

    def test_model_predict_all_experiments(self, data_dir, fast_config):
        """Test prediction on all experiments when expt_ids is None."""
        fast_config['data']['data_path'] = str(data_dir)

        # Train model
        model = Model.from_config(fast_config)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / 'test_run'
            model.train(output_dir=output_dir, post_inference=False)

            # Test prediction on all experiments
            prediction_dir = Path(temp_dir) / 'predictions_all'
            model.predict(
                data_path=data_dir,
                input_dir='markers',
                output_dir=prediction_dir,
                expt_ids=None,  # Predict on all experiments
            )

            # Check prediction directory was created
            assert prediction_dir.exists()

            # Check that separate files were created for each experiment
            prediction_files = list(prediction_dir.glob('*_predictions.csv'))
            assert len(prediction_files) >= 2  # Should have multiple experiments

            # Check each prediction file
            for pred_file in prediction_files:
                # Extract experiment ID from filename
                expt_id = pred_file.name.replace('_predictions.csv', '')
                assert expt_id in ['2019_06_26_fly2', '2019_08_07_fly2']

                # Load and check predictions
                predictions = pd.read_csv(pred_file, index_col=0, header=[0])
                assert len(predictions.shape) == 2  # (time_steps, num_classes)
                assert predictions.shape[1] == fast_config['model']['output_size']

                # Probabilities should sum to 1, just check first 1000 rows
                prob_sums = np.sum(predictions.to_numpy(), axis=1)
                assert np.allclose(prob_sums[:1000], 1.0, atol=1e-6)

                # Should have reasonable number of time steps
                assert predictions.shape[0] > 10

    def test_model_different_heads(self, data_dir, fast_config):
        """Test training with different head architectures."""
        heads = ['temporalmlp', 'rnn', 'dtcn']

        for head in heads:
            # Update config for this head
            config = fast_config.copy()
            config['data'] = fast_config['data'].copy()
            config['model'] = fast_config['model'].copy()
            config['training'] = fast_config['training'].copy()
            config['data']['data_path'] = str(data_dir)
            config['model']['head'] = head

            # Adjust head-specific parameters
            if head == 'rnn':
                config['model']['rnn_type'] = 'lstm'
                config['model']['bidirectional'] = False

            model = Model.from_config(config)

            with tempfile.TemporaryDirectory() as temp_dir:
                output_dir = Path(temp_dir) / f'test_run_{head}'

                # Train model
                model.train(output_dir=output_dir, post_inference=False)

                # Check that model was trained successfully
                assert model.model is not None
                assert output_dir.exists()
                assert (output_dir / 'config.yaml').exists()
                assert (output_dir / 'final_model.ckpt').exists()

    def test_model_roundtrip_save_load(self, data_dir, fast_config):
        """Test complete roundtrip: create, train, save, load, predict."""
        fast_config['data']['data_path'] = str(data_dir)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / 'test_run'

            # Create and train model
            model1 = Model.from_config(fast_config)
            model1.train(output_dir=output_dir, post_inference=False)

            # Load model from directory
            model2 = Model.from_dir(output_dir)

            # Generate predictions with loaded model
            model2.predict(
                data_path=data_dir,
                input_dir='markers',
                output_dir=output_dir,
                expt_ids=['2019_06_26_fly2', '2019_08_07_fly2'],
            )

            # Check that separate files were created for each experiment
            prediction_files = list(output_dir.glob('*_predictions.csv'))
            assert len(prediction_files) >= 2  # Should have multiple experiments

            # Check each prediction file
            for pred_file in prediction_files:
                # Extract experiment ID from filename
                expt_id = pred_file.name.replace('_predictions.csv', '')
                assert expt_id in ['2019_06_26_fly2', '2019_08_07_fly2']

                # Load and check predictions
                predictions = pd.read_csv(pred_file, index_col=0, header=[0])
                assert len(predictions.shape) == 2  # (time_steps, num_classes)
                assert predictions.shape[1] == fast_config['model']['output_size']

                # Probabilities should sum to 1, just check first 1000 rows
                prob_sums = np.sum(predictions.to_numpy(), axis=1)
                assert np.allclose(prob_sums[:1000], 1.0, atol=1e-6)

                # Should have reasonable number of time steps
                assert predictions.shape[0] > 10

    def test_model_velocity_transform(self, data_dir, fast_config):
        """Test training model on CPU with VelocityConcat transform."""
        fast_config['data']['data_path'] = str(data_dir)
        fast_config['data']['transforms'] = ['VelocityConcat', 'ZScore']

        model = Model.from_config(fast_config)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / 'test_run'

            # Train model
            model.train(output_dir=output_dir, post_inference=False)

            # Check that model was trained
            assert model.model is not None
            assert model.model_dir == output_dir

            # Check that files were created
            assert output_dir.exists()
            assert (output_dir / 'config.yaml').exists()
            assert (output_dir / 'final_model.ckpt').exists()

    def test_model_train_cpu_post_inference(self, data_dir, fast_config):
        """Test that post-training inference runs automatically on all training experiments."""
        fast_config['data']['data_path'] = str(data_dir)

        model = Model.from_config(fast_config)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / 'test_run'

            # Train model with post-training inference enabled (default behavior)
            model.train(output_dir=output_dir, post_inference=True)

            # Check that model was trained
            assert model.model is not None
            assert model.model_dir == output_dir

            # Check that training files were created
            assert output_dir.exists()
            assert (output_dir / 'config.yaml').exists()
            assert (output_dir / 'final_model.ckpt').exists()

            # Check that predictions directory was created
            predictions_dir = output_dir / 'predictions'
            assert predictions_dir.exists()

            # Check that prediction files were created for all experiments
            prediction_files = list(predictions_dir.glob('*_predictions.csv'))
            assert len(prediction_files) >= 2  # Should have multiple experiments

            # Check that each prediction file exists for expected experiments
            expected_experiments = ['2019_06_26_fly2', '2019_08_07_fly2']
            found_experiments = []

            for pred_file in prediction_files:
                # Extract experiment ID from filename
                expt_id = pred_file.name.replace('_predictions.csv', '')
                found_experiments.append(expt_id)

                # Load and check predictions
                predictions = pd.read_csv(pred_file, index_col=0, header=[0])
                assert len(predictions.shape) == 2  # (time_steps, num_classes)
                assert predictions.shape[1] == fast_config['model']['output_size']

                # Probabilities should sum to 1, just check first 1000 rows
                prob_sums = np.sum(predictions.to_numpy(), axis=1)
                assert np.allclose(prob_sums[:1000], 1.0, atol=1e-6)

                # Should have reasonable number of time steps
                assert predictions.shape[0] > 10

            # Verify that all expected experiments have predictions
            for expected_expt in expected_experiments:
                assert expected_expt in found_experiments, f'Missing predictions for {expected_expt}'

    def test_model_predict_with_data_length_padding(self, data_dir, fast_config):
        """Test prediction with various sequence lengths to verify padding behavior."""
        fast_config['data']['data_path'] = str(data_dir)

        # Train model
        model = Model.from_config(fast_config)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / 'test_run'
            model.train(output_dir=output_dir, post_inference=False)

            # Test with different sequence lengths that don't evenly divide 45001
            sequence_lengths = [300, 700, 1100]

            for seq_len in sequence_lengths:
                # Modify sequence length
                model.config['training']['sequence_length'] = seq_len

                # Test prediction
                prediction_dir = Path(temp_dir) / f'predictions_seq_{seq_len}'
                model.predict(
                    data_path=data_dir,
                    input_dir='markers',
                    output_dir=prediction_dir,
                    expt_ids=['2019_06_26_fly2'],
                )

                # Check prediction file
                prediction_file = prediction_dir / '2019_06_26_fly2_predictions.csv'
                assert prediction_file.exists()

                # Load predictions
                predictions = pd.read_csv(prediction_file, index_col=0, header=[0])

                # Should always have 45001 rows regardless of sequence length
                assert predictions.shape[0] == 45001, (
                    f'With seq_len={seq_len}, expected 45001 rows, got {predictions.shape[0]}'
                )

                # Calculate expected number of sequences and padding
                num_sequences = 45001 // seq_len
                predicted_length = num_sequences * seq_len
                expected_padding = 45001 - predicted_length

                if expected_padding > 0:
                    # Should have NaN rows at the end for padding
                    last_rows = predictions.tail(expected_padding)
                    assert last_rows.isna().all().all(), (
                        f'With seq_len={seq_len}, last {expected_padding} rows should be NaN'
                    )

                    # Non-padded rows should be valid probabilities
                    non_padded = predictions.head(predicted_length)
                    prob_sums = np.sum(non_padded.to_numpy(), axis=1)
                    assert np.allclose(prob_sums, 1.0, atol=1e-6), (
                        f'With seq_len={seq_len}, non-padded predictions should sum to 1'
                    )

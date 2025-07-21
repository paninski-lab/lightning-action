"""Integration tests for the Model API using real data."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from lightning_action.api.model import Model


class TestModelIntegration:
    """Integration tests for Model API with real data."""

    @pytest.fixture
    def fast_config(self, config_path):
        """Create a fast training config for testing."""
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        # make training fast for testing
        config['training']['num_epochs'] = 2
        config['training']['log_every_n_steps'] = 1
        config['training']['check_val_every_n_epoch'] = 1
        config['training']['batch_size'] = 4
        
        # reduce model size
        config['model']['num_hid_units'] = 8
        config['model']['num_layers'] = 1
        
        return config

    @pytest.fixture
    def gpu_config(self, fast_config):
        """GPU version of fast config."""
        gpu_config = fast_config.copy()
        gpu_config['training']['device'] = 'gpu'
        return gpu_config

    def test_model_from_config_cpu(self, fast_config):
        """Test creating model from config dictionary."""
        model = Model.from_config(fast_config)
        
        assert model.model is not None
        assert model.config == fast_config
        assert model.model_dir is None
        
        # check model components
        assert hasattr(model.model, 'backbone')
        assert hasattr(model.model, 'classifier')

    def test_model_from_config_file(self, fast_config):
        """Test creating model from config file."""
        # create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(fast_config, f)
            temp_config_path = f.name
        
        try:
            model = Model.from_config(temp_config_path)
            
            assert model.model is not None
            assert 'sequence_pad' in model.config['model']  # this gets added before model creation
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
        # update data path to be absolute
        fast_config['data']['data_path'] = str(data_dir)
        
        model = Model.from_config(fast_config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / 'test_run'
            
            # train model
            model.train(output_dir=output_dir, post_inference=False)
            
            # check that model was trained
            assert model.model is not None
            assert model.model_dir == output_dir
            
            # check that files were created
            assert output_dir.exists()
            assert (output_dir / 'config.yaml').exists()
            assert (output_dir / 'final_model.ckpt').exists()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_model_train_gpu(self, data_dir, gpu_config):
        """Test training model on GPU with real data."""
        # update data path to be absolute
        gpu_config['data']['data_path'] = str(data_dir)
        
        model = Model.from_config(gpu_config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / 'test_run_gpu'
            
            # train model
            model.train(output_dir=output_dir, post_inference=False)
            
            # check that model was trained
            assert model.model is not None
            assert model.model_dir == output_dir
            
            # check that files were created
            assert output_dir.exists()
            assert (output_dir / 'config.yaml').exists()
            assert (output_dir / 'final_model.ckpt').exists()

    def test_model_from_dir_after_training(self, data_dir, fast_config):
        """Test loading model from directory after training."""
        # update data path to be absolute
        fast_config['data']['data_path'] = str(data_dir)
        
        # train model
        model1 = Model.from_config(fast_config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / 'test_run'
            model1.train(output_dir=output_dir)
            
            # load model from directory
            model2 = Model.from_dir(output_dir)
            
            assert model2.model is not None
            assert model2.model_dir == output_dir
            assert model2.config is not None
            
            # models should have same structure
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
        # update data path to be absolute
        fast_config['data']['data_path'] = str(data_dir)
        
        # train model
        model = Model.from_config(fast_config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / 'test_run'
            model.train(output_dir=output_dir, post_inference=False)
            
            # test prediction
            prediction_dir = Path(temp_dir) / 'predictions'
            model.predict(
                data_path=data_dir,
                input_dir='markers',
                output_dir=prediction_dir,
                expt_ids=['2019_06_26_fly2'],
            )
            
            # check prediction directory and file were created
            assert prediction_dir.exists()
            prediction_file = prediction_dir / '2019_06_26_fly2_predictions.csv'
            assert prediction_file.exists()
            
            # load and check predictions
            predictions = pd.read_csv(prediction_file, index_col=0, header=[0])
            assert len(predictions.shape) == 2  # (time_steps, num_classes)
            assert predictions.shape[0] < 50000
            assert predictions.shape[1] == fast_config['model']['output_size']
            
            # probabilities should sum to 1
            prob_sums = np.sum(predictions.to_numpy(), axis=1)
            assert np.allclose(prob_sums, 1.0, atol=1e-6)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_model_predict_gpu(self, data_dir, gpu_config):
        """Test prediction on GPU with real data."""
        # update data path to be absolute
        gpu_config['data']['data_path'] = str(data_dir)
        
        # train model
        model = Model.from_config(gpu_config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / 'test_run_gpu'
            model.train(output_dir=output_dir, post_inference=False)
            
            # test prediction
            prediction_dir = Path(temp_dir) / 'predictions_gpu'
            model.predict(
                data_path=data_dir,
                input_dir='markers',
                output_dir=prediction_dir,
                expt_ids=['2019_06_26_fly2'],
            )
            
            # check prediction directory and file were created
            assert prediction_dir.exists()
            prediction_file = prediction_dir / '2019_06_26_fly2_predictions.csv'
            assert prediction_file.exists()
            
            # load and check predictions
            predictions = pd.read_csv(prediction_file, index_col=0, header=[0])
            assert len(predictions.shape) == 2  # (time_steps, num_classes)
            assert predictions.shape[1] == gpu_config['model']['output_size']

    def test_model_predict_all_experiments(self, data_dir, fast_config):
        """Test prediction on all experiments when expt_ids is None."""
        # update data path to be absolute
        fast_config['data']['data_path'] = str(data_dir)
        
        # train model
        model = Model.from_config(fast_config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / 'test_run'
            model.train(output_dir=output_dir, post_inference=False)
            
            # test prediction on all experiments
            prediction_dir = Path(temp_dir) / 'predictions_all'
            model.predict(
                data_path=data_dir,
                input_dir='markers',
                output_dir=prediction_dir,
                expt_ids=None,  # predict on all experiments
            )
            
            # check prediction directory was created
            assert prediction_dir.exists()
            
            # check that separate files were created for each experiment
            prediction_files = list(prediction_dir.glob('*_predictions.csv'))
            assert len(prediction_files) >= 2  # should have multiple experiments
            
            # check each prediction file
            for pred_file in prediction_files:
                # extract experiment ID from filename
                expt_id = pred_file.name.replace('_predictions.csv', '')
                assert expt_id in ['2019_06_26_fly2', '2019_08_07_fly2']
                
                # load and check predictions
                predictions = pd.read_csv(pred_file, index_col=0, header=[0])
                assert len(predictions.shape) == 2  # (time_steps, num_classes)
                assert predictions.shape[1] == fast_config['model']['output_size']
                
                # probabilities should sum to 1
                prob_sums = np.sum(predictions.to_numpy(), axis=1)
                assert np.allclose(prob_sums, 1.0, atol=1e-6)
                
                # should have reasonable number of time steps
                assert predictions.shape[0] > 10

    def test_model_different_backbones(self, data_dir, fast_config):
        """Test training with different backbone architectures."""
        backbones = ['temporalmlp', 'rnn', 'dtcn']
        
        for backbone in backbones:
            # update config for this backbone
            config = fast_config.copy()
            config['data']['data_path'] = str(data_dir)
            config['model']['backbone'] = backbone
            
            # adjust backbone-specific parameters
            if backbone == 'rnn':
                config['model']['rnn_type'] = 'lstm'
                config['model']['bidirectional'] = False
            
            model = Model.from_config(config)
            
            with tempfile.TemporaryDirectory() as temp_dir:
                output_dir = Path(temp_dir) / f'test_run_{backbone}'
                
                # train model
                model.train(output_dir=output_dir, post_inference=False)
                
                # check that model was trained successfully
                assert model.model is not None
                assert output_dir.exists()
                assert (output_dir / 'config.yaml').exists()
                assert (output_dir / 'final_model.ckpt').exists()

    def test_model_roundtrip_save_load(self, data_dir, fast_config):
        """Test complete roundtrip: create, train, save, load, predict."""
        # update data path to be absolute
        fast_config['data']['data_path'] = str(data_dir)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / 'test_run'
            
            # create and train model
            model1 = Model.from_config(fast_config)
            model1.train(output_dir=output_dir)
            
            # load model from directory
            model2 = Model.from_dir(output_dir)
            
            # generate predictions with loaded model
            model2.predict(
                data_path=data_dir,
                input_dir='markers',
                output_dir=output_dir,
                expt_ids=['2019_06_26_fly2', '2019_08_07_fly2'],
            )

            # check that separate files were created for each experiment
            prediction_files = list(output_dir.glob('*_predictions.csv'))
            assert len(prediction_files) >= 2  # should have multiple experiments

            # check each prediction file
            for pred_file in prediction_files:
                # extract experiment ID from filename
                expt_id = pred_file.name.replace('_predictions.csv', '')
                assert expt_id in ['2019_06_26_fly2', '2019_08_07_fly2']

                # load and check predictions
                predictions = pd.read_csv(pred_file, index_col=0, header=[0])
                assert len(predictions.shape) == 2  # (time_steps, num_classes)
                assert predictions.shape[1] == fast_config['model']['output_size']

                # probabilities should sum to 1
                prob_sums = np.sum(predictions.to_numpy(), axis=1)
                assert np.allclose(prob_sums, 1.0, atol=1e-6)

                # should have reasonable number of time steps
                assert predictions.shape[0] > 10

    def test_model_velocity_transform(self, data_dir, fast_config):
        """Test training model on CPU with real data."""
        # update data path to be absolute
        fast_config['data']['data_path'] = str(data_dir)
        fast_config['data']['transforms'] = ['VelocityConcat', 'ZScore']

        model = Model.from_config(fast_config)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / 'test_run'

            # train model
            model.train(output_dir=output_dir, post_inference=False)

            # check that model was trained
            assert model.model is not None
            assert model.model_dir == output_dir

            # check that files were created
            assert output_dir.exists()
            assert (output_dir / 'config.yaml').exists()
            assert (output_dir / 'final_model.ckpt').exists()

    def test_model_train_cpu_post_inference(self, data_dir, fast_config):
        """Test that post-training inference runs automatically on all training experiments."""
        # update data path to be absolute
        fast_config['data']['data_path'] = str(data_dir)
        
        model = Model.from_config(fast_config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / 'test_run'
            
            # train model with post-training inference enabled (default behavior)
            model.train(output_dir=output_dir, post_inference=True)
            
            # check that model was trained
            assert model.model is not None
            assert model.model_dir == output_dir
            
            # check that training files were created
            assert output_dir.exists()
            assert (output_dir / 'config.yaml').exists()
            assert (output_dir / 'final_model.ckpt').exists()
            
            # check that predictions directory was created
            predictions_dir = output_dir / 'predictions'
            assert predictions_dir.exists()
            
            # check that prediction files were created for all experiments
            prediction_files = list(predictions_dir.glob('*_predictions.csv'))
            assert len(prediction_files) >= 2  # should have multiple experiments
            
            # check that each prediction file exists for expected experiments
            expected_experiments = ['2019_06_26_fly2', '2019_08_07_fly2']
            found_experiments = []
            
            for pred_file in prediction_files:
                # extract experiment ID from filename
                expt_id = pred_file.name.replace('_predictions.csv', '')
                found_experiments.append(expt_id)

                # load and check predictions
                predictions = pd.read_csv(pred_file, index_col=0, header=[0])
                assert len(predictions.shape) == 2  # (time_steps, num_classes)
                assert predictions.shape[1] == fast_config['model']['output_size']

                # probabilities should sum to 1
                prob_sums = np.sum(predictions.to_numpy(), axis=1)
                assert np.allclose(prob_sums, 1.0, atol=1e-6)

                # should have reasonable number of time steps
                assert predictions.shape[0] > 10
            
            # verify that all expected experiments have predictions
            for expected_expt in expected_experiments:
                assert expected_expt in found_experiments, f'Missing predictions for {expected_expt}'


class TestModelSequencePadding:
    """Test sequence padding computation in Model.from_config."""
    
    def test_sequence_pad_temporal_mlp(self):
        """Test sequence padding computation for TemporalMLP backbone."""
        config = {
            'data': {
                'data_path': '/tmp/test',
                'input_dir': 'markers',
            },
            'model': {
                'input_size': 10,
                'output_size': 4,
                'backbone': 'temporalmlp',
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
        """Test sequence padding computation for DilatedTCN backbone."""
        config = {
            'data': {
                'data_path': '/tmp/test',
                'input_dir': 'markers',
            },
            'model': {
                'input_size': 10,
                'output_size': 4,
                'backbone': 'dtcn',
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
        """Test sequence padding computation for LSTM backbone."""
        config = {
            'data': {
                'data_path': '/tmp/test',
                'input_dir': 'markers',
            },
            'model': {
                'input_size': 10,
                'output_size': 4,
                'backbone': 'rnn',
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
        """Test sequence padding computation for GRU backbone."""
        config = {
            'data': {
                'data_path': '/tmp/test',
                'input_dir': 'markers',
            },
            'model': {
                'input_size': 10,
                'output_size': 4,
                'backbone': 'rnn',
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
                'backbone': 'temporalmlp',
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
            config['model']['num_lags'] = num_lags
            
            model = Model.from_config(config)
            
            # sequence_pad should equal num_lags for temporal-mlp
            assert model.config['model']['sequence_pad'] == num_lags

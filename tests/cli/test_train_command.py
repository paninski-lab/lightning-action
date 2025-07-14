"""Tests for CLI train command."""

import tempfile
import yaml
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lightning_action.cli.commands.train import (
    apply_config_overrides,
    apply_overrides,
    handle,
    parse_config_value,
    register_parser,
)


class TestParseConfigValue:
    """Test config value parsing."""

    def test_parse_boolean_values(self):
        """Test parsing boolean values."""
        assert parse_config_value('true') is True
        assert parse_config_value('True') is True
        assert parse_config_value('false') is False
        assert parse_config_value('False') is False

    def test_parse_null_values(self):
        """Test parsing null values."""
        assert parse_config_value('null') is None
        assert parse_config_value('none') is None
        assert parse_config_value('None') is None

    def test_parse_integer_values(self):
        """Test parsing integer values."""
        assert parse_config_value('42') == 42
        assert parse_config_value('-10') == -10
        assert parse_config_value('0') == 0

    def test_parse_float_values(self):
        """Test parsing float values."""
        assert parse_config_value('3.14') == 3.14
        assert parse_config_value('-2.5') == -2.5
        assert parse_config_value('0.001') == 0.001

    def test_parse_list_values(self):
        """Test parsing comma-separated list values."""
        assert parse_config_value('a,b,c') == ['a', 'b', 'c']
        assert parse_config_value('1,2,3') == [1, 2, 3]
        assert parse_config_value('true,false') == [True, False]

    def test_parse_string_values(self):
        """Test parsing string values."""
        assert parse_config_value('hello') == 'hello'
        assert parse_config_value('test_string') == 'test_string'


class TestApplyConfigOverrides:
    """Test config override functionality."""

    def test_simple_override(self):
        """Test simple key-value override."""
        config = {'key1': 'old_value'}
        overrides = ['key1=new_value']
        
        result = apply_config_overrides(config, overrides)
        assert result['key1'] == 'new_value'

    def test_nested_override(self):
        """Test nested key override with dot notation."""
        config = {'section': {'key': 'old_value'}}
        overrides = ['section.key=new_value']
        
        result = apply_config_overrides(config, overrides)
        assert result['section']['key'] == 'new_value'

    def test_create_new_nested_key(self):
        """Test creating new nested keys."""
        config = {}
        overrides = ['new_section.new_key=value']
        
        result = apply_config_overrides(config, overrides)
        assert result['new_section']['new_key'] == 'value'

    def test_multiple_overrides(self):
        """Test multiple overrides."""
        config = {'a': 1, 'b': {'c': 2}}
        overrides = ['a=10', 'b.c=20', 'b.d=30']
        
        result = apply_config_overrides(config, overrides)
        assert result['a'] == 10
        assert result['b']['c'] == 20
        assert result['b']['d'] == 30

    def test_invalid_override_format(self):
        """Test error handling for invalid override format."""
        config = {}
        overrides = ['invalid_format']
        
        with pytest.raises(ValueError, match="Override must be in format 'key=value'"):
            apply_config_overrides(config, overrides)


class TestApplyOverrides:
    """Test applying command line overrides to config."""

    def test_data_path_override(self):
        """Test data path override."""
        config = {'data': {'data_path': '/old/path'}}
        args = MagicMock()
        args.data_path = '/new/path'
        args.device = None
        args.epochs = None
        args.batch_size = None
        args.lr = None
        args.seed = None
        args.overrides = None
        
        result = apply_overrides(config, args)
        assert result['data']['data_path'] == '/new/path'

    def test_training_overrides(self):
        """Test training parameter overrides."""
        config = {}
        args = MagicMock()
        args.data_path = None
        args.device = 'gpu'
        args.epochs = 100
        args.batch_size = 32
        args.lr = 0.001
        args.seed = 42
        args.overrides = None
        
        result = apply_overrides(config, args)
        assert result['training']['device'] == 'gpu'
        assert result['training']['num_epochs'] == 100
        assert result['training']['batch_size'] == 32
        assert result['optimizer']['lr'] == 0.001
        assert result['training']['seed'] == 42

    def test_custom_overrides(self):
        """Test custom overrides via --overrides."""
        config = {}
        args = MagicMock()
        args.data_path = None
        args.device = None
        args.epochs = None
        args.batch_size = None
        args.lr = None
        args.seed = None
        args.overrides = ['model.num_layers=5', 'training.device=cpu']
        
        result = apply_overrides(config, args)
        assert result['model']['num_layers'] == 5
        assert result['training']['device'] == 'cpu'


class TestHandle:
    """Test train command handler."""

    @pytest.fixture
    def test_config(self):
        """Create a test config."""
        return {
            'data': {'data_path': '/test/data'},
            'model': {'backbone': 'temporalmlp', 'num_hid_units': 64},
            'training': {'num_epochs': 10, 'device': 'cpu'},
            'optimizer': {'lr': 0.001},
        }

    @pytest.fixture
    def mock_args(self):
        """Create mock arguments."""
        args = MagicMock()
        args.output = None
        args.data_path = None
        args.device = None
        args.epochs = None
        args.batch_size = None
        args.lr = None
        args.seed = None
        args.overrides = None
        return args

    def test_handle_loads_config(self, test_config, mock_args):
        """Test that handle loads config from file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_config, f)
            config_path = f.name

        mock_args.config = config_path

        with patch('lightning_action.api.model.Model.from_config') as mock_model_class:
            mock_model = MagicMock()
            mock_model_class.return_value = mock_model
            
            handle(mock_args)
            
            # check that Model.from_config was called
            mock_model_class.assert_called_once()
            config_arg = mock_model_class.call_args[0][0]
            assert config_arg['data']['data_path'] == '/test/data'

        # cleanup
        Path(config_path).unlink()

    def test_handle_creates_output_dir(self, test_config, mock_args):
        """Test that handle creates output directory."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_config, f)
            config_path = f.name

        mock_args.config = config_path

        with patch('lightning_action.api.model.Model.from_config') as mock_model_class:
            mock_model = MagicMock()
            mock_model_class.return_value = mock_model
            
            handle(mock_args)
            
            # check that output directory was set
            assert mock_args.output is not None
            assert isinstance(mock_args.output, Path)

        # cleanup
        Path(config_path).unlink()

    def test_handle_calls_train(self, test_config, mock_args):
        """Test that handle calls model.train()."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_config, f)
            config_path = f.name

        mock_args.config = config_path

        with patch('lightning_action.api.model.Model.from_config') as mock_model_class:
            mock_model = MagicMock()
            mock_model_class.return_value = mock_model
            
            handle(mock_args)
            
            # check that train was called
            mock_model.train.assert_called_once()
            train_args = mock_model.train.call_args
            assert 'output_dir' in train_args[1]

        # cleanup
        Path(config_path).unlink()

    def test_handle_applies_overrides(self, test_config, mock_args):
        """Test that handle applies command line overrides."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_config, f)
            config_path = f.name

        mock_args.config = config_path
        mock_args.epochs = 50
        mock_args.device = 'gpu'

        with patch('lightning_action.api.model.Model.from_config') as mock_model_class:
            mock_model = MagicMock()
            mock_model_class.return_value = mock_model
            
            handle(mock_args)
            
            # check that overrides were applied
            config_arg = mock_model_class.call_args[0][0]
            assert config_arg['training']['num_epochs'] == 50
            assert config_arg['training']['device'] == 'gpu'

        # cleanup
        Path(config_path).unlink()

    def test_handle_error_handling(self, test_config, mock_args):
        """Test error handling in handle function."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_config, f)
            config_path = f.name

        mock_args.config = config_path

        with patch('lightning_action.api.model.Model.from_config') as mock_model_class:
            mock_model = MagicMock()
            mock_model.train.side_effect = Exception('Training failed')
            mock_model_class.return_value = mock_model
            
            with pytest.raises(Exception, match='Training failed'):
                handle(mock_args)

        # cleanup
        Path(config_path).unlink()

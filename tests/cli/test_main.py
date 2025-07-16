"""Tests for CLI main functionality."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import yaml

from lightning_action.cli.main import main


class TestMain:
    """Test the main CLI entry point."""

    @patch('lightning_action.cli.commands.train.handle')
    def test_main_calls_train_command(self, mock_train_handle):
        """Test that main calls train command handler."""
        # Create a temporary config file
        test_config_data = {
            'data': {'data_path': '/test/data'},
            'model': {'backbone': 'temporalmlp'},
            'training': {'num_epochs': 10}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_config_data, f)
            test_config = f.name
        
        try:
            with patch('sys.argv', ['lightning-action', 'train', '--config', test_config]):
                main()
                
            mock_train_handle.assert_called_once()
        finally:
            Path(test_config).unlink()

    @patch('lightning_action.cli.commands.predict.handle')
    def test_main_calls_predict_command(self, mock_predict_handle):
        """Test that main calls predict command handler."""
        # Create temporary directories
        with tempfile.TemporaryDirectory() as temp_model_dir:
            with tempfile.TemporaryDirectory() as temp_data_dir:
                test_model = temp_model_dir
                test_data = temp_data_dir
                
                with patch(
                    'sys.argv',
                    ['lightning-action', 'predict', '--model', test_model, '--data', test_data]
                ):
                    main()
                    
                mock_predict_handle.assert_called_once()

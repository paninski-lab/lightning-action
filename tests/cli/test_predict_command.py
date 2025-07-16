"""Tests for CLI predict command."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lightning_action.cli.commands.predict import handle


class TestHandle:
    """Test predict command handler."""

    @pytest.fixture
    def mock_args(self):
        """Create mock arguments."""
        args = MagicMock()
        args.model = Path('/test/model')
        args.data = Path('/test/data')
        args.output = None
        args.input_dir = 'markers'
        args.expt_ids = None
        return args

    def test_handle_sets_default_output(self, mock_args):
        """Test that handle sets default output directory."""
        with patch('lightning_action.api.model.Model.from_dir') as mock_from_dir:
            mock_model = MagicMock()
            mock_from_dir.return_value = mock_model
            
            handle(mock_args)
            
            # check that default output was set
            expected_output = mock_args.model / 'predictions'
            assert mock_args.output == expected_output

    def test_handle_uses_custom_output(self, mock_args):
        """Test that handle uses custom output directory when provided."""
        custom_output = Path('/custom/output')
        mock_args.output = custom_output
        
        with patch('lightning_action.api.model.Model.from_dir') as mock_from_dir:
            mock_model = MagicMock()
            mock_from_dir.return_value = mock_model
            
            handle(mock_args)
            
            # check that custom output was preserved
            assert mock_args.output == custom_output

    def test_handle_loads_model(self, mock_args):
        """Test that handle loads model from directory."""
        with patch('lightning_action.api.model.Model.from_dir') as mock_from_dir:
            mock_model = MagicMock()
            mock_from_dir.return_value = mock_model
            
            handle(mock_args)
            
            # check that Model.from_dir was called with correct path
            mock_from_dir.assert_called_once_with(mock_args.model)

    def test_handle_calls_predict(self, mock_args):
        """Test that handle calls model.predict with correct arguments."""
        with patch('lightning_action.api.model.Model.from_dir') as mock_from_dir:
            mock_model = MagicMock()
            mock_from_dir.return_value = mock_model
            
            handle(mock_args)
            
            # check that predict was called with correct arguments
            mock_model.predict.assert_called_once_with(
                data_path=mock_args.data,
                input_dir=mock_args.input_dir,
                output_dir=mock_args.output,
                expt_ids=mock_args.expt_ids,
            )

    def test_handle_with_custom_input_dir(self, mock_args):
        """Test that handle respects custom input directory."""
        mock_args.input_dir = 'features'
        
        with patch('lightning_action.api.model.Model.from_dir') as mock_from_dir:
            mock_model = MagicMock()
            mock_from_dir.return_value = mock_model
            
            handle(mock_args)
            
            # check that predict was called with custom input_dir
            predict_call = mock_model.predict.call_args
            assert predict_call[1]['input_dir'] == 'features'

    def test_handle_with_specific_experiments(self, mock_args):
        """Test that handle respects specific experiment IDs."""
        mock_args.expt_ids = ['exp1', 'exp2']
        
        with patch('lightning_action.api.model.Model.from_dir') as mock_from_dir:
            mock_model = MagicMock()
            mock_from_dir.return_value = mock_model
            
            handle(mock_args)
            
            # check that predict was called with specific experiment IDs
            predict_call = mock_model.predict.call_args
            assert predict_call[1]['expt_ids'] == ['exp1', 'exp2']

    def test_handle_model_loading_error(self, mock_args):
        """Test error handling when model loading fails."""
        with patch('lightning_action.api.model.Model.from_dir') as mock_from_dir:
            mock_from_dir.side_effect = Exception('Model loading failed')
            
            with pytest.raises(Exception, match='Model loading failed'):
                handle(mock_args)

    def test_handle_prediction_error(self, mock_args):
        """Test error handling when prediction fails."""
        with patch('lightning_action.api.model.Model.from_dir') as mock_from_dir:
            mock_model = MagicMock()
            mock_model.predict.side_effect = Exception('Prediction failed')
            mock_from_dir.return_value = mock_model
            
            with pytest.raises(Exception, match='Prediction failed'):
                handle(mock_args)

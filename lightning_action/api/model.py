"""High-level API for lightning-action models (CSV-based pipeline).

This module provides a high-level interface for training, loading, and using
lightning-action models for action segmentation from CSV/tabular data.

The Model class inherits from BaseModelAPI and provides CSV-specific:
- Configuration preprocessing (VelocityConcat transform handling)
- Prediction on CSV marker/feature files
- Post-training inference on tabular data

Example usage:
    # Load a trained model
    model = Model.from_dir('runs/my_experiment')
    
    # Run predictions on new data
    model.predict(
        data_path='/path/to/data',
        input_dir='markers',
        output_dir='/path/to/predictions',
    )
    
    # Or train a new model
    model = Model.from_config('config.yaml')
    model.train(output_dir='runs/new_experiment')
"""

from pathlib import Path
from typing import Any

import lightning as pl
import numpy as np
import pandas as pd
import torch
from typeguard import typechecked

from lightning_action.api.base import BaseModelAPI
from lightning_action.data import DataModule
from lightning_action.models.segmenter import Segmenter
from lightning_action.train import train


@typechecked
class Model(BaseModelAPI[Segmenter]):
    """High-level API wrapper for CSV-based action segmentation models.

    This class manages both the Lightning model and the training/inference
    processes, providing a convenient interface for action segmentation
    tasks using tabular data (markers, features, etc.).

    Inherits from BaseModelAPI and provides CSV-specific implementations
    for model creation, training, and prediction.
    """

    @classmethod
    def _get_model_class(cls) -> type:
        """Return the Segmenter model class."""
        return Segmenter

    @classmethod
    def _get_train_function(cls):
        """Return the CSV pipeline training function."""
        return train

    @classmethod
    def _create_model_from_config(cls, config: dict[str, Any]) -> Segmenter:
        """Create a Segmenter model from configuration.

        Handles CSV-specific config preprocessing:
        - Doubles input_size if VelocityConcat transform is used

        Args:
            config: Configuration dictionary.

        Returns:
            Initialized Segmenter model.
        """
        # Handle VelocityConcat transform (doubles input size)
        if config['data'].get('transforms'):
            if 'VelocityConcat' in config['data']['transforms']:
                config['model']['input_size'] *= 2

        return Segmenter(config)

    def _setup_trainer(self) -> pl.Trainer:
        """Set up a Lightning Trainer for prediction.

        Configures trainer based on the device setting in the training config.
        Supports both CPU and GPU prediction.

        Returns:
            Configured Trainer instance for prediction.
        """
        training_config = self.config.get('training', {})
        device = training_config.get('device', 'cpu')

        trainer_config = {
            'accelerator': 'gpu' if device == 'gpu' and torch.cuda.is_available() else 'cpu',
            'devices': 1,
            'logger': False,
            'enable_checkpointing': False,
            'enable_progress_bar': False,
        }

        return pl.Trainer(**trainer_config)

    def _run_post_training_inference(self) -> None:
        """Run inference on all training experiment IDs after training.

        Extracts experiment IDs from the training configuration and runs
        predictions, saving results to model_dir/predictions/.
        """
        if self.model is None:
            print('Warning: No trained model found, skipping post-training inference')
            return

        if self.model_dir is None:
            print('Warning: No model directory found, skipping post-training inference')
            return

        # Extract data configuration
        data_config = self.config.get('data', {})

        # Check for simplified data_path format
        if 'data_path' in data_config:
            data_path = data_config['data_path']
            input_dir = data_config.get('input_dir', 'markers')
            expt_ids = data_config.get('expt_ids', None)
        else:
            # Full format configuration - can't determine data_path automatically
            if 'ids' in data_config:
                print('Warning: Full data config format detected. Cannot determine '
                      'data_path for automatic inference.')
                print('Skipping post-training inference. Use predict() method manually '
                      'if needed.')
                return
            else:
                print('Warning: No data path found in configuration, skipping '
                      'post-training inference')
                return

        # Create predictions directory
        predictions_dir = self.model_dir / 'predictions'
        predictions_dir.mkdir(exist_ok=True)

        print(f'Running post-training inference on all training experiments...')
        print(f'Data path: {data_path}')
        print(f'Input directory: {input_dir}')
        print(f'Experiment IDs: {expt_ids if expt_ids else "all"}')
        print(f'Predictions will be saved to: {predictions_dir}')

        try:
            self.predict(
                data_path=data_path,
                input_dir=input_dir,
                output_dir=predictions_dir,
                expt_ids=expt_ids,
            )
            print('Post-training inference completed successfully!')
        except Exception as e:
            print(f'Warning: Post-training inference failed with error: {e}')
            print('Training completed successfully, but automatic inference was skipped.')

    @typechecked
    def predict(
        self,
        data_path: str | Path,
        input_dir: str,
        output_dir: str | Path,
        output_file: str | Path | None = None,
        expt_ids: list[str] | None = None,
    ) -> None:
        """Generate predictions for data using the trained model.

        Creates separate prediction files for each experiment in the output
        directory.

        Args:
            data_path: Path to data directory containing signal directories.
            input_dir: Name of input directory ('markers', 'features', etc.).
            output_dir: Directory to save prediction files (one per experiment).
            output_file: Full path to save prediction file; overwrites output_dir
                if not None. Only valid with a single experiment.
            expt_ids: List of experiment IDs to predict on (None for all).

        Raises:
            ValueError: If model is not trained.
            RuntimeError: If output_file specified with multiple experiments.
        """
        data_path = Path(data_path)
        output_dir = Path(output_dir)

        if self.model is None:
            raise ValueError('Model must be trained or loaded before prediction')

        if output_file is not None and (expt_ids is not None and len(expt_ids) > 1):
            raise RuntimeError(
                'Can only supply `output_file` when specifying a single expt_id'
            )

        # Build data configuration
        from lightning_action.train import build_data_config_from_path

        data_config = build_data_config_from_path(
            data_path=data_path,
            expt_ids=expt_ids,
            signal_types=[input_dir],
        )

        training_config = self.config.get('training', {})
        experiment_ids = data_config['ids']

        trainer = self._setup_trainer()

        for expt_id in experiment_ids:
            print(f'Generating predictions for experiment: {expt_id}')

            # Create data config for single experiment
            single_expt_config = build_data_config_from_path(
                data_path=data_path,
                expt_ids=[expt_id],
                signal_types=[input_dir],
                transforms=self.config['data'].get('transforms', None),
            )

            # Create datamodule
            datamodule = DataModule(
                data_config=single_expt_config,
                sequence_length=training_config.get('sequence_length', 500),
                sequence_pad=self.model.sequence_pad,
                batch_size=training_config.get('batch_size', 32),
                num_workers=training_config.get('num_workers', 4),
                train_probability=1.0,  # Use all data for prediction
                val_probability=0.0,
                seed=training_config.get('seed', 42),
            )

            datamodule.setup('predict')

            # Generate predictions
            predictions = trainer.predict(self.model, datamodule=datamodule)

            # Concatenate predictions from all batches
            all_probs = []
            for batch_preds in predictions:
                probs = batch_preds['probabilities'][0]  # Remove batch dim
                all_probs.append(probs.cpu().numpy())

            final_probs = np.vstack(all_probs)

            # Pad with NaNs if needed to match original data length
            original_length = datamodule.dataset.data_lengths[0]
            current_length = final_probs.shape[0]

            if current_length < original_length:
                num_classes = final_probs.shape[1]
                padding_rows = original_length - current_length
                nan_padding = np.full((padding_rows, num_classes), np.nan)
                final_probs = np.vstack([final_probs, nan_padding])
                print(f'Padded predictions from {current_length} to {original_length} rows')

            # Save predictions
            df = pd.DataFrame(
                data=final_probs,
                columns=self.model.config['data']['label_names']
            )

            if output_file is not None:
                output_file_ = Path(output_file)
            else:
                output_file_ = output_dir / f'{expt_id}_predictions.csv'

            output_file_.parent.mkdir(exist_ok=True, parents=True)
            df.to_csv(output_file_)
            print(f'Saved predictions to {output_file_}')

        print(f'Completed predictions for {len(experiment_ids)} experiments in {output_dir}')

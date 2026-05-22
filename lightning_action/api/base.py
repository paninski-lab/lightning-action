"""Base class for high-level model APIs.

This module provides a base class that contains shared functionality between
the CSV-based Model API and the video-based VideoModel API. It handles:

- Configuration loading and validation
- Checkpoint discovery and loading
- Directory management utilities
- Common training/inference patterns

Subclasses must implement:
- _create_model_from_config(): Create the underlying model from config
- _get_model_class(): Return the model class (Segmenter or VideoSegmenter)
- _get_train_function(): Return the training function
- _setup_trainer(): Set up a Lightning Trainer for prediction
- _run_post_training_inference(): Pipeline-specific post-training inference
- predict(): Pipeline-specific prediction logic
"""

import contextlib
import os
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any, Generic, TypeVar

import lightning as pl
import torch

from lightning_action.config import load_config

# Type variable for the model type
ModelT = TypeVar('ModelT', bound=pl.LightningModule)


@contextlib.contextmanager
def chdir(path: Path) -> Generator[None, None, None]:
    """Context manager for temporarily changing working directory.

    Useful when training requires relative paths from the output directory.

    Args:
        path: Directory to change to.

    Yields:
        None. Working directory is restored on exit.

    Example:
        with chdir(Path('/some/dir')):
            # Working directory is /some/dir here
            do_something()
        # Working directory is restored here
    """
    old_cwd = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(old_cwd)


class BaseModelAPI(ABC, Generic[ModelT]):
    """Abstract base class for high-level model API wrappers.

    This class provides shared functionality for model APIs, including:
    - Loading models from directories (config + checkpoint)
    - Creating models from configuration files
    - Training with optional post-training inference
    - Common directory and file management utilities

    Subclasses must implement abstract methods to provide pipeline-specific
    behavior for the CSV-based and video-based workflows.

    Attributes:
        model: The underlying Lightning module.
        config: Configuration dictionary used to create/load the model.
        model_dir: Directory where the model is stored (after training/loading).
    """

    def __init__(
        self,
        model: ModelT,
        config: dict[str, Any],
        model_dir: str | Path | None = None,
    ) -> None:
        """Initialize the model wrapper.

        This constructor is typically not called directly. Use the
        class methods `from_dir()` or `from_config()` instead.

        Args:
            model: Initialized Lightning model.
            config: Configuration dictionary.
            model_dir: Optional path to model directory.
        """
        self.model = model
        self.config = config
        self.model_dir = Path(model_dir) if model_dir is not None else None

    @classmethod
    @abstractmethod
    def _create_model_from_config(cls, config: dict[str, Any]) -> ModelT:
        """Create a model instance from configuration.

        This method should handle any config preprocessing (e.g., computing
        sequence_pad, adjusting input_size for transforms) before creating
        the model.

        Args:
            config: Configuration dictionary.

        Returns:
            Initialized model (untrained).
        """
        pass  # pragma: no cover

    @classmethod
    @abstractmethod
    def _get_model_class(cls) -> type:
        """Return the model class to use for loading checkpoints.

        Returns:
            The Lightning module class (e.g., Segmenter, VideoSegmenter).
        """
        pass  # pragma: no cover

    @classmethod
    @abstractmethod
    def _get_train_function(cls) -> Callable[..., Any]:
        """Return the training function for this pipeline.

        Returns:
            The train function (e.g., train, train_video).
        """
        pass  # pragma: no cover

    @abstractmethod
    def _setup_trainer(self) -> pl.Trainer:
        """Set up a Lightning Trainer for prediction.

        Subclasses implement this with pipeline-specific configuration
        (e.g., CPU vs GPU, precision settings, progress bar options).

        Returns:
            Configured Trainer instance for prediction.
        """
        pass  # pragma: no cover

    @abstractmethod
    def _run_post_training_inference(self) -> None:
        """Run inference on training data after training completes.

        This method should extract experiment/video IDs from the training
        configuration and run predictions, saving results to the model
        directory.
        """
        pass  # pragma: no cover

    @abstractmethod
    def predict(self, *args: Any, **kwargs: Any) -> None:
        """Generate predictions using the trained model.

        Implementation differs between CSV and video pipelines.
        """
        pass  # pragma: no cover

    @classmethod
    def _find_config_file(cls, model_dir: Path) -> Path:
        """Find the configuration file in a model directory.

        Searches for config.yaml first, then falls back to hparams.yaml
        for compatibility with older models.

        Args:
            model_dir: Path to the model directory.

        Returns:
            Path to the configuration file.

        Raises:
            FileNotFoundError: If no config file is found.
        """
        config_path = model_dir / 'config.yaml'
        if not config_path.exists():
            config_path = model_dir / 'hparams.yaml'

        if not config_path.exists():
            raise FileNotFoundError(f'Config file not found in {model_dir}')

        return config_path

    @classmethod
    def _find_checkpoint_file(cls, model_dir: Path) -> Path:
        """Find a checkpoint file in a model directory.

        Searches for checkpoint files with these priorities:
        1. *best*.ckpt (best checkpoint from training)
        2. *.ckpt (any Lightning checkpoint)
        3. *best*.pt (best PyTorch state dict)
        4. *.pt (any PyTorch state dict)

        Args:
            model_dir: Path to the model directory.

        Returns:
            Path to the checkpoint file.

        Raises:
            FileNotFoundError: If no checkpoint file is found.
        """
        checkpoint_patterns = ['*best*.ckpt', '*.ckpt', '*best*.pt', '*.pt']

        for pattern in checkpoint_patterns:
            checkpoints = list(model_dir.rglob(pattern))
            if checkpoints:
                return checkpoints[0]

        raise FileNotFoundError(f'No checkpoint files found in {model_dir}')

    @classmethod
    def _load_checkpoint(
        cls,
        model: ModelT,
        checkpoint_path: Path,
        config: dict[str, Any],
    ) -> ModelT:
        """Load model weights from a checkpoint file.

        Handles both Lightning checkpoint format (.ckpt) and plain
        PyTorch state dicts (.pt).

        Args:
            model: Model instance to load weights into (used for .pt files).
            checkpoint_path: Path to the checkpoint file.
            config: Configuration dictionary (passed to load_from_checkpoint).

        Returns:
            Model with loaded weights.
        """
        model_class = cls._get_model_class()

        if checkpoint_path.suffix == '.ckpt':
            # Lightning checkpoint format - load directly
            model = model_class.load_from_checkpoint(checkpoint_path, config=config)
        else:
            # Plain PyTorch state dict
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(state_dict)

        model.eval()
        print(f'Loaded model weights from {checkpoint_path}')

        return model

    @classmethod
    def from_dir(cls, model_dir: str | Path) -> 'BaseModelAPI[ModelT]':
        """Load a trained model from a directory.

        Searches for configuration and checkpoint files in the directory
        and loads the model with trained weights.

        Expected directory structure:
            model_dir/
                config.yaml (or hparams.yaml)
                checkpoints/
                    best-*.ckpt (or any .ckpt/.pt file)

        Args:
            model_dir: Path to directory containing model files.

        Returns:
            Model API instance with loaded weights.

        Raises:
            FileNotFoundError: If config or checkpoint files are not found.
        """
        model_dir = Path(model_dir)

        # Load configuration
        config_path = cls._find_config_file(model_dir)
        config = load_config(config_path)

        # Create model architecture
        model = cls._create_model_from_config(config)

        # Find and load checkpoint
        checkpoint_path = cls._find_checkpoint_file(model_dir)
        model = cls._load_checkpoint(model, checkpoint_path, config)

        return cls(model, config, model_dir)

    @classmethod
    def from_config(cls, config_path: str | Path | dict) -> 'BaseModelAPI[ModelT]':
        """Create a new untrained model from configuration.

        Args:
            config_path: Path to YAML config file, or dict with config.

        Returns:
            Model API instance with randomly initialized weights.

        Raises:
            FileNotFoundError: If config file path doesn't exist.
        """
        if not isinstance(config_path, dict):
            config = load_config(config_path)
        else:
            config = config_path

        model = cls._create_model_from_config(config)

        return cls(model, config, model_dir=None)

    def train(
        self,
        output_dir: str | Path = 'runs/default',
        post_inference: bool = True,
    ) -> None:
        """Train the model using PyTorch Lightning.

        After training completes, optionally runs inference on all training
        data and saves predictions to output_dir/predictions/.

        Args:
            output_dir: Directory to save checkpoints, logs, and predictions.
            post_inference: If True, run inference on training data after
                training completes.
        """
        self.model_dir = Path(output_dir)
        self.model_dir.mkdir(exist_ok=True, parents=True)

        train_fn = self._get_train_function()

        with chdir(self.model_dir):
            self.model = train_fn(self.config, self.model, output_dir=self.model_dir)

        if post_inference:
            self._run_post_training_inference()

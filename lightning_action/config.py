"""Pydantic models for lightning-action configuration validation.

Defines the expected schema for YAML config files and provides load_config(),
which reads a YAML file, validates it, and returns a plain dict that the rest
of the codebase consumes unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, field_validator

HeadType = Literal[
    'temporal-mlp', 'temporalmlp',
    'tcn',
    'dtcn', 'dilatedtcn',
    'lstm', 'gru', 'rnn',
]


class DataConfig(BaseModel):
    """Pydantic model for the data section of the config."""

    model_config = ConfigDict(extra='allow')


class ModelConfig(BaseModel):
    """Pydantic model for the model section of the config."""

    model_config = ConfigDict(extra='allow')
    head: HeadType
    output_size: int
    num_hid_units: int = 64
    num_layers: int = 2
    num_lags: int = 2


class OptimizerConfig(BaseModel):
    """Pydantic model for the optimizer section of the config."""

    model_config = ConfigDict(extra='allow')
    lr: float
    type: str = 'Adam'

    @field_validator('type')
    @classmethod
    def validate_type(cls, v: str) -> str:
        """Validate that the optimizer type is supported."""
        valid = {'adam', 'adamw', 'sgd'}
        if v.lower() not in valid:
            raise ValueError(
                f'Invalid optimizer type: {v!r}. '
                f'Valid values: {", ".join(sorted(valid))}'
            )
        return v


class TrainingConfig(BaseModel):
    """Pydantic model for the training section of the config."""

    model_config = ConfigDict(extra='allow')
    num_epochs: int
    device: str = 'cpu'
    sequence_length: int = 500
    batch_size: int = 32
    num_workers: int = 4
    train_probability: float = 0.9
    val_probability: float = 0.1

    @field_validator('device')
    @classmethod
    def validate_device(cls, v: str) -> str:
        """Validate that the compute device is supported."""
        valid = {'cpu', 'gpu'}
        if v.lower() not in valid:
            raise ValueError(
                f'Invalid device: {v!r}. '
                f'Valid values: {", ".join(sorted(valid))}'
            )
        return v


class LightningActionConfig(BaseModel):
    """Top-level pydantic model for lightning-action configuration."""

    data: DataConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    training: TrainingConfig


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a YAML configuration file.

    Reads a YAML config, validates it against LightningActionConfig, and
    returns the result as a plain dict so the rest of the codebase is
    unchanged.

    Args:
        path: Path to a YAML configuration file.

    Returns:
        Validated configuration as a plain dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        pydantic.ValidationError: If the config fails schema validation.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Config file not found: {path}')
    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f)
    validated = LightningActionConfig.model_validate(raw)
    return validated.model_dump()

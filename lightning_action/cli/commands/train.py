"""Command to train a model."""

import datetime
import logging
from pathlib import Path
from typing import Any

import yaml
from typeguard import typechecked

from lightning_action.api.model import Model
from lightning_action.cli.types import config_file, output_dir

logger = logging.getLogger('LIGHTNING_ACTION.CLI.TRAIN')


def register_parser(subparsers):
    """Register the train command parser."""
    parser = subparsers.add_parser(
        'train',
        description='Train a Lightning action segmentation model.',
        usage='lightning-action train --config <config_path> [options]',
    )

    # Required arguments
    required = parser.add_argument_group('required arguments')
    required.add_argument(
        '--config', '-c',
        type=config_file,
        required=True,
        help='Path to model configuration file (YAML)',
    )

    # Optional arguments
    optional = parser.add_argument_group('options')
    optional.add_argument(
        '--output-dir', '-o',
        type=output_dir,
        help='Directory to save model outputs (default: ./runs/YYYY-MM-DD/HH-MM-SS)',
    )
    optional.add_argument(
        '--data-dir',
        type=Path,
        help='Override data path specified in config',
    )
    optional.add_argument(
        '--device',
        choices=['cpu', 'gpu'],
        help='Device to use for training (overrides config)',
    )
    optional.add_argument(
        '--epochs',
        type=int,
        help='Number of training epochs (overrides config)',
    )
    optional.add_argument(
        '--batch-size',
        type=int,
        help='Batch size for training (overrides config)',
    )
    optional.add_argument(
        '--lr',
        type=float,
        help='Learning rate (overrides config)',
    )
    optional.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducibility (overrides config)',
    )
    optional.add_argument(
        '--overrides',
        nargs='*',
        metavar='KEY=VALUE',
        help='Override specific config values (format: key.subkey=value)',
    )


@typechecked
def handle(args):
    """Handle the train command execution."""
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Determine output directory
    if not args.output_dir:
        now = datetime.datetime.now()
        args.output_dir = Path('runs') / now.strftime('%Y-%m-%d') / now.strftime('%H-%M-%S')
        args.output_dir.mkdir()

    # Set up logging to the model directory
    _setup_model_logging(args.output_dir)

    # Apply command line overrides
    config = apply_overrides(config, args)

    logger.info(f'Output directory: {args.output_dir}')
    logger.info(f'Config: {config}')

    # Create model from config
    model = Model.from_config(config)

    # Train model
    try:
        model.train(output_dir=args.output_dir)
        logger.info('Training completed successfully')
    except Exception as e:
        logger.error(f'Training failed: {e}', exc_info=True)
        raise


def _setup_model_logging(output_dir: Path):
    """Set up additional logging to the model directory and remove original file handler."""

    # Create log file path
    log_file = output_dir / 'training.log'

    # Get the root logger
    root_logger = logging.getLogger()

    # Create a new file handler for the model directory
    model_handler = logging.FileHandler(log_file)
    model_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s %(levelname)s  %(name)s : %(message)s'
    )
    model_handler.setFormatter(formatter)

    # Add the new handler to the root logger
    root_logger.addHandler(model_handler)

    return model_handler


@typechecked
def apply_overrides(config: dict[str, Any], args) -> dict[str, Any]:
    """Apply command line overrides to config."""
    # Apply direct overrides
    if args.data_dir:
        config['data']['data_path'] = str(args.data_dir)
    
    if args.device:
        config.setdefault('training', {})['device'] = args.device
    
    if args.epochs:
        config.setdefault('training', {})['num_epochs'] = args.epochs
    
    if args.batch_size:
        config.setdefault('training', {})['batch_size'] = args.batch_size
    
    if args.lr:
        config.setdefault('optimizer', {})['lr'] = args.lr
    
    if args.seed:
        config.setdefault('training', {})['seed'] = args.seed

    # Apply custom overrides
    if args.overrides:
        config = apply_config_overrides(config, args.overrides)

    return config


@typechecked
def apply_config_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply command line overrides to config using dot notation."""
    for override in overrides:
        if '=' not in override:
            raise ValueError(f"Override must be in format 'key=value', got: {override}")

        key_path, value = override.split('=', 1)
        keys = key_path.split('.')

        # Parse value
        parsed_value = parse_config_value(value)

        # Navigate to the correct nested location
        current = config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        # Set the final value
        current[keys[-1]] = parsed_value

    return config


@typechecked
def parse_config_value(value: str) -> Any:
    """Parse a config value from string to appropriate type."""
    # Handle booleans
    if value.lower() == 'true':
        return True
    elif value.lower() == 'false':
        return False
    elif value.lower() == 'null' or value.lower() == 'none':
        return None
    
    # Handle numbers
    if value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
        return int(value)
    
    try:
        return float(value)
    except ValueError:
        pass
    
    # Handle lists (comma-separated)
    if ',' in value:
        return [parse_config_value(v.strip()) for v in value.split(',')]
    
    # Return as string
    return value

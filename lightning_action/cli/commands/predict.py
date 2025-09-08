"""Command to run model prediction on data."""

import logging
from pathlib import Path

from typeguard import typechecked

from lightning_action.api.model import Model
from lightning_action.cli.types import valid_dir

logger = logging.getLogger('LIGHTNING_ACTION.CLI.PREDICT')


def register_parser(subparsers):
    """Register the predict command parser."""
    parser = subparsers.add_parser(
        'predict',
        description='Run inference using a trained Lightning action segmentation model.',
        usage='lightning-action predict --model <model_dir> --data <data_path> [options]',
    )

    # Required arguments
    required = parser.add_argument_group('required arguments')
    required.add_argument(
        '--model', '-m',
        type=valid_dir,
        required=True,
        help='Directory containing trained model',
    )
    required.add_argument(
        '--data-dir', '-d',
        type=valid_dir,
        required=True,
        help='Path to data directory with signal directories (markers, labels, etc.)',
    )

    # Optional arguments
    optional = parser.add_argument_group('options')
    optional.add_argument(
        '--output-dir', '-o',
        type=Path,
        help='Directory to save prediction results (default: <model_dir>/predictions)',
    )
    optional.add_argument(
        '--input-dir',
        type=str,
        default='markers',
        help='Input signal directory name (default: markers)',
    )
    optional.add_argument(
        '--expt-ids',
        nargs='*',
        help='Specific experiment IDs to predict (default: all experiments)',
    )


@typechecked
def handle(args):
    """Handle the predict command execution."""
    # Set default output directory
    if not args.output_dir:
        args.output_dir = args.model / 'predictions'

    logger.info(f'Loading model from: {args.model}')
    logger.info(f'Data directory: {args.data_dir}')
    logger.info(f'Input signal type: {args.input_dir}')
    logger.info(f'Output directory: {args.output_dir}')

    if args.expt_ids:
        logger.info(f'Predicting on experiments: {args.expt_ids}')
    else:
        logger.info('Predicting on all available experiments')

    # Load model
    try:
        model = Model.from_dir(args.model)
        logger.info('Model loaded successfully')
    except Exception as e:
        logger.error(f'Failed to load model: {e}')
        raise

    # Run prediction
    try:
        model.predict(
            data_path=args.data_dir,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            expt_ids=args.expt_ids,
        )
        logger.info('Predictions completed successfully')
    except Exception as e:
        logger.error(f'Prediction failed: {e}', exc_info=True)
        raise

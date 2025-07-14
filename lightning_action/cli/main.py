"""Command-line interface for lightning-action action segmentation package."""

import logging
import sys
from argparse import ArgumentParser

from lightning_action.cli import formatting
from lightning_action.cli.commands import COMMANDS


def build_parser() -> ArgumentParser:
    """Build the main argument parser with all subcommands."""

    parser = formatting.ArgumentParser(
        prog='lightning-action',
        description='Lightning-based action segmentation for behavioral analysis.',
    )

    subparsers = parser.add_subparsers(
        dest='command',
        required=True,
        help='Command to run',
        parser_class=formatting.SubArgumentParser,
    )

    # register all commands from the commands module
    for name, module in COMMANDS.items():
        module.register_parser(subparsers)

    return parser


def main():
    """Main CLI entry point."""

    # configure logging once at application startup
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s  %(name)s : %(message)s',
        handlers=[
            logging.StreamHandler(),  # Console output
        ]
    )

    parser = build_parser()

    # if no commands provided, display help
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    # parse arguments
    args = parser.parse_args()

    # get command handler
    command_handler = COMMANDS[args.command].handle

    # execute command
    command_handler(args)


if __name__ == '__main__':
    main()

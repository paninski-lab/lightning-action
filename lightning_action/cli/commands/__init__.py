"""Command modules for the lightning-action CLI."""

from lightning_action.cli.commands import predict, train

# dictionary of all available commands
COMMANDS = {
    'train': train,      # model training
    'predict': predict,  # model inference on keypoints
}

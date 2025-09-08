"""Lightning Action: Modern action segmentation framework built with PyTorch Lightning."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("lightning-action")
except PackageNotFoundError:
    # package is not installed, set a default or read from pyproject.toml
    __version__ = "unknown"

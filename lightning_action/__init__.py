"""Lightning Action: Modern action segmentation framework built with PyTorch Lightning."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("lightning-action")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pytest

from lightning_action.api.model import Model

ROOT = Path(__file__).parent.parent


@pytest.fixture
def create_test_marker_csv():

    def _create_test_marker_csv(file_path: Path, n_frames: int = 100, n_markers: int = 3):
        """Create a test marker CSV file in DLC format."""
        # create multi-level headers
        markers = [f'marker_{i}' for i in range(n_markers)]
        coords = ['x', 'y', 'likelihood']

        # create column structure
        columns = pd.MultiIndex.from_product([['scorer'], markers, coords])

        # create random data
        data = np.random.rand(n_frames, len(markers) * len(coords))

        # create dataframe
        df = pd.DataFrame(data, columns=columns)
        df.index.name = 'frame'

        # save to csv
        df.to_csv(file_path)
        return markers

    return _create_test_marker_csv


@pytest.fixture
def create_test_feature_csv() -> Callable:

    def _create_test_feature_csv(file_path: Path, n_frames: int = 100, n_features: int = 6):
        """Create a test feature CSV file."""
        feature_names = [f'feature_{i}' for i in range(n_features)]
        data = np.random.rand(n_frames, n_features)

        df = pd.DataFrame(data, columns=feature_names)
        df.index.name = 'frame'

        df.to_csv(file_path)
        return feature_names

    return _create_test_feature_csv


@pytest.fixture
def create_test_label_csv():

    def _create_test_label_csv(file_path: Path, n_frames: int = 100, n_classes: int = 4):
        """Create a test label CSV file with one-hot encoding."""
        class_names = [f'class_{i}' for i in range(n_classes)]

        # create one-hot encoded labels
        labels = np.zeros((n_frames, n_classes), dtype=int)
        for i in range(n_frames):
            labels[i, np.random.randint(0, n_classes)] = 1

        df = pd.DataFrame(labels, columns=class_names)
        df.index.name = 'frame'

        df.to_csv(file_path)
        return class_names

    return _create_test_label_csv


@pytest.fixture
def data_dir() -> Path:
    return ROOT.joinpath('data')


@pytest.fixture
def config_path() -> Path:
    return ROOT.joinpath('data', 'fly.yaml')

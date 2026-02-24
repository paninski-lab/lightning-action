import gc
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pytest
import torch

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


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU (deselect with '-m not gpu')"
    )
    config.addinivalue_line(
        "markers", "dali: mark test as requiring NVIDIA DALI"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically add 'gpu' marker to tests in TestDALIIntegration class."""
    for item in items:
        # Check if test has @requires_gpu decorator or is in GPU test class
        if 'TestDALIIntegration' in item.nodeid:
            item.add_marker(pytest.mark.gpu)

        # Skip GPU tests if no GPU available
        if item.get_closest_marker('gpu'):
            if not torch.cuda.is_available():
                item.add_marker(pytest.mark.skip(reason="GPU not available"))


def pytest_runtest_setup(item):
    """Set a default timeout for GPU tests to prevent infinite hangs."""
    if "TestDALIIntegration" in item.nodeid:
        # Add a 120 second timeout for GPU tests
        # This requires pytest-timeout: pip install pytest-timeout
        try:
            timeout_marker = item.get_closest_marker('timeout')
            if timeout_marker is None:
                item.add_marker(pytest.mark.timeout(120))
        except Exception:
            pass  # pytest-timeout not installed


@pytest.fixture
def cleanup_gpu():
    """Fixture to clean up GPU resources before and after each test.

    This is critical for DALI tests to prevent resource leaks and stalls.
    DALI maintains internal state that can cause deadlocks if not properly
    released between tests.
    """
    # Cleanup before test
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    yield

    # Cleanup after test - more aggressive
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Force CUDA context sync
        try:
            torch.cuda.current_stream().synchronize()
        except Exception:
            pass


@pytest.fixture(scope="class")
def cleanup_gpu_class():
    """Class-scoped GPU cleanup fixture.

    Use this for test classes where you want cleanup before/after
    the entire class runs, not between individual tests.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    yield

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


@pytest.fixture
def gpu_device():
    """Fixture that provides GPU device if available, skips test otherwise."""
    if not torch.cuda.is_available():
        pytest.skip("GPU not available")
    return torch.device('cuda:0')


@pytest.fixture
def cpu_device():
    """Fixture that provides CPU device."""
    return torch.device('cpu')

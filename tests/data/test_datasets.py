"""Tests for dataset classes."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from lightning_action.data import FeatureDataset


class TestFeatureDataset:
    """Test the FeatureDataset class."""

    def test_init_validation(self):
        """Test initialization parameter validation."""
        # mismatched lengths should raise error
        with pytest.raises(ValueError, match='must have same length'):
            FeatureDataset(
                ids=['dataset1'],
                signals=[['markers'], ['labels']],  # different length
                transforms=[[None], [None]],
                paths=[[None]]
            )

        # mismatched signal/transform/path lengths should raise error
        with pytest.raises(
            ValueError,
            match='signals, transforms, and paths must have same length'
        ):
            FeatureDataset(
                ids=['dataset1'],
                signals=[['markers', 'labels']],
                transforms=[[None]],  # only one transform for two signals
                paths=[['path1', 'path2']]
            )

    def test_markers_only_dataset(self, create_test_marker_csv):
        """Test dataset with only marker data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # create test files
            marker_file = Path(tmpdir) / 'markers.csv'
            create_test_marker_csv(marker_file, n_frames=50, n_markers=2)
            
            # create dataset
            dataset = FeatureDataset(
                ids=['test_dataset'],
                signals=[['markers']],
                transforms=[[None]],
                paths=[[str(marker_file)]],
                sequence_length=10,
                sequence_pad=2,
            )
            
            # check dataset properties
            assert len(dataset) > 0
            assert dataset.input_size == 4  # 2 markers * 2 coords (x,y)
            assert len(dataset.get_feature_names()) == 4
            assert len(dataset.get_label_names()) == 0
            
            # check sequence
            sequence = dataset[0]
            assert 'markers' in sequence
            assert sequence['markers'].shape[1] == 4  # 2 markers * 2 coords

    def test_markers_and_labels_dataset(self, create_test_marker_csv, create_test_label_csv):
        """Test dataset with both markers and labels."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # create test files
            marker_file = Path(tmpdir) / 'markers.csv'
            label_file = Path(tmpdir) / 'labels.csv'
            
            n_frames = 50
            create_test_marker_csv(marker_file, n_frames=n_frames, n_markers=3)
            create_test_label_csv(label_file, n_frames=n_frames, n_classes=5)
            
            # create dataset
            dataset = FeatureDataset(
                ids=['test_dataset'],
                signals=[['markers', 'labels']],
                transforms=[[None, None]],
                paths=[[str(marker_file), str(label_file)]],
                sequence_length=10,
            )
            
            # check dataset properties
            assert len(dataset) > 0
            assert dataset.input_size == 6  # 3 markers * 2 coords
            assert len(dataset.get_feature_names()) == 6
            assert len(dataset.get_label_names()) == 5
            
            # check sequence
            sequence = dataset[0]
            assert 'markers' in sequence
            assert 'labels' in sequence
            assert sequence['markers'].shape[1] == 6
            assert sequence['labels'].shape[1] == 5

    def test_feature_input_type(self, create_test_feature_csv):
        """Test dataset with feature input type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # create test files
            feature_file = Path(tmpdir) / 'features.csv'
            create_test_feature_csv(feature_file, n_frames=40, n_features=8)
            
            # create dataset
            dataset = FeatureDataset(
                ids=['test_dataset'],
                signals=[['features']],
                transforms=[[None]],
                paths=[[str(feature_file)]],
                sequence_length=5
            )
            
            # check dataset properties
            assert len(dataset) > 0
            assert dataset.input_size == 8
            assert len(dataset.get_feature_names()) == 8

    def test_multiple_datasets(self, create_test_marker_csv):
        """Test with multiple datasets."""
        from lightning_action.data.transforms import ZScore, MotionEnergy
        with tempfile.TemporaryDirectory() as tmpdir:
            # create test files for two datasets
            marker_file1 = Path(tmpdir) / 'markers1.csv'
            marker_file2 = Path(tmpdir) / 'markers2.csv'
            
            create_test_marker_csv(marker_file1, n_frames=30, n_markers=2)
            create_test_marker_csv(marker_file2, n_frames=40, n_markers=2)
            
            # create dataset
            dataset = FeatureDataset(
                ids=['dataset1', 'dataset2'],
                signals=[['markers'], ['markers']],
                transforms=[[None], [[ZScore(), MotionEnergy()]]],
                paths=[[str(marker_file1)], [str(marker_file2)]],
                sequence_length=10
            )

            # check that sequences from both datasets are included
            dataset_ids = [dataset.get_sequence_info(i)['dataset_id'] for i in range(len(dataset))]
            assert 'dataset1' in dataset_ids
            assert 'dataset2' in dataset_ids
            assert len(set(dataset_ids)) == 2

    def test_tensor_output(self, create_test_marker_csv):
        """Test tensor vs numpy output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # create test file
            marker_file = Path(tmpdir) / 'markers.csv'
            create_test_marker_csv(marker_file, n_frames=20, n_markers=1)
            
            # create dataset
            dataset = FeatureDataset(
                ids=['test_dataset'],
                signals=[['markers']],
                transforms=[[None]],
                paths=[[str(marker_file)]],
                sequence_length=5,
            )
            
            # check tensor output (default)
            sequence_tensor = dataset[0]
            assert isinstance(sequence_tensor['markers'], torch.Tensor)
            
            # check numpy output
            sequence_numpy = dataset.__getitem__(idx=0, as_numpy=True)
            assert isinstance(sequence_numpy['markers'], np.ndarray)

    def test_sequence_metadata(self, create_test_marker_csv):
        """Test sequence metadata functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # create test file
            marker_file = Path(tmpdir) / 'markers.csv'
            create_test_marker_csv(marker_file, n_frames=25, n_markers=1)
            
            # create dataset
            dataset = FeatureDataset(
                ids=['test_dataset'],
                signals=[['markers']],
                transforms=[[None]],
                paths=[[str(marker_file)]],
                sequence_length=5
            )
            
            # check metadata
            info = dataset.get_sequence_info(0)
            assert 'dataset_id' in info
            assert 'dataset_idx' in info
            assert 'sequence_idx' in info
            assert info['dataset_id'] == 'test_dataset'

    def test_index_out_of_range(self, create_test_marker_csv):
        """Test index out of range errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # create test file
            marker_file = Path(tmpdir) / 'markers.csv'
            create_test_marker_csv(marker_file, n_frames=15, n_markers=1)
            
            # create dataset
            dataset = FeatureDataset(
                ids=['test_dataset'],
                signals=[['markers']],
                transforms=[[None]],
                paths=[[str(marker_file)]],
                sequence_length=5
            )
            
            # test out of range access
            with pytest.raises(IndexError):
                dataset[len(dataset)]
            
            with pytest.raises(IndexError):
                dataset.get_sequence_info(len(dataset))

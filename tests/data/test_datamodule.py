"""Tests for Lightning DataModule."""

import tempfile
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from lightning_action.data import DataModule


class TestDataModule:
    """Test the DataModule class."""

    def test_init_validation(self):
        """Test initialization parameter validation."""
        # missing required keys should raise error
        with pytest.raises(ValueError, match='data_config must contain keys'):
            DataModule(
                data_config={'ids': ['test']},  # missing signals, transforms, paths
            )

    def test_basic_datamodule_creation(self, create_test_marker_csv, create_test_label_csv):
        """Test basic DataModule creation and setup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # create test files
            marker_file = Path(tmpdir) / 'markers.csv'
            label_file = Path(tmpdir) / 'labels.csv'
            
            create_test_marker_csv(marker_file, n_frames=50, n_markers=2)
            create_test_label_csv(label_file, n_frames=50, n_classes=3)
            
            # create data config
            data_config = {
                'ids': ['test_dataset'],
                'signals': [['markers', 'labels']],
                'transforms': [[None, None]],
                'paths': [[str(marker_file), str(label_file)]],
            }
            
            # create datamodule
            datamodule = DataModule(
                data_config=data_config,
                sequence_length=10,
                batch_size=4,
                train_probability=0.8,
                val_probability=0.2,
            )
            
            # check that dataset was created
            assert datamodule.dataset is not None
            assert len(datamodule.dataset) > 0
            
            # check that splits were created
            assert datamodule.dataset_train is not None
            assert datamodule.dataset_val is not None

    def test_train_dataloader(self, create_test_marker_csv):
        """Test train dataloader creation and properties."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # create test files
            marker_file = Path(tmpdir) / 'markers.csv'
            create_test_marker_csv(marker_file, n_frames=40, n_markers=1)
            
            # create data config
            data_config = {
                'ids': ['test_dataset'],
                'signals': [['markers']],
                'transforms': [[None]],
                'paths': [[str(marker_file)]],
            }
            
            # create datamodule
            datamodule = DataModule(
                data_config=data_config,
                sequence_length=5,
                batch_size=2,
                num_workers=0,  # avoid multiprocessing in tests
                persistent_workers=False,
            )
            
            # get train dataloader
            train_loader = datamodule.train_dataloader()
            
            # check dataloader properties
            assert isinstance(train_loader, DataLoader)
            assert train_loader.batch_size == 2
            assert train_loader.dataset == datamodule.dataset_train
            
            # check that we can iterate through batches
            batch = next(iter(train_loader))
            assert 'input' in batch
            assert batch['input'].shape[0] <= 2  # batch size
            assert isinstance(batch['input'], torch.Tensor)

    def test_val_dataloader(self, create_test_feature_csv):
        """Test validation dataloader creation and properties."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # create test files
            feature_file = Path(tmpdir) / 'features.csv'
            create_test_feature_csv(feature_file, n_frames=30, n_features=4)
            
            # create data config
            data_config = {
                'ids': ['test_dataset'],
                'signals': [['features']],
                'transforms': [[None]],
                'paths': [[str(feature_file)]],
            }
            
            # create datamodule
            datamodule = DataModule(
                data_config=data_config,
                sequence_length=5,
                batch_size=3,
                num_workers=0,  # avoid multiprocessing in tests
                persistent_workers=False,
            )
            
            # get val dataloader
            val_loader = datamodule.val_dataloader()
            
            # check dataloader properties
            assert isinstance(val_loader, DataLoader)
            assert val_loader.batch_size == 3
            assert val_loader.dataset == datamodule.dataset_val
            
            # check that we can iterate through batches
            batch = next(iter(val_loader))
            assert 'input' in batch
            assert batch['input'].shape[0] <= 3  # batch size
            assert isinstance(batch['input'], torch.Tensor)

    def test_multiple_datasets(self, create_test_marker_csv):
        """Test DataModule with multiple datasets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # create test files for two datasets
            marker_file1 = Path(tmpdir) / 'markers1.csv'
            marker_file2 = Path(tmpdir) / 'markers2.csv'
            
            create_test_marker_csv(marker_file1, n_frames=20, n_markers=2)
            create_test_marker_csv(marker_file2, n_frames=25, n_markers=2)
            
            # create data config
            data_config = {
                'ids': ['dataset1', 'dataset2'],
                'signals': [['markers'], ['markers']],
                'transforms': [[None], [None]],
                'paths': [[str(marker_file1)], [str(marker_file2)]],
            }
            
            # create datamodule
            datamodule = DataModule(
                data_config=data_config,
                sequence_length=5,
                batch_size=2,
                num_workers=0,
                persistent_workers=False,
            )
            
            # check that both datasets are represented
            dataset_ids = datamodule.get_dataset_ids()
            assert 'dataset1' in dataset_ids
            assert 'dataset2' in dataset_ids
            
            # check dataloaders work
            train_loader = datamodule.train_dataloader()
            val_loader = datamodule.val_dataloader()
            
            assert len(train_loader) > 0
            assert len(val_loader) > 0

    def test_train_val_split_sizes(self, create_test_marker_csv):
        """Test that train/val split produces correct sizes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # create test file
            marker_file = Path(tmpdir) / 'markers.csv'
            create_test_marker_csv(marker_file, n_frames=50, n_markers=1)
            
            # create data config
            data_config = {
                'ids': ['test_dataset'],
                'signals': [['markers']],
                'transforms': [[None]],
                'paths': [[str(marker_file)]],
            }
            
            # create datamodule with specific split
            datamodule = DataModule(
                data_config=data_config,
                sequence_length=5,
                train_probability=0.7,
                val_probability=0.3,
            )
            
            # check split sizes
            total_size = len(datamodule.dataset)
            train_size = len(datamodule.dataset_train)
            val_size = len(datamodule.dataset_val)
            
            assert train_size + val_size == total_size
            assert train_size > 0
            assert val_size > 0
            
            # check approximate proportions (allowing for rounding)
            train_ratio = train_size / total_size
            assert 0.6 <= train_ratio <= 0.8  # should be around 0.7

    def test_setup_stage_test(self, create_test_marker_csv):
        """Test that setup handles test stage correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # create test file
            marker_file = Path(tmpdir) / 'markers.csv'
            create_test_marker_csv(marker_file, n_frames=20, n_markers=1)
            
            # create data config
            data_config = {
                'ids': ['test_dataset'],
                'signals': [['markers']],
                'transforms': [[None]],
                'paths': [[str(marker_file)]],
            }
            
            # create datamodule
            datamodule = DataModule(
                data_config=data_config,
                sequence_length=5,
            )
            
            # call setup with test stage (should do nothing)
            datamodule.setup(stage='test')
            
            # should still have train/val datasets from initialization
            assert datamodule.dataset_train is not None
            assert datamodule.dataset_val is not None

    def test_dataloader_configuration(self, create_test_marker_csv):
        """Test DataLoader configuration parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # create test file
            marker_file = Path(tmpdir) / 'markers.csv'
            create_test_marker_csv(marker_file, n_frames=30, n_markers=1)
            
            # create data config
            data_config = {
                'ids': ['test_dataset'],
                'signals': [['markers']],
                'transforms': [[None]],
                'paths': [[str(marker_file)]],
            }
            
            # create datamodule with specific parameters
            datamodule = DataModule(
                data_config=data_config,
                sequence_length=5,
                batch_size=4,
                num_workers=0,
                pin_memory=False,
                persistent_workers=False,
            )
            
            # check train dataloader configuration
            train_loader = datamodule.train_dataloader()
            assert train_loader.batch_size == 4
            assert train_loader.num_workers == 0
            assert train_loader.pin_memory is False
            assert train_loader.persistent_workers is False
            
            # check val dataloader configuration
            val_loader = datamodule.val_dataloader()
            assert val_loader.batch_size == 4
            assert val_loader.num_workers == 0
            assert val_loader.pin_memory is False
            assert val_loader.persistent_workers is False

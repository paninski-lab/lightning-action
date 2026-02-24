import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from lightning_action.data import (
    compute_sequence_pad,
    compute_sequences,
    split_sizes_from_probabilities,
)
from lightning_action.data.utils import (
    collect_labels_from_datamodule,
    collect_labels_from_files,
    compute_class_weights,
)


class TestComputeSequences:

    def test_compute_sequences_correct_sizes(self):

        # batch sizes and quantity are correct
        T = 10
        N = 4
        B = 5
        data = np.random.randn(T, N)
        batch_data = compute_sequences(data, sequence_length=B)
        assert len(batch_data) == T // B
        assert batch_data[0].shape == (B, N)
        assert np.all(batch_data[0] == data[:B, :])

    def test_compute_sequences_already_batched(self):

        # pass already batched data (in a list) through without modification
        data = [1, 2, 3]
        batch_data = compute_sequences(data, sequence_length=10)
        assert data == batch_data


class TestComputeSequencePad:
    """Test the compute_sequence_pad function."""

    def test_temporal_mlp(self):
        """Test padding calculation for temporal-mlp model."""
        result = compute_sequence_pad('temporal-mlp', num_lags=5)
        assert result == 5

        result = compute_sequence_pad('temporalmlp', num_lags=5)
        assert result == 5

    def test_tcn(self):
        """Test padding calculation for tcn model."""
        result = compute_sequence_pad('tcn', num_layers=3, num_lags=2)
        expected = (2 ** 3) * 2  # 16
        assert result == expected

    def test_dtcn(self):
        """Test padding calculation for dtcn model."""
        result = compute_sequence_pad('dtcn', num_lags=1, num_layers=3)
        expected = sum([2 * (2 ** n) * 1 for n in range(3)])  # 2 + 4 + 8 = 14
        assert result == expected

    def test_lstm(self):
        """Test padding calculation for lstm model."""
        result = compute_sequence_pad('lstm')
        assert result == 4

    def test_gru(self):
        """Test padding calculation for gru model."""
        result = compute_sequence_pad('gru')
        assert result == 4

    def test_case_insensitive(self):
        """Test that model type is case insensitive."""
        result_lower = compute_sequence_pad('temporal-mlp', num_lags=3)
        result_upper = compute_sequence_pad('TEMPORAL-MLP', num_lags=3)
        assert result_lower == result_upper == 3

    def test_unknown_model_type(self):
        """Test that unknown model type raises ValueError."""
        with pytest.raises(ValueError, match='Unknown model type'):
            compute_sequence_pad('unknown-model')

    def test_unknown_model_type_with_default(self):
        """Test that unknown model type returns default when provided."""
        result = compute_sequence_pad('unknown-model', default=0)
        assert result == 0
        
        result = compute_sequence_pad('transformer', default=10)
        assert result == 10


class TestSplitSizesFromProbabilities:
    """Test the split_sizes_from_probabilities function."""

    def test_basic_split(self):
        """Test basic train/val split with probabilities."""
        result = split_sizes_from_probabilities(100, 0.8, 0.2)
        assert result == [80, 20]

    def test_default_val_probability(self):
        """Test with default validation probability."""
        result = split_sizes_from_probabilities(100, 0.7)
        assert result == [70, 30]

    def test_probabilities_sum_to_one(self):
        """Test that probabilities must sum to 1.0."""
        with pytest.raises(AssertionError, match='Split probabilities must add to 1'):
            split_sizes_from_probabilities(100, 0.6, 0.5)

    def test_minimum_validation_samples(self):
        """Test that 0 validation samples is valid when val_probability is 0."""
        # case where val_probability is 0 — all samples go to training
        result = split_sizes_from_probabilities(10, 1.0, 0.0)
        assert result == [10, 0]

    def test_too_few_total_samples(self):
        """Test that a single sample with 0 val probability returns [1, 0]."""
        result = split_sizes_from_probabilities(1, 1.0, 0.0)
        assert result == [1, 0]

    def test_fractional_results(self):
        """Test handling of fractional results."""
        # 33 samples with 0.6/0.4 split should give 19/13 (floors to 19/13 = 32, need +1)
        result = split_sizes_from_probabilities(33, 0.6, 0.4)
        assert sum(result) == 33
        assert result[0] >= 1  # at least 1 train
        assert result[1] >= 1  # at least 1 val

    def test_edge_case_small_numbers(self):
        """Test edge cases with small total numbers."""
        # minimum viable case
        result = split_sizes_from_probabilities(2, 0.5, 0.5)
        assert result == [1, 1]
        
        # slightly larger
        result = split_sizes_from_probabilities(3, 0.67, 0.33)
        assert sum(result) == 3
        assert result[1] >= 0  # val can be 0 when val_probability rounds down


class TestComputeClassWeights:
    """Test the compute_class_weights function."""

    def test_basic_class_weights(self):
        """Test basic class weight computation with balanced classes."""
        labels = np.array([0, 0, 1, 1, 2, 2])
        weights = compute_class_weights(labels, num_classes=3)
        
        # All classes have equal counts, so weights should all be 1.0
        assert len(weights) == 3
        assert all(abs(w - 1.0) < 1e-6 for w in weights)

    def test_imbalanced_class_weights(self):
        """Test class weights with imbalanced classes."""
        # Class 0: 10 samples, Class 1: 5 samples, Class 2: 2 samples
        labels = np.array([0]*10 + [1]*5 + [2]*2)
        weights = compute_class_weights(labels, num_classes=3)
        
        assert len(weights) == 3
        # Most frequent class (0) should have lowest weight (1.0)
        assert abs(weights[0] - 1.0) < 1e-6
        # Less frequent class (1) should have higher weight
        assert weights[1] > weights[0]
        # Least frequent class (2) should have highest weight
        assert weights[2] > weights[1]

    def test_ignore_index_handling(self):
        """Test that ignore_index values are excluded from weight computation."""
        labels = np.array([0, 0, 1, 1, -100, -100, -100])  # -100 is ignore_index
        weights = compute_class_weights(labels, num_classes=2, ignore_index=-100)
        
        assert len(weights) == 2
        # Both classes have equal counts (2 each)
        assert all(abs(w - 1.0) < 1e-6 for w in weights)

    def test_custom_ignore_index(self):
        """Test with custom ignore_index value."""
        labels = np.array([0, 0, 1, 1, 99, 99])  # 99 is ignore
        weights = compute_class_weights(labels, num_classes=2, ignore_index=99)
        
        assert len(weights) == 2
        assert all(abs(w - 1.0) < 1e-6 for w in weights)

    def test_sqrt_dampening(self):
        """Test that sqrt_dampening reduces weight magnitude for rare classes."""
        labels = np.array([0]*100 + [1])  # Highly imbalanced
        
        weights_no_sqrt = compute_class_weights(
            labels, num_classes=2, sqrt_dampening=False
        )
        weights_sqrt = compute_class_weights(
            labels, num_classes=2, sqrt_dampening=True
        )
        
        # Weight for rare class should be lower with sqrt dampening
        assert weights_sqrt[1] < weights_no_sqrt[1]
        # sqrt(100) = 10 for class 1 with dampening vs 100 without
        assert abs(weights_sqrt[1] - 10.0) < 0.1

    def test_infer_num_classes(self):
        """Test that num_classes is inferred from labels when not provided."""
        labels = np.array([0, 1, 2, 3, 4])
        weights = compute_class_weights(labels, num_classes=None)
        
        # Should infer 5 classes (0-4)
        assert len(weights) == 5

    def test_missing_class_zero_weight(self):
        """Test that classes with no samples get weight 0.0."""
        labels = np.array([0, 0, 2, 2])  # Class 1 is missing
        weights = compute_class_weights(labels, num_classes=3)
        
        assert weights[0] > 0
        assert weights[1] == 0.0  # Missing class
        assert weights[2] > 0

    def test_empty_labels(self):
        """Test handling of empty label array."""
        labels = np.array([])
        weights = compute_class_weights(labels, num_classes=3)
        
        # Should return uniform weights
        assert len(weights) == 3
        assert all(w == 1.0 for w in weights)

    def test_all_ignored_labels(self):
        """Test when all labels are ignore_index."""
        labels = np.array([-100, -100, -100])
        weights = compute_class_weights(labels, num_classes=3, ignore_index=-100)
        
        # Should return uniform weights
        assert len(weights) == 3
        assert all(w == 1.0 for w in weights)

    def test_multidimensional_input(self):
        """Test that multi-dimensional input is flattened."""
        labels = np.array([[0, 1], [2, 0], [1, 2]])
        weights = compute_class_weights(labels, num_classes=3)
        
        assert len(weights) == 3

    def test_single_class(self):
        """Test with only one class present."""
        labels = np.array([0, 0, 0, 0, 0])
        weights = compute_class_weights(labels, num_classes=1)
        
        assert len(weights) == 1
        assert abs(weights[0] - 1.0) < 1e-6

    def test_single_class_with_multiple_declared(self):
        """Test with one class present but multiple declared."""
        labels = np.array([0, 0, 0, 0, 0])
        weights = compute_class_weights(labels, num_classes=3)
        
        assert len(weights) == 3
        assert weights[0] > 0
        assert weights[1] == 0.0  # No samples
        assert weights[2] == 0.0  # No samples


class TestCollectLabelsFromFiles:
    """Test the collect_labels_from_files function for video pipeline."""

    def test_collect_1d_labels(self):
        """Test collecting 1D label arrays (class indices)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test label files
            labels1 = np.array([0, 1, 2, 0, 1])
            labels2 = np.array([2, 2, 1, 0])
            
            path1 = os.path.join(tmpdir, 'video1.npy')
            path2 = os.path.join(tmpdir, 'video2.npy')
            
            np.save(path1, labels1)
            np.save(path2, labels2)
            
            all_labels, num_classes = collect_labels_from_files(
                [path1, path2], show_progress=False
            )
            
            assert len(all_labels) == 9  # 5 + 4
            assert num_classes == 3  # Classes 0, 1, 2
            assert set(all_labels.tolist()) == {0, 1, 2}

    def test_collect_2d_onehot_labels(self):
        """Test collecting 2D one-hot encoded label arrays."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create one-hot encoded labels
            labels1 = np.array([
                [1, 0, 0],  # Class 0
                [0, 1, 0],  # Class 1
                [0, 0, 1],  # Class 2
            ])
            labels2 = np.array([
                [0, 1, 0],  # Class 1
                [0, 0, 1],  # Class 2
            ])
            
            path1 = os.path.join(tmpdir, 'video1.npy')
            path2 = os.path.join(tmpdir, 'video2.npy')
            
            np.save(path1, labels1)
            np.save(path2, labels2)
            
            all_labels, num_classes = collect_labels_from_files(
                [path1, path2], show_progress=False
            )
            
            assert len(all_labels) == 5  # 3 + 2
            assert num_classes == 3  # Inferred from shape
            # Labels should be converted to class indices
            assert all_labels[0] == 0
            assert all_labels[1] == 1
            assert all_labels[2] == 2

    def test_collect_with_ignore_index(self):
        """Test that ignore_index is properly excluded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            labels = np.array([0, 1, -100, 2, -100])
            path = os.path.join(tmpdir, 'video.npy')
            np.save(path, labels)
            
            all_labels, num_classes = collect_labels_from_files(
                [path], ignore_index=-100, show_progress=False
            )
            
            # All labels are returned, but num_classes is computed ignoring -100
            assert len(all_labels) == 5
            assert num_classes == 3  # max(0, 1, 2) + 1

    def test_empty_file_list(self):
        """Test with empty file list."""
        all_labels, num_classes = collect_labels_from_files(
            [], show_progress=False
        )
        
        assert len(all_labels) == 0
        assert num_classes == 0

    def test_single_file(self):
        """Test with single label file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            labels = np.array([0, 1, 1, 0])
            path = os.path.join(tmpdir, 'video.npy')
            np.save(path, labels)
            
            all_labels, num_classes = collect_labels_from_files(
                [path], show_progress=False
            )
            
            assert len(all_labels) == 4
            assert num_classes == 2

    def test_all_ignored_labels_in_file(self):
        """Test file with all ignored labels."""
        with tempfile.TemporaryDirectory() as tmpdir:
            labels = np.array([-100, -100, -100])
            path = os.path.join(tmpdir, 'video.npy')
            np.save(path, labels)
            
            # Also add a normal file to get num_classes
            labels2 = np.array([0, 1])
            path2 = os.path.join(tmpdir, 'video2.npy')
            np.save(path2, labels2)
            
            all_labels, num_classes = collect_labels_from_files(
                [path, path2], ignore_index=-100, show_progress=False
            )
            
            assert len(all_labels) == 5  # 3 + 2
            assert num_classes == 2

    def test_squeezable_2d_labels(self):
        """Test 2D labels with shape (n, 1) that need squeezing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Shape (n, 1) - not one-hot, just wrong shape
            labels = np.array([[0], [1], [2], [1]])
            path = os.path.join(tmpdir, 'video.npy')
            np.save(path, labels)
            
            all_labels, num_classes = collect_labels_from_files(
                [path], show_progress=False
            )
            
            assert all_labels.shape == (4,)  # Should be squeezed
            assert num_classes == 3


class TestCollectLabelsFromDatamodule:
    """Test the collect_labels_from_datamodule function for CSV pipeline."""

    def test_basic_collection(self):
        """Test basic label collection from mock dataset."""
        # Create mock dataset
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=3)
        mock_dataset.__getitem__ = MagicMock(side_effect=[
            {'labels': np.array([0, 1, 2])},
            {'labels': np.array([1, 2, 0])},
            {'labels': np.array([2, 0, 1])},
        ])
        
        all_labels, num_classes = collect_labels_from_datamodule(
            mock_dataset, show_progress=False
        )
        
        assert len(all_labels) == 9  # 3 * 3
        assert num_classes == 3

    def test_collection_with_torch_tensors(self):
        """Test label collection when dataset returns torch tensors."""
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=2)
        mock_dataset.__getitem__ = MagicMock(side_effect=[
            {'labels': torch.tensor([0, 1, 2])},
            {'labels': torch.tensor([3, 4])},
        ])
        
        all_labels, num_classes = collect_labels_from_datamodule(
            mock_dataset, show_progress=False
        )
        
        assert len(all_labels) == 5
        assert num_classes == 5  # 0-4

    def test_collection_with_onehot_labels(self):
        """Test label collection with one-hot encoded labels."""
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=2)
        mock_dataset.__getitem__ = MagicMock(side_effect=[
            {'labels': np.array([[1, 0, 0], [0, 1, 0]])},  # Classes 0, 1
            {'labels': np.array([[0, 0, 1]])},  # Class 2
        ])
        
        all_labels, num_classes = collect_labels_from_datamodule(
            mock_dataset, show_progress=False
        )
        
        assert len(all_labels) == 3
        assert num_classes == 3
        assert list(all_labels) == [0, 1, 2]

    def test_collection_with_ignore_index(self):
        """Test that ignore_index is properly handled."""
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=2)
        mock_dataset.__getitem__ = MagicMock(side_effect=[
            {'labels': np.array([0, -100, 1])},
            {'labels': np.array([-100, 2])},
        ])
        
        all_labels, num_classes = collect_labels_from_datamodule(
            mock_dataset, ignore_index=-100, show_progress=False
        )
        
        assert len(all_labels) == 5  # All labels returned
        assert num_classes == 3  # But num_classes computed ignoring -100

    def test_empty_dataset(self):
        """Test with empty dataset."""
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=0)
        mock_dataset.label_names = ['class_0', 'class_1', 'class_2']
        
        all_labels, num_classes = collect_labels_from_datamodule(
            mock_dataset, show_progress=False
        )
        
        assert len(all_labels) == 0
        assert num_classes == 3  # From label_names

    def test_missing_labels_key(self):
        """Test dataset items missing 'labels' key."""
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=2)
        mock_dataset.__getitem__ = MagicMock(side_effect=[
            {'input': np.array([1, 2, 3])},  # No 'labels' key
            {'labels': np.array([0, 1])},
        ])
        mock_dataset.label_names = ['a', 'b']
        
        all_labels, num_classes = collect_labels_from_datamodule(
            mock_dataset, show_progress=False
        )
        
        # Should only get labels from second item
        assert len(all_labels) == 2
        assert num_classes == 2

    def test_fallback_to_label_names_attribute(self):
        """Test falling back to dataset.label_names for num_classes."""
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=1)
        mock_dataset.__getitem__ = MagicMock(return_value={'input': np.array([1])})
        mock_dataset.label_names = ['behavior_a', 'behavior_b', 'behavior_c']
        
        all_labels, num_classes = collect_labels_from_datamodule(
            mock_dataset, show_progress=False
        )
        
        assert len(all_labels) == 0
        assert num_classes == 3


class TestComputeClassWeightsIntegration:
    """Integration tests combining collect_* functions with compute_class_weights."""

    def test_video_pipeline_weights(self):
        """Test full video pipeline: files -> collect -> weights."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create realistic video label files
            # Video 1: mostly class 0 with some class 1
            labels1 = np.concatenate([
                np.zeros(80),  # 80 frames of class 0
                np.ones(20),   # 20 frames of class 1
            ]).astype(np.int64)
            
            # Video 2: balanced
            labels2 = np.concatenate([
                np.zeros(50),
                np.ones(50),
            ]).astype(np.int64)
            
            path1 = os.path.join(tmpdir, 'video1.npy')
            path2 = os.path.join(tmpdir, 'video2.npy')
            np.save(path1, labels1)
            np.save(path2, labels2)
            
            # Collect labels
            all_labels, num_classes = collect_labels_from_files(
                [path1, path2], show_progress=False
            )
            
            # Compute weights (video pipeline uses sqrt_dampening=True)
            weights = compute_class_weights(
                all_labels, num_classes=num_classes, sqrt_dampening=True
            )
            
            # Class 0: 130 samples, Class 1: 70 samples
            # Weight for class 0 should be 1.0 (most frequent)
            # Weight for class 1 should be sqrt(130/70) ≈ 1.36
            assert len(weights) == 2
            assert abs(weights[0] - 1.0) < 1e-6
            assert 1.3 < weights[1] < 1.5

    def test_csv_pipeline_weights(self):
        """Test full CSV pipeline: datamodule -> collect -> weights."""
        # Create mock dataset mimicking FeatureDataset behavior
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=5)
        
        # Simulate sequences with one-hot labels
        # Create imbalanced data: more class 0 than class 1
        mock_dataset.__getitem__ = MagicMock(side_effect=[
            {'labels': np.array([[1, 0]] * 100)},  # 100 frames of class 0
            {'labels': np.array([[1, 0]] * 100)},  # 100 frames of class 0
            {'labels': np.array([[1, 0]] * 80 + [[0, 1]] * 20)},  # 80 class 0, 20 class 1
            {'labels': np.array([[0, 1]] * 50)},  # 50 frames of class 1
            {'labels': np.array([[0, 1]] * 30)},  # 30 frames of class 1
        ])
        
        # Collect labels
        all_labels, num_classes = collect_labels_from_datamodule(
            mock_dataset, show_progress=False
        )
        
        # Compute weights (CSV pipeline uses sqrt_dampening=False)
        weights = compute_class_weights(
            all_labels, num_classes=num_classes, sqrt_dampening=False
        )
        
        # Class 0: 280 samples, Class 1: 100 samples
        # Weight for class 0 should be 1.0
        # Weight for class 1 should be 280/100 = 2.8
        assert len(weights) == 2
        assert abs(weights[0] - 1.0) < 1e-6
        assert abs(weights[1] - 2.8) < 0.1

    def test_multiclass_weights(self):
        """Test class weights with more than 2 classes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 4-class problem with varying frequencies
            labels = np.array([0]*1000 + [1]*500 + [2]*250 + [3]*100)
            path = os.path.join(tmpdir, 'video.npy')
            np.save(path, labels)
            
            all_labels, num_classes = collect_labels_from_files(
                [path], show_progress=False
            )
            weights = compute_class_weights(
                all_labels, num_classes=num_classes, sqrt_dampening=False
            )
            
            assert len(weights) == 4
            # Weights should be in increasing order (class 0 lowest, class 3 highest)
            assert weights[0] < weights[1] < weights[2] < weights[3]
            # Most frequent class weight should be 1.0
            assert abs(weights[0] - 1.0) < 1e-6
            # Rarest class weight should be 10.0 (1000/100)
            assert abs(weights[3] - 10.0) < 0.1

    def test_with_padding_frames(self):
        """Test handling of ignore_index padding frames (typical in video)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Labels with padding at boundaries (common in video segmentation)
            labels = np.array(
                [-100] * 32 +  # Padding at start
                [0] * 100 +
                [1] * 50 +
                [-100] * 32    # Padding at end
            )
            path = os.path.join(tmpdir, 'video.npy')
            np.save(path, labels)
            
            all_labels, num_classes = collect_labels_from_files(
                [path], ignore_index=-100, show_progress=False
            )
            weights = compute_class_weights(
                all_labels, num_classes=num_classes, ignore_index=-100
            )
            
            # Only valid labels (0 and 1) should be counted
            assert len(weights) == 2
            assert abs(weights[0] - 1.0) < 1e-6  # Class 0 is most frequent
            assert abs(weights[1] - 2.0) < 0.1   # Class 1 is half as frequent

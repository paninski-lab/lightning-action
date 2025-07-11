import numpy as np
import pytest

from lightning_action.data import (
    compute_sequence_pad,
    compute_sequences,
    split_sizes_from_probabilities,
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
        result = compute_sequence_pad('temporal-mlp', n_lags=5)
        assert result == 5

    def test_tcn(self):
        """Test padding calculation for tcn model."""
        result = compute_sequence_pad('tcn', n_layers=3, n_lags=2)
        expected = (2 ** 3) * 2  # 16
        assert result == expected

    def test_dtcn(self):
        """Test padding calculation for dtcn model."""
        result = compute_sequence_pad('dtcn', n_lags=1, n_hid_layers=3)
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
        result_lower = compute_sequence_pad('temporal-mlp', n_lags=3)
        result_upper = compute_sequence_pad('TEMPORAL-MLP', n_lags=3)
        assert result_lower == result_upper == 3

    def test_unknown_model_type(self):
        """Test that unknown model type raises ValueError."""
        with pytest.raises(ValueError, match='Unknown model type'):
            compute_sequence_pad('unknown-model')


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
        """Test that at least one validation sample is guaranteed."""
        # case where val_probability would give 0 samples
        result = split_sizes_from_probabilities(10, 1.0, 0.0)
        assert result == [9, 1]  # should adjust to ensure 1 val sample

    def test_too_few_total_samples(self):
        """Test error when not enough samples for train and val."""
        with pytest.raises(ValueError, match='Must have at least two sequences'):
            split_sizes_from_probabilities(1, 1.0, 0.0)

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
        assert result[1] >= 1  # ensure at least 1 val

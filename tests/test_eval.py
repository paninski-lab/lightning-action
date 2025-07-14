"""Tests for evaluation functions."""

import numpy as np
import pytest

from lightning_action.eval import get_precision_recall, int_over_union, run_lengths


class TestGetPrecisionRecall:
    """Test the get_precision_recall function."""

    def test_perfect_classification(self):
        """Test precision and recall with perfect classification."""
        true_classes = np.array([0, 1, 2, 1, 2, 0, 1, 2])
        pred_classes = np.array([0, 1, 2, 1, 2, 0, 1, 2])
        
        result = get_precision_recall(true_classes, pred_classes, background=None)
        
        # perfect classification should have precision and recall of 1.0
        assert np.allclose(result['precision'], [1.0, 1.0, 1.0])
        assert np.allclose(result['recall'], [1.0, 1.0, 1.0])
        assert np.allclose(result['f1'], [1.0, 1.0, 1.0])

    def test_with_background_class(self):
        """Test precision and recall with background class."""
        # background=0, classes 1,2,3
        true_classes = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        pred_classes = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        
        result = get_precision_recall(true_classes, pred_classes, background=0)
        
        # should exclude background class from metrics
        assert len(result['precision']) == 3  # classes 1, 2, 3
        assert np.allclose(result['precision'], [1.0, 1.0, 1.0])
        assert np.allclose(result['recall'], [1.0, 1.0, 1.0])

    def test_imperfect_classification(self):
        """Test precision and recall with classification errors."""
        true_classes = np.array([0, 1, 1, 1, 2, 2, 2])
        pred_classes = np.array([0, 1, 1, 2, 2, 2, 1])  # one error each class
        
        result = get_precision_recall(true_classes, pred_classes, background=None)
        
        # class 1: 2 correct predictions out of 3 true → recall = 2/3
        # class 1: 2 correct predictions out of 3 predicted → precision = 2/3
        # class 2: 2 correct predictions out of 3 true → recall = 2/3  
        # class 2: 2 correct predictions out of 3 predicted → precision = 2/3
        expected_precision = [1, 2/3, 2/3]
        expected_recall = [1, 2/3, 2/3]
        
        assert np.allclose(result['precision'], expected_precision, atol=1e-6)
        assert np.allclose(result['recall'], expected_recall, atol=1e-6)

    def test_specified_n_classes(self):
        """Test with specified number of classes."""
        true_classes = np.array([1, 1, 2, 2])
        pred_classes = np.array([1, 1, 2, 2])
        
        result = get_precision_recall(true_classes, pred_classes, background=0, n_classes=3)
        
        # should return metrics for 3 classes even though class 3 not present
        assert len(result['precision']) == 3

    def test_mismatched_array_sizes(self):
        """Test error handling for mismatched array sizes."""
        true_classes = np.array([1, 2, 3])
        pred_classes = np.array([1, 2])  # different size
        
        with pytest.raises(AssertionError):
            get_precision_recall(true_classes, pred_classes)

    def test_single_class(self):
        """Test with single class data."""
        true_classes = np.array([1, 1, 1, 1])
        pred_classes = np.array([1, 1, 1, 1])
        
        result = get_precision_recall(true_classes, pred_classes, background=0)
        
        assert len(result['precision']) == 1
        assert result['precision'][0] == 1.0
        assert result['recall'][0] == 1.0

    def test_f1_score_calculation(self):
        """Test F1 score calculation."""
        # create scenario where precision != recall
        true_classes = np.array([1, 1, 1, 2, 2])
        pred_classes = np.array([1, 1, 2, 2, 2])
        
        result = get_precision_recall(true_classes, pred_classes, background=None)
        
        # manually calculate expected F1
        p = result['precision']
        r = result['recall']
        expected_f1 = 2 * p * r / (p + r + 1e-10)
        
        assert np.allclose(result['f1'], expected_f1)

    def test_zero_division_handling(self):
        """Test handling of zero division in metrics."""
        true_classes = np.array([1, 1, 1])
        pred_classes = np.array([2, 2, 2])  # no correct predictions
        
        result = get_precision_recall(true_classes, pred_classes, background=None)
        
        # should handle zero division gracefully
        assert not np.isnan(result['precision']).any()
        assert not np.isnan(result['recall']).any()


class TestIntOverUnion:
    """Test the int_over_union function."""

    def test_identical_arrays(self):
        """Test IoU with identical arrays."""
        array1 = np.array([1, 1, 2, 2, 3, 3])
        array2 = np.array([1, 1, 2, 2, 3, 3])
        
        result = int_over_union(array1, array2)
        
        # IoU should be 1.0 for all classes when arrays are identical
        for val in [1, 2, 3]:
            assert result[val] == 1.0

    def test_no_overlap(self):
        """Test IoU with no overlap."""
        array1 = np.array([1, 1, 1])
        array2 = np.array([2, 2, 2])
        
        result = int_over_union(array1, array2)
        
        # no overlap should give IoU of 0
        assert result[1] == 0.0
        assert result[2] == 0.0

    def test_partial_overlap(self):
        """Test IoU with partial overlap."""
        array1 = np.array([1, 1, 1, 2, 2])
        array2 = np.array([1, 1, 2, 2, 2])
        
        result = int_over_union(array1, array2)
        
        # class 1: intersection=2, union=3 → IoU = 2/3
        # class 2: intersection=2, union=3 → IoU = 2/3
        assert np.isclose(result[1], 2/3)
        assert np.isclose(result[2], 2/3)

    def test_single_element_arrays(self):
        """Test IoU with single element arrays."""
        array1 = np.array([1])
        array2 = np.array([1])
        
        result = int_over_union(array1, array2)
        
        assert result[1] == 1.0

    def test_empty_intersection(self):
        """Test IoU when one class has no intersection."""
        array1 = np.array([1, 1, 2, 3])
        array2 = np.array([1, 2, 2, 2])
        
        result = int_over_union(array1, array2)
        
        # class 3 appears in array1 but not array2
        assert result[3] == 0.0

    def test_different_value_ranges(self):
        """Test IoU with different value ranges."""
        array1 = np.array([0, 1, 2])
        array2 = np.array([2, 3, 2])
        
        result = int_over_union(array1, array2)
        
        # only class 2 overlaps
        assert result[2] == 1/2  # intersection=1, union=2
        assert result[0] == 0.0
        assert result[1] == 0.0
        assert result[3] == 0.0

    def test_result_keys(self):
        """Test that result contains keys for all unique values."""
        array1 = np.array([1, 3, 5])
        array2 = np.array([2, 4, 5])
        
        result = int_over_union(array1, array2)
        
        expected_keys = {1, 2, 3, 4, 5}
        assert set(result.keys()) == expected_keys

    def test_large_arrays(self):
        """Test IoU with larger arrays."""
        array1 = np.random.randint(0, 3, 1000)
        array2 = np.random.randint(0, 3, 1000)
        
        result = int_over_union(array1, array2)
        
        # basic sanity checks
        assert all(0 <= iou <= 1 for iou in result.values() if not np.isnan(iou))
        assert set(result.keys()) == set(np.unique(np.concatenate([array1, array2])))


class TestRunLengths:
    """Test the run_lengths function."""

    def test_example_from_docstring(self):
        """Test with the example from the function docstring."""
        array = np.array([1, 1, 1, 0, 0, 4, 4, 4, 4, 4, 4, 0, 1, 1, 1, 1])
        
        result = run_lengths(array)
        
        expected = {0: [2, 1], 1: [3, 4], 2: [], 3: [], 4: [6]}
        assert result == expected

    def test_single_run(self):
        """Test with single continuous run."""
        array = np.array([2, 2, 2, 2, 2])
        
        result = run_lengths(array)
        
        expected = {0: [], 1: [], 2: [5]}
        assert result == expected

    def test_alternating_pattern(self):
        """Test with alternating pattern."""
        array = np.array([0, 1, 0, 1, 0])
        
        result = run_lengths(array)
        
        expected = {0: [1, 1, 1], 1: [1, 1]}
        assert result == expected

    def test_single_element(self):
        """Test with single element array."""
        array = np.array([3])
        
        result = run_lengths(array)
        
        expected = {0: [], 1: [], 2: [], 3: [1]}
        assert result == expected

    def test_all_same_value(self):
        """Test with all same values."""
        array = np.array([7, 7, 7, 7])
        
        result = run_lengths(array)
        
        # should have keys for 0 through 7
        expected_keys = set(range(8))
        assert set(result.keys()) == expected_keys
        assert result[7] == [4]
        for i in range(7):
            assert result[i] == []

    def test_sequential_values(self):
        """Test with sequential increasing values."""
        array = np.array([0, 1, 2, 3, 4])
        
        result = run_lengths(array)
        
        expected = {0: [1], 1: [1], 2: [1], 3: [1], 4: [1]}
        assert result == expected

    def test_complex_pattern(self):
        """Test with complex run pattern."""
        array = np.array([1, 1, 2, 3, 3, 3, 1, 2, 2, 2, 2])
        
        result = run_lengths(array)
        
        expected = {0: [], 1: [2, 1], 2: [1, 4], 3: [3]}
        assert result == expected

    def test_zero_in_array(self):
        """Test with zeros in the array."""
        array = np.array([0, 0, 1, 1, 0, 2])
        
        result = run_lengths(array)
        
        expected = {0: [2, 1], 1: [2], 2: [1]}
        assert result == expected

    def test_result_structure(self):
        """Test that result has correct structure."""
        array = np.array([1, 2, 3])
        
        result = run_lengths(array)
        
        # should have keys from 0 to max value
        assert set(result.keys()) == {0, 1, 2, 3}
        # all values should be lists
        assert all(isinstance(v, list) for v in result.values())

    def test_large_array(self):
        """Test with larger array."""
        # create array with known pattern
        array = np.array([0] * 100 + [1] * 50 + [2] * 25)
        
        result = run_lengths(array)
        
        expected = {0: [100], 1: [50], 2: [25]}
        assert result == expected

    def test_max_value_calculation(self):
        """Test that keys go up to max value in array."""
        array = np.array([1, 5, 2])  # max value is 5
        
        result = run_lengths(array)
        
        # should have keys 0, 1, 2, 3, 4, 5
        expected_keys = set(range(6))
        assert set(result.keys()) == expected_keys
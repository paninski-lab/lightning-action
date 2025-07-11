"""Tests for data transform classes."""

import numpy as np
import pytest

from lightning_action.data import Compose, MotionEnergy, Transform, ZScore


class MockTransform(Transform):
    """Mock transform for testing purposes."""
    
    def __init__(self, add_value: float = 1.0):
        self.add_value = add_value
    
    def __call__(self, data):
        return data + self.add_value
    
    def __repr__(self):
        return f'MockTransform(add_value={self.add_value})'


class TestTransform:
    """Test the abstract Transform class."""
    
    def test_abstract_methods(self):
        """Test that Transform cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Transform()


class TestCompose:
    """Test the Compose class."""
    
    def test_single_transform(self):
        """Test compose with single transform."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        transform = Compose([MockTransform(add_value=1.0)])
        
        result = transform(data)
        expected = data + 1.0
        
        np.testing.assert_array_equal(result, expected)
    
    def test_multiple_transforms(self):
        """Test compose with multiple transforms."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        transform = Compose([
            MockTransform(add_value=1.0),
            MockTransform(add_value=2.0),
        ])
        
        result = transform(data)
        expected = data + 1.0 + 2.0  # applied sequentially
        
        np.testing.assert_array_equal(result, expected)
    
    def test_empty_transforms(self):
        """Test compose with empty transform list."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        transform = Compose([])
        
        result = transform(data)
        
        np.testing.assert_array_equal(result, data)
    
    def test_repr(self):
        """Test string representation."""
        transform = Compose([
            MockTransform(add_value=1.0),
            MockTransform(add_value=2.0),
        ])
        
        repr_str = repr(transform)
        assert 'Compose(' in repr_str
        assert 'MockTransform(add_value=1.0)' in repr_str
        assert 'MockTransform(add_value=2.0)' in repr_str
        assert repr_str.endswith(')')
    
    def test_repr_empty(self):
        """Test string representation with empty transforms."""
        transform = Compose([])
        repr_str = repr(transform)
        assert repr_str == 'Compose()'


class TestMotionEnergy:
    """Test the MotionEnergy class."""
    
    def test_basic_motion_energy(self):
        """Test basic motion energy computation."""
        # create data with known differences
        data = np.array([
            [1.0, 1.0],
            [2.0, 3.0],
            [4.0, 2.0],
        ], dtype=np.float32)
        
        transform = MotionEnergy()
        result = transform(data)
        
        # first row should be zeros
        np.testing.assert_array_equal(result[0], [0.0, 0.0])
        
        # second row should be abs diff of rows 0 and 1
        np.testing.assert_array_equal(result[1], [1.0, 2.0])
        
        # third row should be abs diff of rows 1 and 2
        np.testing.assert_array_equal(result[2], [2.0, 1.0])
    
    def test_single_timepoint(self):
        """Test motion energy with single time point."""
        data = np.array([[1.0, 2.0]], dtype=np.float32)
        
        transform = MotionEnergy()
        result = transform(data)
        
        # should return zeros with same shape
        expected = np.zeros_like(data)
        np.testing.assert_array_equal(result, expected)
    
    def test_empty_data(self):
        """Test motion energy with empty data."""
        data = np.array([], dtype=np.float32).reshape(0, 2)
        
        transform = MotionEnergy()
        result = transform(data)
        
        # should return zeros with same shape
        expected = np.zeros_like(data)
        np.testing.assert_array_equal(result, expected)
    
    def test_negative_values(self):
        """Test motion energy with negative values."""
        data = np.array([
            [1.0, -1.0],
            [-2.0, 3.0],
        ], dtype=np.float32)
        
        transform = MotionEnergy()
        result = transform(data)
        
        # first row should be zeros
        np.testing.assert_array_equal(result[0], [0.0, 0.0])
        
        # second row should be absolute differences
        np.testing.assert_array_equal(result[1], [3.0, 4.0])
    
    def test_output_shape(self):
        """Test that output shape matches input shape."""
        data = np.random.rand(10, 5).astype(np.float32)
        
        transform = MotionEnergy()
        result = transform(data)
        
        assert result.shape == data.shape
    
    def test_repr(self):
        """Test string representation."""
        transform = MotionEnergy()
        assert repr(transform) == 'MotionEnergy()'


class TestZScore:
    """Test the ZScore class."""
    
    def test_basic_zscore(self):
        """Test basic z-score normalization."""
        # create data with known mean and std
        data = np.array([
            [0.0, 10.0],
            [1.0, 20.0],
            [2.0, 30.0],
        ], dtype=np.float32)
        
        transform = ZScore()
        result = transform(data)
        
        # check that means are approximately zero
        means = np.mean(result, axis=0)
        np.testing.assert_allclose(means, [0.0, 0.0], atol=1e-6)
        
        # check that stds are approximately one
        stds = np.std(result, axis=0)
        np.testing.assert_allclose(stds, [1.0, 1.0], atol=1e-6)
    
    def test_zero_variance_features(self):
        """Test z-score with zero-variance features."""
        data = np.array([
            [1.0, 5.0],
            [2.0, 5.0],  # second feature has zero variance
            [3.0, 5.0],
        ], dtype=np.float32)
        
        transform = ZScore()
        result = transform(data)
        
        # first feature should be normalized
        means = np.mean(result, axis=0)
        stds = np.std(result, axis=0)
        np.testing.assert_allclose(means[0], 0.0, atol=1e-6)
        np.testing.assert_allclose(stds[0], 1.0, atol=1e-6)
        
        # second feature should remain unchanged
        np.testing.assert_array_equal(result[:, 1], [0.0, 0.0, 0.0])  # mean subtracted
    
    def test_single_timepoint(self):
        """Test z-score with single time point."""
        data = np.array([[1.0, 2.0]], dtype=np.float32)
        
        transform = ZScore()
        result = transform(data)
        
        # with single time point, std is 0, should get zeros after mean subtraction
        expected = np.array([[0.0, 0.0]], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)
    
    def test_custom_eps(self):
        """Test z-score with custom epsilon value."""
        data = np.array([
            [1.0, 5.0],
            [1.0, 5.0],  # both features have zero variance
        ], dtype=np.float32)
        
        transform = ZScore(eps=1e-3)
        result = transform(data)
        
        # should get zeros after mean subtraction (no division by small std)
        expected = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)
    
    def test_immutable_input(self):
        """Test that input data is not modified."""
        data = np.array([
            [0.0, 10.0],
            [1.0, 20.0],
            [2.0, 30.0],
        ], dtype=np.float32)
        original_data = data.copy()
        
        transform = ZScore()
        result = transform(data)
        
        # input should be unchanged
        np.testing.assert_array_equal(data, original_data)
        
        # output should be different
        assert not np.array_equal(result, data)
    
    def test_output_shape(self):
        """Test that output shape matches input shape."""
        data = np.random.rand(8, 4).astype(np.float32)
        
        transform = ZScore()
        result = transform(data)
        
        assert result.shape == data.shape
    
    def test_repr(self):
        """Test string representation."""
        transform = ZScore()
        assert repr(transform) == 'ZScore(eps=1e-08)'
        
        transform_custom = ZScore(eps=1e-5)
        assert repr(transform_custom) == 'ZScore(eps=1e-05)'


class TestTransformIntegration:
    """Test transform integration and edge cases."""
    
    def test_compose_with_real_transforms(self):
        """Test compose with actual transforms."""
        data = np.array([
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [1.0, 1.0],
        ], dtype=np.float32)
        
        # create composite transform
        transform = Compose([
            ZScore(),
            MotionEnergy(),
        ])
        
        result = transform(data)
        
        # check shape is preserved
        assert result.shape == data.shape
        
        # check that first row is zeros (from motion energy)
        np.testing.assert_array_equal(result[0], [0.0, 0.0])
    
    def test_transform_with_different_dtypes(self):
        """Test transforms with different numpy dtypes."""
        data_float32 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        data_float64 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        
        transform = ZScore()
        
        result_32 = transform(data_float32)
        result_64 = transform(data_float64)
        
        # results should be close regardless of input dtype
        np.testing.assert_allclose(result_32, result_64, rtol=1e-6)
    
    def test_transform_with_large_arrays(self):
        """Test transforms with larger arrays."""
        # create larger random data
        np.random.seed(42)
        data = np.random.randn(1000, 10).astype(np.float32)
        
        transform = Compose([
            ZScore(),
            MotionEnergy(),
        ])
        
        result = transform(data)
        
        # check shape preservation
        assert result.shape == data.shape
        
        # check that it doesn't crash and produces reasonable output
        assert np.isfinite(result).all()

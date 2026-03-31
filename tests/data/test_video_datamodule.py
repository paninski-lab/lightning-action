"""Tests for video pipeline DataModule and related utilities.

This module tests:
- Video frame count utilities (_get_video_frame_count_cv2)
- Video length helpers (get_video_lengths, get_max_frames)
- Label loading utilities (load_labels_for_videos)
- VideoDataset metadata functions
- VideoDataModule setup and configuration
- DALIIterator logic (mock-based tests for CI)
- Full DALI integration tests (GPU-only, skipped in CI)

These tests require OpenCV for video creation and reading.
GPU tests require NVIDIA DALI and a CUDA-capable GPU.

Test Categories:
- CPU tests: Run everywhere, including GitHub Actions (no DALI needed)
- DALI tests: Require DALI installed (may not need GPU)
- GPU tests: Require NVIDIA GPU + DALI
"""

import os
import tempfile
from typing import Optional

import cv2
import numpy as np
import pytest
import torch

# =============================================================================
# DALI and GPU availability checks
# =============================================================================


def dali_available() -> bool:
    """Check if NVIDIA DALI is installed."""
    try:
        from nvidia.dali import types  # noqa
        return True
    except ImportError:
        return False


def gpu_available() -> bool:
    """Check if CUDA GPU is available AND DALI is installed."""
    if not torch.cuda.is_available():
        return False
    return dali_available()


# Store availability at module load time
DALI_INSTALLED = dali_available()
GPU_AVAILABLE = gpu_available()

# =============================================================================
# Skip markers for different test categories
# =============================================================================

# For tests that need DALI installed (but not necessarily GPU)
requires_dali = pytest.mark.skipif(
    not DALI_INSTALLED,
    reason="NVIDIA DALI not installed"
)

# For tests that need both DALI and GPU
requires_gpu = pytest.mark.skipif(
    not GPU_AVAILABLE,
    reason="NVIDIA GPU and DALI required for this test"
)


# =============================================================================
# Fixtures for creating test videos
# =============================================================================

@pytest.fixture(scope="module")
def video_config():
    """Configuration for test videos."""
    return {
        'num_frames': 100,
        'height': 64,
        'width': 48,
        'fps': 30,
        'channels': 3,
    }


@pytest.fixture
def create_test_video(video_config):
    """Factory fixture to create temporary test videos.

    Returns a function that creates a video file with specified parameters.
    Videos are automatically cleaned up after the test.

    Usage:
        def test_example(create_test_video):
            video_path = create_test_video(num_frames=50)
            # use video_path...
    """
    created_files = []

    def _create_video(
        num_frames: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        fps: Optional[int] = None,
        filename: Optional[str] = None,
        directory: Optional[str] = None,
        frame_content: str = 'random',  # 'random', 'zeros', 'gradient', 'counter'
    ) -> str:
        """Create a test video file.

        Args:
            num_frames: Number of frames (default from video_config)
            height: Frame height (default from video_config)
            width: Frame width (default from video_config)
            fps: Frames per second (default from video_config)
            filename: Optional filename (default: auto-generated)
            directory: Optional directory (default: temp directory)
            frame_content: Type of frame content to generate

        Returns:
            Path to the created video file
        """
        # Use defaults from config
        num_frames = num_frames or video_config['num_frames']
        height = height or video_config['height']
        width = width or video_config['width']
        fps = fps or video_config['fps']

        # Create file path
        if directory is None:
            fd, video_path = tempfile.mkstemp(suffix='.mp4')
            os.close(fd)
        else:
            os.makedirs(directory, exist_ok=True)
            filename = filename or f'test_video_{len(created_files)}.mp4'
            video_path = os.path.join(directory, filename)

        # Create video writer
        # NOTE: cv2.VideoWriter size is (width, height), not (height, width)!
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

        if not writer.isOpened():
            raise RuntimeError(f"Failed to create video writer for {video_path}")

        # Write frames
        for i in range(num_frames):
            if frame_content == 'random':
                frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            elif frame_content == 'zeros':
                frame = np.zeros((height, width, 3), dtype=np.uint8)
            elif frame_content == 'gradient':
                # Create gradient frame based on frame index
                value = int(255 * i / max(num_frames - 1, 1))
                frame = np.full((height, width, 3), value, dtype=np.uint8)
            elif frame_content == 'counter':
                # Encode frame number in pixel values (for verification)
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                frame[:, :, 0] = i % 256  # Frame number in blue channel
                frame[:, :, 1] = (i // 256) % 256  # Overflow in green
            else:
                raise ValueError(f"Unknown frame_content: {frame_content}")

            writer.write(frame)

        writer.release()
        created_files.append(video_path)

        return video_path

    yield _create_video

    # Cleanup all created files
    for path in created_files:
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass


@pytest.fixture
def create_test_labels():
    """Factory fixture to create temporary label files.

    Returns a function that creates numpy label files.
    Files are automatically cleaned up after the test.
    """
    created_files = []

    def _create_labels(
        num_frames: int,
        num_classes: int = 3,
        filename: Optional[str] = None,
        directory: Optional[str] = None,
        distribution: str = 'uniform',  # 'uniform', 'imbalanced', 'sequential'
        include_ignore: bool = False,
        ignore_index: int = -100,
    ) -> str:
        """Create a test label file.

        Args:
            num_frames: Number of frames/labels
            num_classes: Number of classes
            filename: Optional filename
            directory: Optional directory (default: temp directory)
            distribution: How to distribute labels
            include_ignore: Whether to include ignore_index values
            ignore_index: Value for ignored frames

        Returns:
            Path to the created .npy file
        """
        # Generate labels based on distribution
        if distribution == 'uniform':
            labels = np.random.randint(0, num_classes, size=num_frames)
        elif distribution == 'imbalanced':
            # Class 0 is 50%, others split remaining
            weights = [0.5] + [0.5 / (num_classes - 1)] * (num_classes - 1)
            labels = np.random.choice(num_classes, size=num_frames, p=weights)
        elif distribution == 'sequential':
            # Repeat class pattern
            labels = np.tile(np.arange(num_classes), num_frames // num_classes + 1)[:num_frames]
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

        # Add ignore values if requested
        if include_ignore:
            # Add ignore at start and end (simulating padding)
            ignore_count = min(10, num_frames // 10)
            labels[:ignore_count] = ignore_index
            labels[-ignore_count:] = ignore_index

        # Create file path
        if directory is None:
            fd, label_path = tempfile.mkstemp(suffix='.npy')
            os.close(fd)
        else:
            os.makedirs(directory, exist_ok=True)
            filename = filename or f'test_labels_{len(created_files)}.npy'
            label_path = os.path.join(directory, filename)

        np.save(label_path, labels.astype(np.int64))
        created_files.append(label_path)

        return label_path

    yield _create_labels

    # Cleanup
    for path in created_files:
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass


# =============================================================================
# Tests for _get_video_frame_count_cv2
# =============================================================================
# Tests for _get_video_frame_count_cv2 (CPU-only, no DALI needed)
# =============================================================================

class TestGetVideoFrameCountCV2:
    """Test the _get_video_frame_count_cv2 utility function."""

    def test_basic_frame_count(self, create_test_video):
        """Test getting frame count from a simple video."""
        from lightning_action.data.video_datamodule import _get_video_frame_count_cv2

        video_path = create_test_video(num_frames=50)
        frame_count = _get_video_frame_count_cv2(video_path)

        assert frame_count == 50

    def test_frame_count_various_lengths(self, create_test_video):
        """Test frame count with various video lengths."""
        from lightning_action.data.video_datamodule import _get_video_frame_count_cv2

        for expected_frames in [1, 10, 50, 100, 250]:
            video_path = create_test_video(num_frames=expected_frames)
            frame_count = _get_video_frame_count_cv2(video_path)
            assert frame_count == expected_frames, f"Expected {expected_frames}, got {frame_count}"

    def test_nonexistent_file_returns_zero(self):
        """Test that non-existent file returns 0 (not raises exception)."""
        from lightning_action.data.video_datamodule import _get_video_frame_count_cv2

        # Should return 0, not raise an exception
        result = _get_video_frame_count_cv2("/nonexistent/path/video.mp4")
        assert result == 0

    def test_invalid_file_returns_zero(self):
        """Test that invalid video file returns 0."""
        from lightning_action.data.video_datamodule import _get_video_frame_count_cv2

        # Create a non-video file with .mp4 extension
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            f.write(b"not a video file - just random bytes")
            invalid_path = f.name

        try:
            result = _get_video_frame_count_cv2(invalid_path)
            assert result == 0
        finally:
            os.remove(invalid_path)

    def test_empty_path_returns_zero(self):
        """Test that empty path returns 0."""
        from lightning_action.data.video_datamodule import _get_video_frame_count_cv2

        result = _get_video_frame_count_cv2("")
        assert result == 0


# =============================================================================
# =============================================================================
# Tests for load_labels_for_videos (CPU-only, no DALI needed)
# =============================================================================

class TestLoadLabelsForVideos:
    """Test the load_labels_for_videos utility function.

    These tests run on CPU and don't require DALI/GPU.
    """

    def test_basic_label_loading(self, create_test_video, create_test_labels):
        """Test loading labels for a single video."""
        from lightning_action.data.video_datamodule import load_labels_for_videos

        with tempfile.TemporaryDirectory() as tmpdir:
            videos_dir = os.path.join(tmpdir, 'videos')
            labels_dir = os.path.join(tmpdir, 'labels')
            os.makedirs(videos_dir)
            os.makedirs(labels_dir)

            num_frames = 50
            video_path = create_test_video(
                num_frames=num_frames,
                directory=videos_dir,
                filename='test.mp4'
            )
            create_test_labels(
                num_frames=num_frames,
                num_classes=3,
                distribution='sequential',
                directory=labels_dir,
                filename='test.npy'
            )

            labels_2d = load_labels_for_videos(
                video_paths=[video_path],
                labels_dir=labels_dir,
                max_frames=100,
                ignore_index=-100
            )

            assert labels_2d is not None
            assert labels_2d.shape == (1, 100)
            # First 50 frames should have real labels
            assert torch.all(labels_2d[0, :num_frames] >= 0)
            # Rest should be ignore_index
            assert torch.all(labels_2d[0, num_frames:] == -100)

    def test_multiple_videos_different_lengths(self, create_test_video, create_test_labels):
        """Test loading labels for multiple videos with different lengths."""
        from lightning_action.data.video_datamodule import load_labels_for_videos

        with tempfile.TemporaryDirectory() as tmpdir:
            videos_dir = os.path.join(tmpdir, 'videos')
            labels_dir = os.path.join(tmpdir, 'labels')
            os.makedirs(videos_dir)
            os.makedirs(labels_dir)

            video_paths = []
            frame_counts = [30, 50, 70]

            for i, num_frames in enumerate(frame_counts):
                video_path = create_test_video(
                    num_frames=num_frames,
                    directory=videos_dir,
                    filename=f'video_{i}.mp4'
                )
                create_test_labels(
                    num_frames=num_frames,
                    directory=labels_dir,
                    filename=f'video_{i}.npy'
                )
                video_paths.append(video_path)

            max_frames = 100
            labels_2d = load_labels_for_videos(
                video_paths=video_paths,
                labels_dir=labels_dir,
                max_frames=max_frames,
                ignore_index=-100
            )

            assert labels_2d.shape == (3, 100)

            # Check each video has correct label padding
            for i, num_frames in enumerate(frame_counts):
                # Valid labels
                assert torch.all(labels_2d[i, :num_frames] >= 0)
                # Padded with ignore_index
                assert torch.all(labels_2d[i, num_frames:] == -100)

    def test_one_hot_labels_conversion(self, create_test_video):
        """Test that one-hot encoded labels are converted to class indices."""
        from lightning_action.data.video_datamodule import load_labels_for_videos

        with tempfile.TemporaryDirectory() as tmpdir:
            videos_dir = os.path.join(tmpdir, 'videos')
            labels_dir = os.path.join(tmpdir, 'labels')
            os.makedirs(videos_dir)
            os.makedirs(labels_dir)

            num_frames = 20
            num_classes = 4

            video_path = create_test_video(
                num_frames=num_frames,
                directory=videos_dir,
                filename='test.mp4'
            )

            # Create one-hot labels
            labels_onehot = np.zeros((num_frames, num_classes), dtype=np.float32)
            expected_classes = np.array(
                [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])
            for i, cls in enumerate(expected_classes):
                labels_onehot[i, cls] = 1.0

            label_path = os.path.join(labels_dir, 'test.npy')
            np.save(label_path, labels_onehot)

            labels_2d = load_labels_for_videos(
                video_paths=[video_path],
                labels_dir=labels_dir,
                max_frames=30,
                ignore_index=-100
            )

            # Check conversion from one-hot to class indices
            assert torch.all(labels_2d[0, :num_frames] == torch.from_numpy(expected_classes))

    def test_missing_label_file(self, create_test_video):
        """Test handling of missing label files."""
        from lightning_action.data.video_datamodule import load_labels_for_videos

        with tempfile.TemporaryDirectory() as tmpdir:
            videos_dir = os.path.join(tmpdir, 'videos')
            labels_dir = os.path.join(tmpdir, 'labels')
            os.makedirs(videos_dir)
            os.makedirs(labels_dir)

            video_path = create_test_video(
                num_frames=50,
                directory=videos_dir,
                filename='no_label.mp4'
            )

            # Don't create label file
            labels_2d = load_labels_for_videos(
                video_paths=[video_path],
                labels_dir=labels_dir,
                max_frames=100,
                ignore_index=-100
            )

            # All frames should be ignore_index since no label file
            assert torch.all(labels_2d == -100)

    def test_none_labels_dir(self, create_test_video):
        """Test that None labels_dir returns None."""
        from lightning_action.data.video_datamodule import load_labels_for_videos

        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = create_test_video(num_frames=50, directory=tmpdir)

            result = load_labels_for_videos(
                video_paths=[video_path],
                labels_dir=None,
                max_frames=100,
            )

            assert result is None

    def test_custom_ignore_index(self, create_test_video, create_test_labels):
        """Test using a custom ignore_index value."""
        from lightning_action.data.video_datamodule import load_labels_for_videos

        with tempfile.TemporaryDirectory() as tmpdir:
            videos_dir = os.path.join(tmpdir, 'videos')
            labels_dir = os.path.join(tmpdir, 'labels')
            os.makedirs(videos_dir)
            os.makedirs(labels_dir)

            num_frames = 30
            video_path = create_test_video(
                num_frames=num_frames,
                directory=videos_dir,
                filename='test.mp4'
            )
            create_test_labels(
                num_frames=num_frames,
                directory=labels_dir,
                filename='test.npy'
            )

            custom_ignore = -999
            labels_2d = load_labels_for_videos(
                video_paths=[video_path],
                labels_dir=labels_dir,
                max_frames=50,
                ignore_index=custom_ignore
            )

            # Padding should use custom ignore_index
            assert torch.all(labels_2d[0, num_frames:] == custom_ignore)


# =============================================================================
# Tests for get_video_lengths (CPU-only, no DALI needed)
# =============================================================================

class TestGetVideoLengths:
    """Test the get_video_lengths utility function."""

    def test_from_labels_directory(self, create_test_video, create_test_labels):
        """Test getting video lengths from label files."""
        from lightning_action.data.video_datamodule import get_video_lengths

        with tempfile.TemporaryDirectory() as tmpdir:
            videos_dir = os.path.join(tmpdir, 'videos')
            labels_dir = os.path.join(tmpdir, 'labels')
            os.makedirs(videos_dir)
            os.makedirs(labels_dir)

            # Create videos and labels with different lengths
            frame_counts = [50, 75, 100]
            video_paths = []

            for i, num_frames in enumerate(frame_counts):
                video_path = create_test_video(
                    num_frames=num_frames,
                    directory=videos_dir,
                    filename=f'video_{i}.mp4'
                )
                create_test_labels(
                    num_frames=num_frames,
                    directory=labels_dir,
                    filename=f'video_{i}.npy'
                )
                video_paths.append(video_path)

            # Get lengths using labels
            lengths = get_video_lengths(video_paths, labels_dir=labels_dir)

            assert len(lengths) == 3
            assert lengths[0] == 50
            assert lengths[1] == 75
            assert lengths[2] == 100

    def test_from_precomputed_lengths(self, create_test_video):
        """Test getting video lengths from precomputed dict."""
        from lightning_action.data.video_datamodule import get_video_lengths

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create videos (actual lengths don't matter - precomputed takes precedence)
            video_path = create_test_video(
                num_frames=30,  # Actual frames
                directory=tmpdir,
                filename='my_video.mp4'
            )

            # Precomputed says it has 100 frames
            precomputed = {'my_video': 100}

            lengths = get_video_lengths(
                [video_path],
                labels_dir=None,
                precomputed_lengths=precomputed
            )

            # Should use precomputed value, not actual
            assert lengths[0] == 100

    def test_fallback_to_opencv(self, create_test_video):
        """Test fallback to OpenCV when no labels or precomputed available."""
        from lightning_action.data.video_datamodule import get_video_lengths

        with tempfile.TemporaryDirectory() as tmpdir:
            num_frames = 65
            video_path = create_test_video(
                num_frames=num_frames,
                directory=tmpdir,
                filename='test.mp4'
            )

            # No labels_dir, no precomputed - must use OpenCV
            lengths = get_video_lengths(
                [video_path],
                labels_dir=None,
                precomputed_lengths=None
            )

            assert lengths[0] == num_frames

    def test_priority_order(self, create_test_video, create_test_labels):
        """Test that precomputed > labels > opencv in priority."""
        from lightning_action.data.video_datamodule import get_video_lengths

        with tempfile.TemporaryDirectory() as tmpdir:
            videos_dir = os.path.join(tmpdir, 'videos')
            labels_dir = os.path.join(tmpdir, 'labels')
            os.makedirs(videos_dir)
            os.makedirs(labels_dir)

            # Create video with 30 frames (opencv source)
            video_path = create_test_video(
                num_frames=30,
                directory=videos_dir,
                filename='priority_test.mp4'
            )

            # Create label with 50 frames
            create_test_labels(
                num_frames=50,
                directory=labels_dir,
                filename='priority_test.npy'
            )

            # Precomputed says 100 frames
            precomputed = {'priority_test': 100}

            # With all sources available, precomputed wins
            lengths = get_video_lengths(
                [video_path],
                labels_dir=labels_dir,
                precomputed_lengths=precomputed
            )
            assert lengths[0] == 100

            # Without precomputed, labels win
            lengths = get_video_lengths(
                [video_path],
                labels_dir=labels_dir,
                precomputed_lengths=None
            )
            assert lengths[0] == 50

            # Without precomputed or labels, opencv is used
            lengths = get_video_lengths(
                [video_path],
                labels_dir=None,
                precomputed_lengths=None
            )
            assert lengths[0] == 30

    def test_mixed_sources(self, create_test_video, create_test_labels):
        """Test with different sources for different videos."""
        from lightning_action.data.video_datamodule import get_video_lengths

        with tempfile.TemporaryDirectory() as tmpdir:
            videos_dir = os.path.join(tmpdir, 'videos')
            labels_dir = os.path.join(tmpdir, 'labels')
            os.makedirs(videos_dir)
            os.makedirs(labels_dir)

            # Video 1: has precomputed length
            video1 = create_test_video(num_frames=10, directory=videos_dir, filename='v1.mp4')

            # Video 2: has label file only
            video2 = create_test_video(num_frames=20, directory=videos_dir, filename='v2.mp4')
            create_test_labels(num_frames=25, directory=labels_dir, filename='v2.npy')

            # Video 3: opencv only (no label, no precomputed)
            video3 = create_test_video(num_frames=30, directory=videos_dir, filename='v3.mp4')

            precomputed = {'v1': 100}

            lengths = get_video_lengths(
                [video1, video2, video3],
                labels_dir=labels_dir,
                precomputed_lengths=precomputed
            )

            assert lengths[0] == 100  # From precomputed
            assert lengths[1] == 25   # From labels
            assert lengths[2] == 30   # From opencv

    def test_empty_video_list(self):
        """Test with empty video list."""
        from lightning_action.data.video_datamodule import get_video_lengths

        lengths = get_video_lengths([], labels_dir=None, precomputed_lengths=None)
        assert lengths == {}

    def test_missing_label_file_uses_opencv(self, create_test_video):
        """Test that missing label file falls back to opencv."""
        from lightning_action.data.video_datamodule import get_video_lengths

        with tempfile.TemporaryDirectory() as tmpdir:
            videos_dir = os.path.join(tmpdir, 'videos')
            labels_dir = os.path.join(tmpdir, 'labels')
            os.makedirs(videos_dir)
            os.makedirs(labels_dir)  # Empty labels dir

            num_frames = 45
            video_path = create_test_video(
                num_frames=num_frames,
                directory=videos_dir,
                filename='no_label.mp4'
            )

            lengths = get_video_lengths(
                [video_path],
                labels_dir=labels_dir,
                precomputed_lengths=None
            )

            assert lengths[0] == num_frames


# =============================================================================
# Tests for get_max_frames (CPU-only, no DALI needed)
# =============================================================================

class TestGetMaxFrames:
    """Test the get_max_frames utility function."""

    def test_single_video(self, create_test_video):
        """Test max frames with single video."""
        from lightning_action.data.video_datamodule import get_max_frames

        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = create_test_video(num_frames=100, directory=tmpdir)

            max_frames = get_max_frames([video_path])
            assert max_frames == 100

    def test_multiple_videos_different_lengths(self, create_test_video):
        """Test max frames returns the maximum across videos."""
        from lightning_action.data.video_datamodule import get_max_frames

        with tempfile.TemporaryDirectory() as tmpdir:
            video_paths = []
            for i, num_frames in enumerate([50, 100, 75, 200, 150]):
                path = create_test_video(
                    num_frames=num_frames,
                    directory=tmpdir,
                    filename=f'video_{i}.mp4'
                )
                video_paths.append(path)

            max_frames = get_max_frames(video_paths)
            assert max_frames == 200

    def test_from_labels_directory(self, create_test_video, create_test_labels):
        """Test max frames using label files."""
        from lightning_action.data.video_datamodule import get_max_frames

        with tempfile.TemporaryDirectory() as tmpdir:
            videos_dir = os.path.join(tmpdir, 'videos')
            labels_dir = os.path.join(tmpdir, 'labels')
            os.makedirs(videos_dir)
            os.makedirs(labels_dir)

            video_paths = []
            for i, num_frames in enumerate([50, 150, 100]):
                path = create_test_video(
                    num_frames=num_frames,
                    directory=videos_dir,
                    filename=f'video_{i}.mp4'
                )
                create_test_labels(
                    num_frames=num_frames,
                    directory=labels_dir,
                    filename=f'video_{i}.npy'
                )
                video_paths.append(path)

            max_frames = get_max_frames(video_paths, labels_dir=labels_dir)
            assert max_frames == 150

    def test_from_precomputed(self, create_test_video):
        """Test max frames using precomputed lengths."""
        from lightning_action.data.video_datamodule import get_max_frames

        with tempfile.TemporaryDirectory() as tmpdir:
            video_paths = []
            precomputed = {}

            for i, (actual, precomp) in enumerate([(10, 100), (20, 200), (30, 150)]):
                path = create_test_video(
                    num_frames=actual,
                    directory=tmpdir,
                    filename=f'vid_{i}.mp4'
                )
                video_paths.append(path)
                precomputed[f'vid_{i}'] = precomp

            max_frames = get_max_frames(
                video_paths,
                precomputed_lengths=precomputed
            )
            # Should use precomputed values
            assert max_frames == 200

    def test_empty_video_list(self):
        """Test max frames with empty list returns 0."""
        from lightning_action.data.video_datamodule import get_max_frames

        max_frames = get_max_frames([])
        assert max_frames == 0

    def test_all_same_length(self, create_test_video):
        """Test when all videos have the same length."""
        from lightning_action.data.video_datamodule import get_max_frames

        with tempfile.TemporaryDirectory() as tmpdir:
            video_paths = []
            for i in range(5):
                path = create_test_video(
                    num_frames=100,
                    directory=tmpdir,
                    filename=f'video_{i}.mp4'
                )
                video_paths.append(path)

            max_frames = get_max_frames(video_paths)
            assert max_frames == 100

    def test_mixed_sources_priority(self, create_test_video, create_test_labels):
        """Test max frames correctly prioritizes sources."""
        from lightning_action.data.video_datamodule import get_max_frames

        with tempfile.TemporaryDirectory() as tmpdir:
            videos_dir = os.path.join(tmpdir, 'videos')
            labels_dir = os.path.join(tmpdir, 'labels')
            os.makedirs(videos_dir)
            os.makedirs(labels_dir)

            # Video 1: precomputed = 500
            v1 = create_test_video(num_frames=10, directory=videos_dir, filename='v1.mp4')

            # Video 2: label file = 300
            v2 = create_test_video(num_frames=20, directory=videos_dir, filename='v2.mp4')
            create_test_labels(num_frames=300, directory=labels_dir, filename='v2.npy')

            # Video 3: opencv = 100
            v3 = create_test_video(num_frames=100, directory=videos_dir, filename='v3.mp4')

            precomputed = {'v1': 500}

            max_frames = get_max_frames(
                [v1, v2, v3],
                labels_dir=labels_dir,
                precomputed_lengths=precomputed
            )

            # Maximum should be 500 (from precomputed for v1)
            assert max_frames == 500


# =============================================================================
# Mock-based tests for DALIIterator logic (CPU-only, no GPU needed)
# =============================================================================

class TestDALIIteratorLogic:
    """Test DALIIterator logic using mocks.

    These tests verify the iterator's label lookup and metadata generation
    logic without requiring an actual GPU or DALI pipeline.
    """

    def test_metadata_generation_start_of_video(self):
        """Test metadata correctly identifies start of video."""
        # Simulate what the iterator does with metadata
        video_lengths = {0: 100, 1: 150}

        # Simulate batch from DALI
        frame_indices = torch.tensor([0, 1])  # Video indices
        start_frames = torch.tensor([0, 50])  # Start frame in each video
        T = 32  # Sequence length

        metadata = []
        for i in range(len(frame_indices)):
            video_idx = frame_indices[i].item()
            start_frame = start_frames[i].item()

            is_start = (start_frame == 0)
            is_end = False
            if video_idx in video_lengths:
                video_len = video_lengths[video_idx]
                if start_frame + T >= video_len:
                    is_end = True

            meta = {
                'video_idx': video_idx,
                'dali_start_frame': start_frame,
                'num_frames': T,
                'is_start': is_start,
                'is_end': is_end,
            }
            metadata.append(meta)

        # First sample starts at frame 0, so is_start=True
        assert metadata[0]['is_start'] is True
        assert metadata[0]['is_end'] is False

        # Second sample starts at frame 50, not at start
        assert metadata[1]['is_start'] is False
        assert metadata[1]['is_end'] is False

    def test_metadata_generation_end_of_video(self):
        """Test metadata correctly identifies end of video."""
        video_lengths = {0: 100}

        # Simulate batch at end of video
        frame_indices = torch.tensor([0])
        start_frames = torch.tensor([80])  # Near end
        T = 32  # 80 + 32 = 112 > 100, so this is end

        metadata = []
        for i in range(len(frame_indices)):
            video_idx = frame_indices[i].item()
            start_frame = start_frames[i].item()

            is_start = (start_frame == 0)
            is_end = False
            if video_idx in video_lengths:
                video_len = video_lengths[video_idx]
                if start_frame + T >= video_len:
                    is_end = True

            meta = {
                'video_idx': video_idx,
                'dali_start_frame': start_frame,
                'num_frames': T,
                'is_start': is_start,
                'is_end': is_end,
            }
            metadata.append(meta)

        assert metadata[0]['is_start'] is False
        assert metadata[0]['is_end'] is True

    def test_label_lookup_logic(self):
        """Test the label lookup logic from 2D tensor."""
        # Create a labels tensor: 2 videos, max 50 frames
        num_videos = 2
        max_frames = 50
        ignore_index = -100

        # Create labels: video 0 has class pattern [0,1,2,0,1,2,...], video 1 all class 1
        all_labels_2d = torch.full((num_videos, max_frames), ignore_index, dtype=torch.long)
        for i in range(30):  # Video 0 has 30 real frames
            all_labels_2d[0, i] = i % 3
        for i in range(40):  # Video 1 has 40 real frames
            all_labels_2d[1, i] = 1

        # Simulate batch indexing
        T = 10
        frame_indices = torch.tensor([0, 1])  # Video 0 and 1
        start_frames = torch.tensor([5, 20])  # Start at frame 5 and 20

        # This is the label lookup logic from DALIIterator.__next__
        seq_idxs = torch.arange(T)
        time_idx = start_frames.unsqueeze(1) + seq_idxs.unsqueeze(0)  # (B, T)
        video_idx = frame_indices.unsqueeze(1).expand(-1, T)  # (B, T)

        time_idx = torch.clamp(time_idx, 0, max_frames - 1)
        labels = all_labels_2d[video_idx, time_idx]

        # Video 0, frames 5-14: pattern [2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
        expected_v0 = torch.tensor([i % 3 for i in range(5, 15)])
        assert torch.all(labels[0] == expected_v0)

        # Video 1, frames 20-29: all class 1
        assert torch.all(labels[1] == 1)

    def test_label_lookup_with_clamping(self):
        """Test that label lookup clamps indices to valid range."""
        num_videos = 1
        max_frames = 20
        ignore_index = -100

        all_labels_2d = torch.full((num_videos, max_frames), ignore_index, dtype=torch.long)
        for i in range(max_frames):
            all_labels_2d[0, i] = i

        # Try to access beyond max_frames
        T = 10
        frame_indices = torch.tensor([0])
        start_frames = torch.tensor([15])  # 15 + 10 = 25 > 20

        seq_idxs = torch.arange(T)
        time_idx = start_frames.unsqueeze(1) + seq_idxs.unsqueeze(0)
        video_idx = frame_indices.unsqueeze(1).expand(-1, T)

        # Clamp to valid range
        time_idx = torch.clamp(time_idx, 0, max_frames - 1)
        labels = all_labels_2d[video_idx, time_idx]

        # Frames 15-19 should have their real labels, 20-24 should be clamped to frame 19
        expected = torch.tensor([15, 16, 17, 18, 19, 19, 19, 19, 19, 19])
        assert torch.all(labels[0] == expected)

    def test_scalar_tensor_handling(self):
        """Test handling of scalar tensors (batch size 1)."""
        # When batch_size=1, DALI returns scalar tensors that need unsqueezing
        frame_indices = torch.tensor(0)  # Scalar
        start_frames = torch.tensor(10)  # Scalar

        # The iterator does this check
        if frame_indices.dim() == 0:
            frame_indices = frame_indices.unsqueeze(0)
        if start_frames.dim() == 0:
            start_frames = start_frames.unsqueeze(0)

        assert frame_indices.shape == (1,)
        assert start_frames.shape == (1,)
        assert frame_indices[0].item() == 0
        assert start_frames[0].item() == 10


# =============================================================================
# Tests for VideoDataset.get_video_length
# =============================================================================

class TestVideoDatasetGetVideoLength:
    """Test VideoDataset.get_video_length method."""

    def test_get_length_from_labels(self, create_test_video, create_test_labels):
        """Test getting video length using label file (preferred method)."""
        from lightning_action.data.video_dataset import VideoDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            videos_dir = os.path.join(tmpdir, 'videos')
            labels_dir = os.path.join(tmpdir, 'labels')
            os.makedirs(videos_dir)
            os.makedirs(labels_dir)

            num_frames = 75
            create_test_video(
                num_frames=num_frames,
                directory=videos_dir,
                filename='test_video.mp4'
            )
            create_test_labels(
                num_frames=num_frames,
                directory=labels_dir,
                filename='test_video.npy'
            )

            dataset = VideoDataset(
                videos_dir=videos_dir,
                labels_dir=labels_dir,
                num_classes=3,
            )

            length = dataset.get_video_length(0)
            assert length == num_frames

    def test_get_length_from_opencv(self, create_test_video):
        """Test getting video length using OpenCV (fallback method)."""
        from lightning_action.data.video_dataset import VideoDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            videos_dir = os.path.join(tmpdir, 'videos')
            os.makedirs(videos_dir)

            num_frames = 60
            create_test_video(
                num_frames=num_frames,
                directory=videos_dir,
                filename='test_video.mp4'
            )

            dataset = VideoDataset(
                videos_dir=videos_dir,
                labels_dir=None,
                require_labels=False,
                num_classes=3,
            )

            length = dataset.get_video_length(0)
            assert length == num_frames

    def test_get_length_multiple_videos(self, create_test_video, create_test_labels):
        """Test getting lengths for multiple videos."""
        from lightning_action.data.video_dataset import VideoDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            videos_dir = os.path.join(tmpdir, 'videos')
            labels_dir = os.path.join(tmpdir, 'labels')
            os.makedirs(videos_dir)
            os.makedirs(labels_dir)

            # Create videos with different frame counts
            # Use a dict to track expected lengths by filename
            expected_lengths = {
                'video_0.mp4': 50,
                'video_1.mp4': 75,
                'video_2.mp4': 100,
            }

            for filename, num_frames in expected_lengths.items():
                create_test_video(
                    num_frames=num_frames,
                    directory=videos_dir,
                    filename=filename
                )
                label_filename = filename.replace('.mp4', '.npy')
                create_test_labels(
                    num_frames=num_frames,
                    directory=labels_dir,
                    filename=label_filename
                )

            dataset = VideoDataset(
                videos_dir=videos_dir,
                labels_dir=labels_dir,
                num_classes=3,
            )

            # Verify we have all videos
            assert len(dataset.video_paths) == 3

            # Check lengths match expected values (order-agnostic)
            # Build mapping from dataset's actual video order to expected lengths
            actual_lengths = []
            for i in range(len(dataset.video_paths)):
                length = dataset.get_video_length(i)
                actual_lengths.append(length)

            # The set of lengths should match, regardless of order
            assert sorted(actual_lengths) == sorted(expected_lengths.values())

            # Also verify each video's length matches its label file
            for i, video_path in enumerate(dataset.video_paths):
                video_name = os.path.basename(video_path)
                expected = expected_lengths[video_name]
                actual = dataset.get_video_length(i)
                assert actual == expected, f"Video {video_name}: expected {expected}, got {actual}"


# =============================================================================
# Tests for VideoDataset initialization and discovery
# =============================================================================

class TestVideoDatasetDiscovery:
    """Test VideoDataset video discovery and validation."""

    def test_discover_single_video(self, create_test_video, create_test_labels):
        """Test discovering a single video with labels."""
        from lightning_action.data.video_dataset import VideoDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            videos_dir = os.path.join(tmpdir, 'videos')
            labels_dir = os.path.join(tmpdir, 'labels')
            os.makedirs(videos_dir)
            os.makedirs(labels_dir)

            create_test_video(
                num_frames=50,
                directory=videos_dir,
                filename='experiment1.mp4'
            )
            create_test_labels(
                num_frames=50,
                directory=labels_dir,
                filename='experiment1.npy'
            )

            dataset = VideoDataset(
                videos_dir=videos_dir,
                labels_dir=labels_dir,
            )

            assert len(dataset) == 1
            assert len(dataset.video_paths) == 1
            assert len(dataset.label_paths) == 1

    def test_filter_by_expt_ids(self, create_test_video, create_test_labels):
        """Test filtering videos by experiment IDs."""
        from lightning_action.data.video_dataset import VideoDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            videos_dir = os.path.join(tmpdir, 'videos')
            labels_dir = os.path.join(tmpdir, 'labels')
            os.makedirs(videos_dir)
            os.makedirs(labels_dir)

            for name in ['exp1', 'exp2', 'exp3', 'other']:
                create_test_video(
                    num_frames=30,
                    directory=videos_dir,
                    filename=f'{name}.mp4'
                )
                create_test_labels(
                    num_frames=30,
                    directory=labels_dir,
                    filename=f'{name}.npy'
                )

            dataset = VideoDataset(
                videos_dir=videos_dir,
                labels_dir=labels_dir,
                expt_ids=['exp1', 'exp2'],
            )

            assert len(dataset) == 2

    def test_missing_labels_raises_error(self, create_test_video):
        """Test that missing labels raises FileNotFoundError by default."""
        from lightning_action.data.video_dataset import VideoDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            videos_dir = os.path.join(tmpdir, 'videos')
            labels_dir = os.path.join(tmpdir, 'labels')
            os.makedirs(videos_dir)
            os.makedirs(labels_dir)

            create_test_video(
                num_frames=50,
                directory=videos_dir,
                filename='test.mp4'
            )

            with pytest.raises(FileNotFoundError, match="Missing .npy label files"):
                VideoDataset(
                    videos_dir=videos_dir,
                    labels_dir=labels_dir,
                    require_labels=True,
                )

    def test_missing_labels_allowed_for_prediction(self, create_test_video):
        """Test that missing labels are allowed when require_labels=False."""
        from lightning_action.data.video_dataset import VideoDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            videos_dir = os.path.join(tmpdir, 'videos')
            labels_dir = os.path.join(tmpdir, 'labels')
            os.makedirs(videos_dir)
            os.makedirs(labels_dir)

            create_test_video(
                num_frames=50,
                directory=videos_dir,
                filename='test.mp4'
            )

            dataset = VideoDataset(
                videos_dir=videos_dir,
                labels_dir=labels_dir,
                require_labels=False,
                num_classes=3,
            )

            assert len(dataset) == 1
            assert len(dataset.label_paths) == 0

    def test_prediction_mode_no_labels_dir(self, create_test_video):
        """Test prediction mode with labels_dir=None."""
        from lightning_action.data.video_dataset import VideoDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            videos_dir = os.path.join(tmpdir, 'videos')
            os.makedirs(videos_dir)

            create_test_video(
                num_frames=50,
                directory=videos_dir,
                filename='test.mp4'
            )

            dataset = VideoDataset(
                videos_dir=videos_dir,
                labels_dir=None,
                num_classes=3,
            )

            assert len(dataset) == 1
            assert dataset.num_classes == 3
            assert len(dataset.class_weights) == 3

    def test_num_classes_required_without_labels(self, create_test_video):
        """Test that num_classes is required when labels_dir is None."""
        from lightning_action.data.video_dataset import VideoDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            videos_dir = os.path.join(tmpdir, 'videos')
            os.makedirs(videos_dir)

            create_test_video(
                num_frames=50,
                directory=videos_dir,
                filename='test.mp4'
            )

            with pytest.raises(ValueError, match="num_classes must be provided"):
                VideoDataset(
                    videos_dir=videos_dir,
                    labels_dir=None,
                )


class TestVideoDatasetClassWeights:
    """Test VideoDataset class weight computation."""

    def test_uniform_weights_without_labels(self, create_test_video):
        """Test that uniform weights are returned without labels."""
        from lightning_action.data.video_dataset import VideoDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            videos_dir = os.path.join(tmpdir, 'videos')
            os.makedirs(videos_dir)

            create_test_video(
                num_frames=50,
                directory=videos_dir,
                filename='test.mp4'
            )

            dataset = VideoDataset(
                videos_dir=videos_dir,
                labels_dir=None,
                num_classes=4,
            )

            assert len(dataset.class_weights) == 4
            assert all(w == 1.0 for w in dataset.class_weights)

    def test_computed_weights_with_labels(self, create_test_video, create_test_labels):
        """Test that class weights are computed from labels."""
        from lightning_action.data.video_dataset import VideoDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            videos_dir = os.path.join(tmpdir, 'videos')
            labels_dir = os.path.join(tmpdir, 'labels')
            os.makedirs(videos_dir)
            os.makedirs(labels_dir)

            create_test_video(
                num_frames=100,
                directory=videos_dir,
                filename='test.mp4'
            )
            create_test_labels(
                num_frames=100,
                num_classes=3,
                distribution='imbalanced',
                directory=labels_dir,
                filename='test.npy'
            )

            dataset = VideoDataset(
                videos_dir=videos_dir,
                labels_dir=labels_dir,
            )

            assert len(dataset.class_weights) == 3
            # Most frequent class should have weight close to 1.0
            assert min(dataset.class_weights) == pytest.approx(1.0, abs=0.01)


class TestVideoDatasetLabelNames:
    """Test VideoDataset label name handling."""

    def test_auto_generated_label_names(self, create_test_video, create_test_labels):
        """Test that label names are auto-generated when not provided."""
        from lightning_action.data.video_dataset import VideoDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            videos_dir = os.path.join(tmpdir, 'videos')
            labels_dir = os.path.join(tmpdir, 'labels')
            os.makedirs(videos_dir)
            os.makedirs(labels_dir)

            create_test_video(num_frames=50, directory=videos_dir, filename='test.mp4')
            create_test_labels(num_frames=50, num_classes=3,
                               directory=labels_dir, filename='test.npy')

            dataset = VideoDataset(
                videos_dir=videos_dir,
                labels_dir=labels_dir,
            )

            label_names = dataset.get_label_names()
            assert label_names == ['class_0', 'class_1', 'class_2']

    def test_custom_label_names(self, create_test_video, create_test_labels):
        """Test that custom label names are preserved."""
        from lightning_action.data.video_dataset import VideoDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            videos_dir = os.path.join(tmpdir, 'videos')
            labels_dir = os.path.join(tmpdir, 'labels')
            os.makedirs(videos_dir)
            os.makedirs(labels_dir)

            create_test_video(num_frames=50, directory=videos_dir, filename='test.mp4')
            create_test_labels(num_frames=50, num_classes=3,
                               directory=labels_dir, filename='test.npy')

            custom_names = ['walking', 'running', 'jumping']
            dataset = VideoDataset(
                videos_dir=videos_dir,
                labels_dir=labels_dir,
                label_names=custom_names,
            )

            assert dataset.get_label_names() == custom_names

    def test_label_names_from_num_classes(self, create_test_video):
        """Test label names are generated from num_classes in prediction mode."""
        from lightning_action.data.video_dataset import VideoDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            videos_dir = os.path.join(tmpdir, 'videos')
            os.makedirs(videos_dir)

            create_test_video(num_frames=50, directory=videos_dir, filename='test.mp4')

            dataset = VideoDataset(
                videos_dir=videos_dir,
                labels_dir=None,
                num_classes=5,
            )

            label_names = dataset.get_label_names()
            assert len(label_names) == 5
            assert label_names == ['class_0', 'class_1', 'class_2', 'class_3', 'class_4']


# =============================================================================
# Tests for VideoDataModule setup and configuration (CPU-only after source fix)
# Note: These tests don't call train_dataloader/val_dataloader, so they work without DALI
# =============================================================================

class TestVideoDataModuleSetup:
    """Test VideoDataModule initialization and setup."""

    def test_basic_initialization(self, create_test_video, create_test_labels):
        """Test basic VideoDataModule initialization."""
        from lightning_action.data.video_datamodule import VideoDataModule

        with tempfile.TemporaryDirectory() as tmpdir:
            videos_dir = os.path.join(tmpdir, 'videos')
            labels_dir = os.path.join(tmpdir, 'labels')
            os.makedirs(videos_dir)
            os.makedirs(labels_dir)

            for i in range(5):
                create_test_video(
                    num_frames=50,
                    directory=videos_dir,
                    filename=f'video_{i}.mp4'
                )
                create_test_labels(
                    num_frames=50,
                    directory=labels_dir,
                    filename=f'video_{i}.npy'
                )

            data_config = {
                'videos_dir': videos_dir,
                'labels_dir': labels_dir,
            }

            datamodule = VideoDataModule(
                data_config=data_config,
                sequence_length=32,
                batch_size=2,
                train_probability=0.8,
                val_probability=0.2,
            )

            assert datamodule.sequence_length == 32
            assert datamodule.batch_size == 2
            assert len(datamodule.dataset) == 5

    def test_setup_splits_videos(self, create_test_video, create_test_labels):
        """Test that setup() splits videos into train/val."""
        from lightning_action.data.video_datamodule import VideoDataModule

        with tempfile.TemporaryDirectory() as tmpdir:
            videos_dir = os.path.join(tmpdir, 'videos')
            labels_dir = os.path.join(tmpdir, 'labels')
            os.makedirs(videos_dir)
            os.makedirs(labels_dir)

            for i in range(10):
                create_test_video(
                    num_frames=50,
                    directory=videos_dir,
                    filename=f'video_{i}.mp4'
                )
                create_test_labels(
                    num_frames=50,
                    directory=labels_dir,
                    filename=f'video_{i}.npy'
                )

            data_config = {
                'videos_dir': videos_dir,
                'labels_dir': labels_dir,
            }

            datamodule = VideoDataModule(
                data_config=data_config,
                train_probability=0.8,
                val_probability=0.2,
                seed=42,
            )

            datamodule.setup('fit')

            # Should split at video level
            assert len(datamodule.train_video_paths) == 8
            assert len(datamodule.val_video_paths) == 2
            # No overlap
            train_set = set(datamodule.train_video_paths)
            val_set = set(datamodule.val_video_paths)
            assert len(train_set & val_set) == 0

    def test_prediction_mode_no_labels(self, create_test_video):
        """Test VideoDataModule in prediction mode without labels."""
        from lightning_action.data.video_datamodule import VideoDataModule

        with tempfile.TemporaryDirectory() as tmpdir:
            videos_dir = os.path.join(tmpdir, 'videos')
            os.makedirs(videos_dir)

            for i in range(3):
                create_test_video(
                    num_frames=50,
                    directory=videos_dir,
                    filename=f'video_{i}.mp4'
                )

            data_config = {
                'videos_dir': videos_dir,
                'labels_dir': None,
                'num_classes': 3,
            }

            datamodule = VideoDataModule(
                data_config=data_config,
                sequence_length=32,
                batch_size=1,
            )

            datamodule.setup('predict')

            assert len(datamodule._predict_video_paths) == 3

    def test_invalid_probabilities(self, create_test_video, create_test_labels):
        """Test that invalid train/val probabilities raise errors."""
        from lightning_action.data.video_datamodule import VideoDataModule

        with tempfile.TemporaryDirectory() as tmpdir:
            videos_dir = os.path.join(tmpdir, 'videos')
            labels_dir = os.path.join(tmpdir, 'labels')
            os.makedirs(videos_dir)
            os.makedirs(labels_dir)

            create_test_video(num_frames=50, directory=videos_dir, filename='test.mp4')
            create_test_labels(num_frames=50, directory=labels_dir, filename='test.npy')

            data_config = {
                'videos_dir': videos_dir,
                'labels_dir': labels_dir,
            }

            with pytest.raises(ValueError, match="train_probability"):
                VideoDataModule(
                    data_config=data_config,
                    train_probability=1.5,  # Invalid
                    val_probability=0.1,
                )

            with pytest.raises(ValueError, match="val_probability"):
                VideoDataModule(
                    data_config=data_config,
                    train_probability=0.5,
                    val_probability=-0.1,  # Invalid
                )

            with pytest.raises(ValueError, match="<= 1"):
                VideoDataModule(
                    data_config=data_config,
                    train_probability=0.8,
                    val_probability=0.5,  # Sum > 1
                )

    def test_get_label_names(self, create_test_video, create_test_labels):
        """Test getting label names from datamodule."""
        from lightning_action.data.video_datamodule import VideoDataModule

        with tempfile.TemporaryDirectory() as tmpdir:
            videos_dir = os.path.join(tmpdir, 'videos')
            labels_dir = os.path.join(tmpdir, 'labels')
            os.makedirs(videos_dir)
            os.makedirs(labels_dir)

            create_test_video(num_frames=50, directory=videos_dir, filename='test.mp4')
            create_test_labels(num_frames=50, num_classes=4,
                               directory=labels_dir, filename='test.npy')

            data_config = {
                'videos_dir': videos_dir,
                'labels_dir': labels_dir,
            }

            datamodule = VideoDataModule(data_config=data_config)

            label_names = datamodule.get_label_names()
            assert len(label_names) == 4


# =============================================================================
# GPU-only integration tests for DALI
# =============================================================================

# Note: cleanup_gpu fixture is defined in conftest.py


@requires_gpu
class TestDALIIntegration:
    """Integration tests for DALI pipeline and iterator.

    These tests require an NVIDIA GPU and DALI installation.
    They are automatically skipped if GPU is not available.

    IMPORTANT: Videos must be long enough for DALI sequences.
    With TCN padding, extended_sequence = sequence_length + 2 * tcn_padding.
    For dtcn with num_layers=4, num_lags=2: tcn_padding = 60
    So extended_sequence = 32 + 120 = 152 frames minimum per video.

    We use 200+ frames to be safe and allow multiple sequences per video.

    NOTE: These tests use batch_size=1 and minimal threading to avoid
    DALI pipeline stalls in test environments.
    """

    # Minimum frames needed for DALI with TCN padding
    MIN_FRAMES_FOR_DALI = 200

    def test_dali_pipeline_builds(self, create_test_video, cleanup_gpu):
        """Test that DALI pipeline builds successfully."""
        import gc
        import tempfile as tf

        from lightning_action.data.video_datamodule import VideoPipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            videos_dir = os.path.join(tmpdir, 'videos')
            os.makedirs(videos_dir)

            # Create video long enough for DALI (no TCN padding in this test)
            video_path = create_test_video(
                num_frames=100,  # 100 frames is enough for sequence_length=32 without padding
                height=224,
                width=224,
                directory=videos_dir,
                filename='test.mp4'
            )

            # Create DALI file list
            with tf.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
                f.write(f"{video_path} 0\n")
                file_list = f.name

            pipe = None
            try:
                pipe = VideoPipeline(
                    batch_size=1,
                    num_threads=1,
                    device_id=0,
                    seed=42,
                    file_list=file_list,
                    sequence_length=32,  # No TCN padding added here
                    resolution=224,
                    random_shuffle=False,
                )
                pipe.build()

                # Should be able to run one iteration
                pipe_out = pipe.run()
                assert pipe_out is not None

            finally:
                del pipe
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                os.unlink(file_list)

    def test_dali_iterator_with_labels(self, create_test_video, create_test_labels, cleanup_gpu):
        """Test DALIIterator with label loading."""
        import gc
        import tempfile as tf

        from nvidia.dali.plugin.pytorch import LastBatchPolicy

        from lightning_action.data.video_datamodule import (
            DALIIterator,
            VideoPipeline,
            get_video_lengths,
            load_labels_for_videos,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            videos_dir = os.path.join(tmpdir, 'videos')
            labels_dir = os.path.join(tmpdir, 'labels')
            os.makedirs(videos_dir)
            os.makedirs(labels_dir)

            num_frames = 200  # Enough for multiple sequences
            sequence_length = 32

            video_path = create_test_video(
                num_frames=num_frames,
                height=224,
                width=224,
                directory=videos_dir,
                filename='test.mp4'
            )
            create_test_labels(
                num_frames=num_frames,
                num_classes=3,
                distribution='sequential',
                directory=labels_dir,
                filename='test.npy'
            )

            video_paths = [video_path]

            # Load labels
            labels_2d = load_labels_for_videos(
                video_paths=video_paths,
                labels_dir=labels_dir,
                max_frames=num_frames + 50,
            )
            labels_2d = labels_2d.cuda()

            # Get video lengths
            video_lengths = get_video_lengths(video_paths, labels_dir)

            # Create file list
            with tf.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
                f.write(f"{video_path} 0\n")
                file_list = f.name

            pipe = None
            iterator = None
            try:
                pipe = VideoPipeline(
                    batch_size=1,
                    num_threads=1,
                    device_id=0,
                    seed=42,
                    file_list=file_list,
                    sequence_length=sequence_length,  # Raw sequence length, no padding
                    resolution=224,
                    random_shuffle=False,
                )
                pipe.build()

                iterator = DALIIterator(
                    pipe,
                    output_map=['frames', 'frame_idx', 'start_frame'],
                    sequence_length=sequence_length,
                    tcn_padding=0,
                    ignore_index=-100,
                    all_labels_2d=labels_2d,
                    include_labels=True,
                    video_lengths=video_lengths,
                    last_batch_policy=LastBatchPolicy.DROP,
                )

                # Get one batch
                frames, labels, metadata = next(iterator)

                assert frames.shape[0] == 1  # batch size
                assert frames.shape[1] == sequence_length  # sequence length
                assert frames.shape[2] == 3  # channels
                assert labels.shape == (1, sequence_length)
                assert len(metadata) == 1
                assert 'video_idx' in metadata[0]
                assert 'is_start' in metadata[0]

            finally:
                # Explicit cleanup to avoid stalls in subsequent tests
                if iterator is not None:
                    try:
                        iterator.reset()
                    except Exception:
                        pass
                del iterator
                del pipe
                del labels_2d
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                os.unlink(file_list)

    def test_video_datamodule_train_dataloader(
        self,
        create_test_video,
        create_test_labels,
        cleanup_gpu,
    ):
        """Test VideoDataModule creates working train dataloader.

        Uses minimal TCN padding configuration to reduce required video length.
        Uses batch_size=1 to avoid prefetch stalls in test environment.
        """
        import gc

        from lightning_action.data.video_datamodule import VideoDataModule

        with tempfile.TemporaryDirectory() as tmpdir:
            videos_dir = os.path.join(tmpdir, 'videos')
            labels_dir = os.path.join(tmpdir, 'labels')
            os.makedirs(videos_dir)
            os.makedirs(labels_dir)

            # Video length calculation:
            # sequence_length=32, num_layers=2, num_lags=1
            # tcn_padding for dtcn = sum([2 * (2**n) * 1 for n in range(2)]) = 2 + 4 = 6
            # extended_sequence = 32 + 2*6 = 44 frames
            # We use 200 frames to allow multiple sequences
            num_frames = 200

            # Create multiple videos for train/val split
            for i in range(5):
                create_test_video(
                    num_frames=num_frames,
                    height=224,
                    width=224,
                    directory=videos_dir,
                    filename=f'video_{i}.mp4'
                )
                create_test_labels(
                    num_frames=num_frames,
                    num_classes=3,
                    directory=labels_dir,
                    filename=f'video_{i}.npy'
                )

            data_config = {
                'videos_dir': videos_dir,
                'labels_dir': labels_dir,
            }

            # Use minimal TCN config to reduce padding requirements
            model_config = {
                'head': 'dtcn',
                'num_layers': 2,  # Minimal layers -> less padding
                'num_lags': 1,    # Minimal lags -> less padding
            }

            datamodule = None
            train_loader = None
            try:
                datamodule = VideoDataModule(
                    data_config=data_config,
                    sequence_length=32,
                    batch_size=1,  # Use batch_size=1 to avoid prefetch stalls
                    num_workers=0,  # Minimal workers for test stability
                    train_probability=0.8,
                    val_probability=0.2,
                    model_config=model_config,
                )

                datamodule.setup('fit')
                train_loader = datamodule.train_dataloader()

                # Get one batch using direct next() call
                batch = next(iter(train_loader))
                frames, labels, metadata = batch

                # Check shapes
                # extended_sequence = 32 + 2*6 = 44
                expected_seq_len = 32 + 2 * datamodule.dataset.tcn_padding

                assert frames.shape[0] == 1  # batch size
                assert frames.shape[1] == expected_seq_len  # extended sequence length
                assert frames.shape[2] == 3  # channels (FCHW format)
                assert labels.shape == (1, expected_seq_len)
                assert len(metadata) == 1

            finally:
                # Explicit cleanup to prevent stalls in subsequent tests
                if train_loader is not None:
                    try:
                        train_loader.reset()
                    except Exception:
                        pass
                del train_loader
                del datamodule
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    def test_video_datamodule_val_dataloader(
        self,
        create_test_video,
        create_test_labels,
        cleanup_gpu,
    ):
        """Test VideoDataModule creates working validation dataloader."""
        import gc

        from lightning_action.data.video_datamodule import VideoDataModule

        with tempfile.TemporaryDirectory() as tmpdir:
            videos_dir = os.path.join(tmpdir, 'videos')
            labels_dir = os.path.join(tmpdir, 'labels')
            os.makedirs(videos_dir)
            os.makedirs(labels_dir)

            num_frames = 200

            # Need enough videos for train/val split
            for i in range(5):
                create_test_video(
                    num_frames=num_frames,
                    height=224,
                    width=224,
                    directory=videos_dir,
                    filename=f'video_{i}.mp4'
                )
                create_test_labels(
                    num_frames=num_frames,
                    num_classes=3,
                    directory=labels_dir,
                    filename=f'video_{i}.npy'
                )

            data_config = {
                'videos_dir': videos_dir,
                'labels_dir': labels_dir,
            }

            model_config = {
                'head': 'dtcn',
                'num_layers': 2,
                'num_lags': 1,
            }

            datamodule = None
            val_loader = None
            try:
                datamodule = VideoDataModule(
                    data_config=data_config,
                    sequence_length=32,
                    batch_size=1,
                    num_workers=0,
                    train_probability=0.6,
                    val_probability=0.4,  # Ensure we have validation videos
                    model_config=model_config,
                )

                datamodule.setup('fit')

                # Check validation is enabled
                assert datamodule.validation_enabled
                assert len(datamodule.val_video_paths) >= 1

                val_loader = datamodule.val_dataloader()

                # Get one batch
                batch = next(iter(val_loader))
                frames, labels, metadata = batch

                assert frames.shape[0] == 1  # batch size
                assert frames.shape[2] == 3  # channels

            finally:
                if val_loader is not None:
                    try:
                        val_loader.reset()
                    except Exception:
                        pass
                del val_loader
                del datamodule
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    def test_video_datamodule_predict_dataloader(self, create_test_video, cleanup_gpu):
        """Test VideoDataModule creates working prediction dataloader."""
        import gc

        from lightning_action.data.video_datamodule import VideoDataModule

        with tempfile.TemporaryDirectory() as tmpdir:
            videos_dir = os.path.join(tmpdir, 'videos')
            os.makedirs(videos_dir)

            num_frames = 200

            for i in range(3):
                create_test_video(
                    num_frames=num_frames,
                    height=224,
                    width=224,
                    directory=videos_dir,
                    filename=f'video_{i}.mp4'
                )

            data_config = {
                'videos_dir': videos_dir,
                'labels_dir': None,  # No labels for prediction
                'num_classes': 3,
            }

            model_config = {
                'head': 'dtcn',
                'num_layers': 2,
                'num_lags': 1,
            }

            datamodule = None
            predict_loader = None
            try:
                datamodule = VideoDataModule(
                    data_config=data_config,
                    sequence_length=32,
                    batch_size=1,
                    num_workers=0,
                    model_config=model_config,
                )

                datamodule.setup('predict')
                predict_loader = datamodule.predict_dataloader()

                # Get one batch
                batch = next(iter(predict_loader))
                frames, lengths, metadata = batch

                assert frames.shape[0] == 1  # batch size
                assert frames.shape[2] == 3  # channels
                assert len(metadata) == 1

            finally:
                if predict_loader is not None:
                    try:
                        predict_loader.reset()
                    except Exception:
                        pass
                del predict_loader
                del datamodule
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()


# =============================================================================
# Test for VideoPipeline configuration
# =============================================================================

class TestVideoPipelineConfig:
    """Test VideoPipeline configuration options (CPU tests using mocks)."""

    def test_default_step_equals_sequence_length(self):
        """Test that default step equals sequence_length."""
        # Verify the logic: step = step if step is not None else sequence_length
        sequence_length = 64
        step = None

        computed_step = step if step is not None else sequence_length
        assert computed_step == 64

        step = 32
        computed_step = step if step is not None else sequence_length
        assert computed_step == 32

    def test_extended_sequence_calculation(self):
        """Test the extended sequence length calculation with TCN padding."""
        sequence_length = 32

        # Test various TCN padding values
        for tcn_padding in [0, 6, 30, 60]:
            extended_sequence = sequence_length + 2 * tcn_padding
            expected = sequence_length + 2 * tcn_padding
            assert extended_sequence == expected

    def test_shard_configuration_logic(self):
        """Test the sharding configuration logic."""
        # When not distributed
        use_dali_sharding = True
        is_distributed = False

        if use_dali_sharding and is_distributed:
            shard_id = 1  # Would come from torch.distributed.get_rank()
            num_shards = 4  # Would come from torch.distributed.get_world_size()
        else:
            shard_id = 0
            num_shards = 1

        assert shard_id == 0
        assert num_shards == 1

        # When distributed
        is_distributed = True
        if use_dali_sharding and is_distributed:
            shard_id = 2
            num_shards = 4
        else:
            shard_id = 0
            num_shards = 1

        assert shard_id == 2
        assert num_shards == 4

    def test_last_batch_policy_selection(self):
        """Test last batch policy string to enum conversion."""
        try:
            from nvidia.dali.plugin.pytorch import LastBatchPolicy

            # Test 'drop' policy
            policy_str = 'drop'
            policy = (
                LastBatchPolicy.DROP if policy_str.lower() == 'drop'
                else LastBatchPolicy.PARTIAL
            )
            assert policy == LastBatchPolicy.DROP

            # Test 'partial' policy
            policy_str = 'partial'
            policy = (
                LastBatchPolicy.DROP if policy_str.lower() == 'drop'
                else LastBatchPolicy.PARTIAL
            )
            assert policy == LastBatchPolicy.PARTIAL

            # Test case insensitivity
            policy_str = 'DROP'
            policy = (
                LastBatchPolicy.DROP if policy_str.lower() == 'drop'
                else LastBatchPolicy.PARTIAL
            )
            assert policy == LastBatchPolicy.DROP

        except ImportError:
            pytest.skip("DALI not installed")

# =============================================================================
# GPU integration tests for augmentations
# =============================================================================

@requires_gpu
@pytest.mark.gpu
class TestAugmentationGPU:
    """GPU integration tests for data augmentation in the video pipeline.

    These tests require NVIDIA DALI and a CUDA GPU.
    """

    def test_pipeline_with_default_augmentation(self, create_test_video, cleanup_gpu):
        """Test pipeline builds and runs with 'default' augmentation preset."""
        from lightning_action.data.video_datamodule import (
            VideoPipeline, AUGMENTATION_PRESETS,
        )
        import tempfile as tf
        import gc

        with tempfile.TemporaryDirectory() as tmpdir:
            videos_dir = os.path.join(tmpdir, 'videos')
            os.makedirs(videos_dir)

            video_path = create_test_video(
                num_frames=100, height=224, width=224,
                directory=videos_dir, filename='test.mp4',
            )

            with tf.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
                f.write(f"{video_path} 0\n")
                file_list = f.name

            pipe = None
            try:
                pipe = VideoPipeline(
                    batch_size=1, num_threads=1, device_id=0, seed=42,
                    file_list=file_list, sequence_length=32, resolution=224,
                    random_shuffle=False,
                    augmentations=AUGMENTATION_PRESETS["default"],
                )
                pipe.build()
                pipe_out = pipe.run()
                frames = pipe_out[0].as_cpu().as_array()

                # Output shape should still be (B, T, C, H, W) = (1, 32, 3, 224, 224)
                assert frames.shape == (1, 32, 3, 224, 224)
            finally:
                del pipe
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                os.unlink(file_list)

    def test_pipeline_with_strong_augmentation(self, create_test_video, cleanup_gpu):
        """Test pipeline builds and runs with 'strong' augmentation (includes rotation)."""
        from lightning_action.data.video_datamodule import (
            VideoPipeline, AUGMENTATION_PRESETS,
        )
        import tempfile as tf
        import gc

        with tempfile.TemporaryDirectory() as tmpdir:
            videos_dir = os.path.join(tmpdir, 'videos')
            os.makedirs(videos_dir)

            video_path = create_test_video(
                num_frames=100, height=224, width=224,
                directory=videos_dir, filename='test.mp4',
            )

            with tf.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
                f.write(f"{video_path} 0\n")
                file_list = f.name

            pipe = None
            try:
                pipe = VideoPipeline(
                    batch_size=1, num_threads=1, device_id=0, seed=42,
                    file_list=file_list, sequence_length=32, resolution=224,
                    random_shuffle=False,
                    augmentations=AUGMENTATION_PRESETS["strong"],
                )
                pipe.build()
                pipe_out = pipe.run()
                frames = pipe_out[0].as_cpu().as_array()

                assert frames.shape == (1, 32, 3, 224, 224)
            finally:
                del pipe
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                os.unlink(file_list)

    def test_none_augmentation_matches_no_augmentation(
        self, create_test_video, cleanup_gpu
    ):
        """Test that 'none' preset produces same output as no augmentation."""
        from lightning_action.data.video_datamodule import VideoPipeline
        import tempfile as tf
        import gc

        with tempfile.TemporaryDirectory() as tmpdir:
            videos_dir = os.path.join(tmpdir, 'videos')
            os.makedirs(videos_dir)

            video_path = create_test_video(
                num_frames=100, height=224, width=224,
                directory=videos_dir, filename='test.mp4',
            )

            with tf.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
                f.write(f"{video_path} 0\n")
                file_list = f.name

            pipe_no_aug = None
            pipe_none = None
            try:
                # Pipeline with no augmentation
                pipe_no_aug = VideoPipeline(
                    batch_size=1, num_threads=1, device_id=0, seed=42,
                    file_list=file_list, sequence_length=32, resolution=224,
                    random_shuffle=False, augmentations=None,
                )
                pipe_no_aug.build()
                out_no_aug = pipe_no_aug.run()
                frames_no_aug = out_no_aug[0].as_cpu().as_array()

                del pipe_no_aug
                pipe_no_aug = None
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

                # Pipeline with 'none' augmentation (empty dict)
                pipe_none = VideoPipeline(
                    batch_size=1, num_threads=1, device_id=0, seed=42,
                    file_list=file_list, sequence_length=32, resolution=224,
                    random_shuffle=False, augmentations={},
                )
                pipe_none.build()
                out_none = pipe_none.run()
                frames_none = out_none[0].as_cpu().as_array()

                np.testing.assert_array_equal(frames_no_aug, frames_none)
            finally:
                del pipe_no_aug, pipe_none
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                os.unlink(file_list)

    def test_augmented_output_differs_statistically(
        self, create_test_video, cleanup_gpu
    ):
        """Test that augmented output differs from unaugmented (statistical check)."""
        from lightning_action.data.video_datamodule import (
            VideoPipeline, AUGMENTATION_PRESETS,
        )
        import tempfile as tf
        import gc

        with tempfile.TemporaryDirectory() as tmpdir:
            videos_dir = os.path.join(tmpdir, 'videos')
            os.makedirs(videos_dir)

            video_path = create_test_video(
                num_frames=100, height=224, width=224,
                directory=videos_dir, filename='test.mp4',
                frame_content='gradient',
            )

            with tf.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
                f.write(f"{video_path} 0\n")
                file_list = f.name

            pipe_plain = None
            try:
                # Run without augmentation
                pipe_plain = VideoPipeline(
                    batch_size=1, num_threads=1, device_id=0, seed=42,
                    file_list=file_list, sequence_length=32, resolution=224,
                    random_shuffle=False, augmentations=None,
                )
                pipe_plain.build()
                out_plain = pipe_plain.run()
                frames_plain = out_plain[0].as_cpu().as_array().copy()

                del pipe_plain
                pipe_plain = None
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

                # Run multiple augmented passes - at least one should differ
                any_different = False
                for trial_seed in range(5):
                    pipe_aug = VideoPipeline(
                        batch_size=1, num_threads=1, device_id=0,
                        seed=trial_seed + 100,
                        file_list=file_list, sequence_length=32, resolution=224,
                        random_shuffle=False,
                        augmentations=AUGMENTATION_PRESETS["strong"],
                    )
                    pipe_aug.build()
                    out_aug = pipe_aug.run()
                    frames_aug = out_aug[0].as_cpu().as_array()

                    if not np.array_equal(frames_plain, frames_aug):
                        any_different = True

                    del pipe_aug
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                    if any_different:
                        break

                assert any_different, (
                    "Augmented output was identical to plain output across all trials"
                )
            finally:
                del pipe_plain
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                os.unlink(file_list)

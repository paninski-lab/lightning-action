"""Dataset for video action segmentation.

This module contains the VideoDataset class for loading video chunks and labels.
"""

import os
import tempfile
import torch
import numpy as np
from torch.utils.data import Dataset
from nvidia.dali.pipeline import Pipeline
from nvidia.dali import fn, types
from typeguard import typechecked
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

class VideoDataset(Dataset):
    """Dataset for loading video chunks and corresponding labels.
    
    Loads MP4 videos in chunks using DALI for efficient GPU processing.
    """

    @typechecked
    def __init__(
        self,
        videos_dir: str,
        labels_dir: str,
        chunk_size: int = 128,
        resolution: int = 224,
        expt_ids: list[str] | None = None,
        input_size: int = 1536,
        num_lags: int = 0,
        ignore_index: int = -100,
    ):
        """Initialize VideoDataset.
        
        Args:
            videos_dir: directory containing MP4 video files
            labels_dir: directory containing NumPy (.npy) label files
            chunk_size: number of frames per chunk for predictions
            resolution: output frame resolution
            expt_ids: list of experiment IDs to filter videos (optional)
            input_size: dimensionality of input features after processing
            num_lags: number of context frames on each side
            ignore_index: value for ignored labels
            
        Raises:
            FileNotFoundError: if missing .npy files
            ValueError: if invalid video/label lengths or class mismatch
        """
        self.videos = [f for f in os.listdir(videos_dir) if f.endswith('.mp4')]
        if expt_ids:
            self.videos = [v for v in self.videos if any(v.startswith(eid) for eid in expt_ids)]
        self.videos_dir = videos_dir
        self.labels_dir = labels_dir
        self.chunk_size = chunk_size
        self.resolution = resolution
        self.input_size = input_size
        self.num_lags = num_lags
        self.ignore_index = ignore_index
        
        # compute chunks per video and determine num_classes
        self.video_chunks = []
        self.label_names = []
        self.num_classes = 0
        missing_npy = []
        for video in tqdm(self.videos, desc="Initializing Video Chunks"):
            label_path = os.path.join(self.labels_dir, video.replace('.mp4', '.npy'))
            if not os.path.exists(label_path):
                missing_npy.append(video)
            else:
                labels = np.load(label_path)
                if labels.ndim > 1 and labels.shape[1] > 1:
                    if not self.label_names:
                        self.label_names = [f'class_{i}' for i in range(labels.shape[1])]
                        self.num_classes = labels.shape[1]
                    labels = np.argmax(labels, axis=1)
                else:
                    unique_labels = np.unique(labels[labels >= 0])
                    if not self.label_names:
                        self.num_classes = int(max(unique_labels) + 1) if unique_labels.size > 0 else 1
                        self.label_names = [f'class_{i}' for i in range(self.num_classes)]
                total_frames = len(labels)
                num_chunks = (total_frames + self.chunk_size - 1) // self.chunk_size
                self.video_chunks.extend([(video, i) for i in range(num_chunks)])
        if missing_npy:
            raise FileNotFoundError(f"Missing .npy files for {len(missing_npy)} videos: {missing_npy[:5]}...")
        logger.info(f"Found {len(self.videos)} videos, {len(self.video_chunks)} chunks, {self.num_classes} classes")

    def __len__(self) -> int:
        """Return the total number of chunks across all videos."""
        return len(self.video_chunks)

    @typechecked
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a single video chunk and its labels.
        
        Args:
            idx: chunk index
            
        Returns:
            tuple of (frames, labels)
            
        Raises:
            IndexError: if idx is out of range
            ValueError: if frame size is invalid
        """
        if idx >= len(self.video_chunks):
            raise IndexError(f'Index {idx} out of range for dataset of size {len(self.video_chunks)}')
        
        video, chunk_idx = self.video_chunks[idx]
        video_path = os.path.join(self.videos_dir, video)
        label_path = os.path.join(self.labels_dir, video.replace('.mp4', '.npy'))
        
        # load labels
        labels = np.load(label_path)
        if labels.ndim > 1:
            labels = np.argmax(labels, axis=1)
        total_frames = len(labels)
        
        # compute middle chunk window
        start_middle = chunk_idx * self.chunk_size
        end_middle = min(start_middle + self.chunk_size, total_frames)
        middle_len = end_middle - start_middle
        
        # compute extended window for context
        start_frame = max(0, start_middle - self.num_lags)
        end_frame = min(total_frames, end_middle + self.num_lags)
        
        # create temporary file_list for DALI to specify start_frame
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp:
            tmp.write(f"{video_path} {start_frame}\n")
            tmp_file = tmp.name
        
        loaded_length = end_frame - start_frame
        
        # DALI pipeline for GPU video loading
        pipe = Pipeline(batch_size=1, num_threads=2, device_id=0)
        with pipe:
            frames = fn.readers.video(
                device="gpu",
                filenames=[video_path],
                sequence_length=loaded_length,
                skip_vfr_check=True,
                file_list_frame_num=True,
                file_list_include_preceding_frame=False,
                stride=1,
                step=-1,
                random_shuffle=False,
                pad_last_batch=False,
                image_type=types.RGB,
                dtype=types.UINT8,
            )
            frames = fn.resize(
                frames,
                resize_x=self.resolution,
                resize_y=self.resolution,
                interp_type=types.INTERP_LINEAR,
            )
            frames = fn.crop_mirror_normalize(
                frames,
                dtype=types.FLOAT,
                output_layout="FCHW",
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
            )
            pipe.set_outputs(frames)
        pipe.build()
        frames_gpu = pipe.run()[0]
        frames = torch.from_numpy(frames_gpu.as_cpu().as_array()[0])
        
        # clean up temp file
        os.unlink(tmp_file)
        
        # reshape to [loaded_length, 3, resolution, resolution]
        frames = frames.view(loaded_length, 3, self.resolution, self.resolution)
        
        # compute padding amounts
        extended_size = self.chunk_size + 2 * self.num_lags
        loaded_pre = start_middle - start_frame
        loaded_post = end_frame - end_middle
        left_pad = self.num_lags - loaded_pre
        right_pad = self.num_lags - loaded_post
        
        # pad frames with replicate manually
        if left_pad > 0 or right_pad > 0:
            if left_pad > 0:
                left_padding = frames[0:1].repeat(left_pad, 1, 1, 1)
                frames = torch.cat([left_padding, frames], dim=0)
            if right_pad > 0:
                right_padding = frames[-1:].repeat(right_pad, 1, 1, 1)
                frames = torch.cat([frames, right_padding], dim=0)
        
        # add extra padding if middle_len < chunk_size
        current_len = frames.shape[0]
        if current_len < extended_size:
            extra_right_pad = extended_size - current_len
            extra_padding = frames[-1:].repeat(extra_right_pad, 1, 1, 1)
            frames = torch.cat([frames, extra_padding], dim=0)
        
        # validate frame shape
        if frames.shape[0] != extended_size:
            raise ValueError(f"Extended frame size mismatch: got {frames.shape[0]}, expected {extended_size}")
        if frames.shape[1:] != (3, self.resolution, self.resolution):
            raise ValueError(f"Frame size mismatch: got {frames.shape[1:]}")
        
        # extract and pad labels for middle only
        middle_labels = labels[start_middle:end_middle]
        chunk_labels = torch.from_numpy(middle_labels).long()
        middle_len = len(chunk_labels)
        if middle_len < self.chunk_size:
            pad = (0, self.chunk_size - middle_len)
            chunk_labels = torch.nn.functional.pad(chunk_labels, pad, value=self.ignore_index)
        
        return frames, chunk_labels

    def get_label_names(self) -> list[str]:
        """Get list of label/class names."""
        return self.label_names.copy()

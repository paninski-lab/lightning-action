# Lightning Action - Video Pipeline

A guide to training action segmentation models directly from raw video files, using GPU-accelerated video decoding via NVIDIA DALI.

> This document covers the **video pipeline**. For the keypoint/CSV pipeline (markers, hand-crafted features), see the [main README](../README.md).

## Features

- **GPU-Accelerated Video Decoding**: NVIDIA DALI handles video reading, resizing, and normalization entirely on the GPU
- **Multiple Backbone Architectures**: ViT-MAE, ResNet (torchvision), and ResNet-BEAST variants
- **Temporal Heads**: Same heads as the keypoint pipeline — TemporalMLP, RNN (LSTM/GRU), and Dilated TCN
- **Backbone Freezing & Fine-Tuning**: Freeze the backbone entirely, or fine-tune with a separate learning rate
- **Frame-Level Predictions**: Generate per-frame class probabilities from raw `.mp4` files

## Installation

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA support (**required** — DALI does not support CPU-only operation)

### Install from Source

```bash
git clone https://github.com/paninski-lab/lightning-action.git
cd lightning-action
pip install -e .
```

### Install NVIDIA DALI

DALI must be installed separately. Match the package to your CUDA version:

```bash
# CUDA 12.x
pip install nvidia-dali-cuda120

# CUDA 11.x
pip install nvidia-dali-cuda110
```

See the [DALI installation guide](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/installation.html) for details.

## Quick Start

### 1. Prepare Your Data

Organize your data in the following structure:

```
data/
├── videos/
│   ├── experiment1.mp4
│   ├── experiment2.mp4
│   └── ...
└── labels_numpy/
    ├── experiment1.npy
    ├── experiment2.npy
    └── ...
```

**Videos** must be `.mp4` files. Each video file name (stem) is used as the experiment ID.

**Labels** are NumPy `.npy` files with the same stem as the corresponding video. Each file contains one of:
- A 1D array of integer class indices, shape `(num_frames,)` — e.g. `[0, 0, 1, 2, 2, ...]`
- A 2D one-hot array, shape `(num_frames, num_classes)` — automatically converted to class indices

Use `-100` for frames that should be ignored during training (the default `ignore_index`).

### 2. Create a Configuration File

Create a YAML configuration file with `data`, `model`, `optimizer`, and `training` sections:

```yaml
data:
  videos_dir: /path/to/data/videos
  labels_dir: /path/to/data/labels_numpy
  transform_mode: imagenet           # frame normalization preset (see table below)
  expt_ids:                          # list of video stems to use
    - experiment1
    - experiment2
  ignore_index: -100                 # label value to ignore in loss
  seed: 42

model:
  # Backbone (frame encoder)
  backbone: vitmae                   # see backbone table below
  backbone_config_path: configs/backbones/vit.yaml
  backbone_checkpoint: /path/to/backbone/weights.ckpt  # optional pretrained weights
  freeze_backbone: false             # true = frozen; false = fine-tune last layer

  # Temporal head
  head: dtcn                         # temporalmlp, rnn, or dtcn
  num_hid_units: 32
  num_layers: 4
  num_lags: 2                        # temporal context (dtcn/temporalmlp only)
  dropout_rate: 0.1
  output_size: 3                     # number of action classes
  seed: 42

optimizer:
  type: Adam
  lr: 0.0001
  backbone_lr: 0.0001               # separate LR for backbone (when not frozen)
  scheduler:
    use_scheduler: true
    type: CosineAnnealingWarmRestarts
    T_0: 34
    T_mult: 2
    eta_min_factor: 20

training:
  num_epochs: 160
  batch_size: 2
  sequence_length: 500               # frames per training chunk
  device: gpu
  num_workers: 4
  train_probability: 1.0             # fraction of videos for training
  val_probability: 0.0               # fraction of videos for validation
  checkpointing: true
  early_stopping: false
```

#### Available Backbones

| Config value | Architecture | Feature dim | Notes |
|---|---|---|---|
| `vitmae` | ViT-MAE (Base) | 768 | Requires `backbone_config_path` pointing to a ViT config |
| `resnet18` | ResNet-18 (torchvision) | 512 | Standard ImageNet-pretrained |
| `resnet34` | ResNet-34 (torchvision) | 512 | |
| `resnet50` | ResNet-50 (torchvision) | 2048 | |
| `resnet101` | ResNet-101 (torchvision) | 2048 | |
| `resnet152` | ResNet-152 (torchvision) | 2048 | |
| `resnet18-beast` | ResNet-18 (BEAST) | 512 | Custom BEAST implementation |
| `resnet34-beast` | ResNet-34 (BEAST) | 512 | |
| `resnet50-beast` | ResNet-50 (BEAST) | 2048 | |
| `resnet101-beast` | ResNet-101 (BEAST) | 2048 | |
| `resnet152-beast` | ResNet-152 (BEAST) | 2048 | |

The model's `input_size` is auto-computed from the backbone's feature dimension (doubled when velocity features are enabled, which is the default).

### 3. Train a Model

#### Using the CLI

```bash
litaction train --config configs/my_video_config.yaml --output-dir runs/my_experiment
```

CLI options (override config values):

| Flag | Description |
|---|---|
| `--output-dir`, `-o` | Output directory (default: `runs/YYYY-MM-DD/HH-MM-SS`) |
| `--data-dir` | Override `data.data_path` |
| `--device` | `cpu` or `gpu` |
| `--epochs` | Number of training epochs |
| `--batch-size` | Batch size |
| `--lr` | Learning rate |
| `--seed` | Random seed |
| `--overrides KEY=VALUE ...` | Arbitrary config overrides (dot notation) |

#### Using the Python API

```python
from lightning_action.api.video_model import VideoModel

# Create model from config
model = VideoModel.from_config('configs/my_video_config.yaml')

# Train (saves checkpoints, config, and TensorBoard logs to output_dir)
model.train(output_dir='runs/my_experiment')
```

### 4. Generate Predictions

After training, use the Python API to generate per-frame predictions:

```python
from lightning_action.api.video_model import VideoModel

# Load a trained model
model = VideoModel.from_dir('runs/my_experiment')

# Predict on all .mp4 files in a directory
model.predict(
    videos_dir='/path/to/videos',
    output_dir='runs/my_experiment/predictions',
)

# Or predict specific videos by experiment ID
model.predict(
    videos_dir='/path/to/videos',
    output_dir='runs/my_experiment/predictions',
    expt_ids=['experiment1', 'experiment3'],
)
```

Each prediction is saved as a CSV file (`<expt_id>_predictions.csv`) with columns:

| frame | class_0 | class_1 | class_2 |
|---|---|---|---|
| 0 | 0.85 | 0.10 | 0.05 |
| 1 | 0.80 | 0.12 | 0.08 |
| ... | ... | ... | ... |

If `label_names` are specified in the config, those names are used as column headers instead of `class_0`, `class_1`, etc.

## Monitoring Training with TensorBoard

Lightning Action automatically logs training metrics to TensorBoard. To visualize your training progress:

1. **Launch TensorBoard** after starting training:
   ```bash
   tensorboard --logdir runs/
   ```

2. **Open your browser** and navigate to `http://localhost:6006` to view the TensorBoard dashboard.

3. **Available metrics** include:
   - Training and validation loss
   - Training and validation accuracy
   - Training and validation F1 score
   - Learning rate schedules

**Tip**: Keep TensorBoard running while training multiple experiments to compare results in real-time.

## Further Reading

- [Main README](../README.md) — Keypoint/CSV pipeline, contributing guidelines, license, and citation
- [Video Pipeline Architecture Guide](video_pipeline_guide.md) — Developer-oriented deep-dive into the video pipeline internals

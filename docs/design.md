# Lightning Action — Architecture

This document describes the internal design of the codebase: the class hierarchy, data flow,
model architecture, and extension points. It covers both the CSV/keypoint pipeline and the video
pipeline together, highlighting where they share infrastructure and where they diverge.

For user-facing setup instructions, see:
- [README](../README.md) — CSV/keypoint pipeline quickstart
- [Video Pipeline Quickstart](video_pipeline_quickstart.md) — video pipeline quickstart
- [Video Pipeline Guide](video_pipeline_guide.md) — video pipeline internals in reading order

---

## Two Pipelines, One Framework

Lightning Action has two data pipelines that share a common model and training infrastructure:

| | **CSV pipeline** | **Video pipeline** |
|---|---|---|
| Input | Marker/feature CSV files | `.mp4` video files |
| Labels | CSV files | NumPy `.npy` arrays |
| DataModule | `DataModule` | `VideoDataModule` |
| Dataset | `FeatureDataset` | `VideoDataset` + DALI |
| Model | `Segmenter` | `VideoSegmenter` |
| API class | `Model` | `VideoModel` |
| Train function | `train()` | `train_video()` |
| GPU required | No | Yes (DALI) |

Both pipelines use the same temporal heads, the same loss and metrics logic, the same optimizer
configuration, and the same output directory structure.

---

## Class Hierarchy

```
pl.LightningModule
└── BaseModel                    (segmenter.py)
    ├── Segmenter                (segmenter.py)       ← CSV pipeline model
    └── VideoBaseModel           (video_segmenter.py)
        └── VideoSegmenter       (video_segmenter.py) ← video pipeline model

pl.LightningDataModule
├── DataModule                   (data/datamodule.py) ← CSV pipeline
└── VideoDataModule              (data/video_datamodule.py) ← video pipeline

ABC (BaseModelAPI)               (api/base.py)
├── Model                        (api/model.py)       ← CSV pipeline entry point
└── VideoModel                   (api/video_model.py) ← video pipeline entry point
```

---

## API Layer (`api/`)

`BaseModelAPI` is the user-facing entry point. Both `Model` and `VideoModel` inherit it and
override six abstract methods to plug in their pipeline-specific components.

### Factory methods

```python
# Create an untrained model from a YAML config
model = Model.from_config('configs/my_config.yaml')

# Load a trained model from a run directory
model = Model.from_dir('runs/my_experiment')
```

`from_config` reads the YAML, calls `_create_model_from_config()`, and returns an API wrapper.
`from_dir` additionally discovers the checkpoint file (prefers `*best*.ckpt`, then any `.ckpt`,
then `.pt`) and loads weights.

### Training

```python
model.train(output_dir='runs/my_experiment')
```

Calls `_get_train_function()` to get the appropriate `train()` or `train_video()` function, then
runs it inside a `chdir()` context so relative paths in the config resolve from `output_dir`.
After training, optionally runs `_run_post_training_inference()` to save predictions on the
training data.

### Abstract methods subclasses must implement

| Method | Purpose |
|---|---|
| `_create_model_from_config(config)` | Instantiate the Lightning model |
| `_get_model_class()` | Return `Segmenter` or `VideoSegmenter` (for checkpoint loading) |
| `_get_train_function()` | Return `train` or `train_video` |
| `_setup_trainer()` | Build a `pl.Trainer` for inference |
| `_run_post_training_inference()` | Save predictions on training data |
| `predict(...)` | Pipeline-specific inference |

---

## Model Layer (`models/`)

### BaseModel

`BaseModel` is a `pl.LightningModule` that provides everything except the architecture itself:

- **Metrics** — `torchmetrics.Accuracy` and `torchmetrics.F1Score` for train and val
- **Loss** — cross-entropy with optional per-class weights and an `ignore_index`
- **Optimizer configuration** — Adam / AdamW / SGD, optional LR scheduler (step, cosine,
  cosine warm restarts, reduce-on-plateau)
- **Sequence padding** — strips TCN receptive-field padding from predictions and targets before
  computing metrics (`_remove_padding`)
- **Lightning hooks** — `training_step`, `validation_step`, `predict_step`

Subclasses must implement two abstract methods: `_build_model()` and `forward()`.

### Segmenter (CSV pipeline)

`Segmenter` builds a two-part architecture:

```
input features (B, T, input_size)
        │
        ▼
   temporal head
 (TemporalMLP / RNN / DilatedTCN)
        │
        ▼
  linear classifier
        │
        ▼
logits (B, T, num_classes)
```

The head type is chosen by `config['model']['head']`. See [Temporal Heads](#temporal-heads) below.

### VideoSegmenter (video pipeline)

`VideoSegmenter` extends `VideoBaseModel` (which extends `BaseModel`) and builds a
five-stage architecture:

```
frames (B, T, C, H, W)
        │
        ▼  reshape to (B*T, C, H, W)
    backbone
 (ResNet / ResNetBeast / ViTMAE)
        │  spatial features (B*T, D, H', W')
        ▼
  attention pooling neck
 (MultiheadAttentionPooling)
        │  frame vectors (B, T, D)
        ▼
  velocity concatenation        ← optional (doubles feature dim to 2D)
        │
        ▼
   temporal head
 (TemporalMLP / RNN / DilatedTCN)
        │
        ▼
  linear classifier
        │
        ▼
logits (B, T, num_classes)
```

`input_size` for the temporal head is computed automatically from the backbone's `hidden_size`
(and doubled when velocity concat is enabled), so users do not need to calculate it manually.

`VideoBaseModel` also handles:
- Unpacking DALI's `(frames, labels, metadata)` tuples instead of the CSV pipeline's
  `(features, labels)` dicts
- Boundary-aware prediction slicing — overlapping chunks at video boundaries are stitched
  correctly so every frame gets exactly one prediction
- DDP-safe batch skipping — when all frames in a batch have `ignore_index` labels, all GPUs
  still participate in the forward pass to avoid NCCL hangs

---

## Temporal Heads (`models/heads/`)

All three heads share the same constructor signature:

```python
head = TemporalMLP(input_size, num_hid_units, num_layers, num_lags, ...)
head = RNN(input_size, num_hid_units, num_layers, rnn_type, bidirectional, ...)
head = DilatedTCN(input_size, num_hid_units, num_layers, num_lags, ...)
```

| Head | Config value | Description |
|---|---|---|
| `TemporalMLP` | `temporalmlp` | MLP with lagged inputs — concatenates the current and `num_lags` past frames before each linear layer |
| `RNN` | `rnn` | LSTM or GRU. Set `rnn_type: lstm` or `rnn_type: gru`; optionally `bidirectional: true` |
| `DilatedTCN` | `dtcn` | Dilated temporal convolutional network; receptive field grows exponentially with depth |

All heads output a tensor of shape `(B, T, num_hid_units)`, which the linear classifier maps to
`(B, T, num_classes)`.

### Receptive field padding (TCN and TemporalMLP)

Dilated TCN and TemporalMLP need extra context frames on each side of a sequence. The padding
size is computed by `compute_sequence_pad()` (`data/utils.py`) and stored as
`model.sequence_pad`. During training, batches are generated with the extra padding included;
`BaseModel._remove_padding` strips it before computing metrics.

---

## Frame Backbones (`models/backbones/`)

Used only in the video pipeline. All three backbones expose the same interface:

| Property / Method | Type | Description |
|---|---|---|
| `hidden_size` | `int` | Output feature dimension |
| `num_channels` | `int` | Expected input channels (3 for RGB) |
| `image_size` | `int` | Expected input resolution |
| `patch_size` | `int` | Effective spatial stride |
| `backbone_type` | `str` | String identifier |
| `forward(x)` | `(B,C,H,W) → (B,D,H',W')` | Extract spatial features |
| `load_pretrained_weights(path)` | — | Load from a checkpoint |
| `get_last_layer_params()` | `Iterator[Parameter]` | Parameters of the final layer (for fine-tuning) |

| Config value | Class | Feature dim |
|---|---|---|
| `vitmae` | `ViTMAEBackbone` | 768 (base) |
| `resnet18` … `resnet152` | `ResNetBackbone` | 512 / 2048 |
| `resnet18-beast` … `resnet152-beast` | `ResNetBeastBackbone` | 512 / 2048 |

### Backbone freezing

Set `freeze_backbone: true` to freeze all backbone weights. Set `freeze_backbone: false` to
fine-tune: the backbone gets its own LR (`backbone_lr`) while the head and classifier use the
main `lr`. This is implemented via separate optimizer parameter groups in
`VideoSegmenter._get_optimizer_params()`.

---

## Attention Pooling Neck (`models/necks/`)

`MultiheadAttentionPooling` sits between the backbone and the temporal head. It collapses a
spatial feature grid `(B, num_patches, D)` into a single frame vector `(B, 1, D)` using Pooling
by Multi-head Attention (PMA) from the Set Transformer paper:

```
patch features (B, num_patches, D)   ← keys and values
learnable seed vectors (1, num_seeds, D)   ← queries
        │
  multi-head cross-attention
        │
  optional FFN + layer norm
        │
frame vectors (B, num_seeds, D)
```

With `num_seeds=1` (default) this produces exactly one vector per frame.

---

## Data Layer (`data/`)

### CSV pipeline

```
config['data']['data_path']
        │
        ▼
  DataModule.setup()
        │
        ├── FeatureDataset (train split)
        │     ├── loads marker / feature CSVs
        │     ├── applies transforms (ZScore, MotionEnergy, VelocityConcat, …)
        │     └── yields fixed-length sequences {'input': tensor, 'labels': tensor}
        │
        └── FeatureDataset (val split)
```

Data transforms are composable via `Compose`. Available transforms:

| Class | Effect |
|---|---|
| `ZScore` | Zero-mean, unit-variance normalization per feature |
| `MotionEnergy` | Absolute frame-to-frame difference |
| `VelocityConcat` | Appends velocity (diff) features, doubling the feature dim |

### Video pipeline

```
config['data']['videos_dir'] + 'labels_dir'
        │
        ▼
  VideoDataset.setup()        ← file discovery, label loading, class weight computation
        │
        ▼
  VideoDataModule.setup()     ← video-level train/val split
        │
        ├── VideoPipeline (DALI, train)   ← GPU decode, resize, normalize
        │     └── DALIIterator             ← attaches labels + chunk metadata
        │
        └── VideoPipeline (DALI, val)
```

Videos are split at the video level (not chunk level) to prevent data leakage.
DALI decodes video directly on the GPU; each batch has shape `(B, T, C, H, W)`.

---

## Training Orchestration

### CSV pipeline: `train()` in `train.py`

1. `reset_seeds()` for reproducibility
2. Build `DataModule`, call `setup()`
3. Compute class weights from training labels; inject into config
4. Configure `TensorBoardLogger`, `ModelCheckpoint`, optional `EarlyStopping`, `LearningRateMonitor`
5. `trainer.fit(model, datamodule)`
6. `save_config(config, output_dir)` — writes final config (with computed weights, label names)

### Video pipeline: `train_video()` in `video_train.py`

Same flow as above, plus:
- `mp.set_start_method("spawn")` — required for DALI + DDP
- `use_distributed_sampler=False` — DALI handles its own GPU sharding
- NCCL backend for multi-GPU gradient sync

Both pipelines share utilities from `train_utils.py`: `reset_seeds`, `get_callbacks_from_config`,
`validate_config`, `save_config`, `update_config_with_class_weights`, `update_config_with_label_names`.

---

## Output Directory Structure

After training, the output directory contains:

```
runs/my_experiment/
├── config.yaml          ← final config (including computed class weights, label names)
├── checkpoints/
│   └── best-epoch=N-val_loss=X.ckpt
├── lightning_logs/
│   └── version_0/       ← TensorBoard logs
└── predictions/         ← post-training inference (if post_inference=True)
    ├── experiment1.csv
    └── experiment2.csv
```

---

## Configuration Structure

Configs are YAML files with four top-level sections. See `configs/README.md` for the full
parameter reference.

```yaml
data:
  data_path: /path/to/data     # (CSV) root directory
  input_dir: markers           # (CSV) subdirectory with input CSVs
  labels_dir: labels           # (CSV) subdirectory with label CSVs
  transforms: [ZScore]         # (CSV) list of transform names
  videos_dir: /path/to/videos  # (video) directory of .mp4 files
  labels_dir: /path/to/labels  # (video) directory of .npy label files
  expt_ids: [exp1, exp2]       # experiment IDs to include
  ignore_index: -100           # label value excluded from loss and metrics
  seed: 42

model:
  head: temporalmlp            # temporalmlp | rnn | dtcn
  input_size: 34               # (CSV) feature dimension
  output_size: 4               # number of classes
  num_hid_units: 256
  num_layers: 4
  num_lags: 2                  # temporal context (temporalmlp, dtcn)
  dropout_rate: 0.1
  backbone: vitmae             # (video) backbone name
  backbone_config_path: ...    # (video) backbone-specific config
  freeze_backbone: false       # (video) freeze backbone weights
  seed: 42

optimizer:
  type: Adam                   # Adam | AdamW | SGD
  lr: 0.001
  wd: 0.0                      # weight decay
  backbone_lr: 0.0001          # (video) separate LR for backbone
  scheduler:
    use_scheduler: true
    type: cosine               # step | cosine | cosine_warm_restarts | reduce_on_plateau

training:
  num_epochs: 100
  batch_size: 32
  sequence_length: 500         # frames per training chunk
  device: cpu                  # cpu | gpu
  train_probability: 0.8       # fraction of data for training
  val_probability: 0.2
  checkpointing: true
  early_stopping: false
  seed: 0
```

---

## Extension Points

### Adding a temporal head

1. Create `lightning_action/models/heads/my_head.py` implementing `nn.Module` with the
   constructor signature `(input_size, num_hid_units, num_layers, ...)` and a `forward` method
   that returns `(B, T, num_hid_units)`.
2. Export it from `lightning_action/models/heads/__init__.py`.
3. Add a branch in `Segmenter._build_head()` (and `VideoSegmenter._build_head()`) for the new
   `head` config value.
4. Add a test file at `tests/models/heads/test_my_head.py`.

### Adding a backbone (video pipeline only)

1. Create `lightning_action/models/backbones/my_backbone.py` as a subclass of `nn.Module`
   implementing the backbone interface (see [Frame Backbones](#frame-backbones) above):
   `hidden_size`, `num_channels`, `image_size`, `patch_size`, `backbone_type`, `forward()`,
   `load_pretrained_weights()`, `get_last_layer_params()`.
2. Register it in `VideoSegmenter._build_backbone()` for the new `backbone` config value.
3. Optionally add a YAML config under `configs/backbones/`.
4. Add a test file at `tests/models/backbones/test_my_backbone.py`.

### Adding a data transform (CSV pipeline only)

1. Add a class to `lightning_action/data/transforms.py` inheriting from `Transform` and
   implementing `__call__(x) -> x` and `__repr__() -> str`.
2. Register the class name in `DataModule._build_transforms()` so it can be referenced by name
   in configs.
3. Add tests in `tests/data/test_transforms.py`.

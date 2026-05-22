"""Tests for pydantic config validation."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from lightning_action.config import (
    DataConfig,
    LightningActionConfig,
    ModelConfig,
    OptimizerConfig,
    TrainingConfig,
    load_config,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_MINIMAL_CSV_CONFIG: dict = {
    'data': {'data_path': '/data'},
    'model': {'head': 'temporalmlp', 'output_size': 4, 'input_size': 10},
    'optimizer': {'lr': 1e-4, 'type': 'Adam'},
    'training': {'num_epochs': 10, 'device': 'cpu'},
}


def _write_yaml(tmp_path: Path, data: dict) -> Path:
    """Write a dict to a temporary YAML file and return the path."""
    p = tmp_path / 'config.yaml'
    with open(p, 'w') as f:
        yaml.dump(data, f)
    return p


# ---------------------------------------------------------------------------
# DataConfig
# ---------------------------------------------------------------------------

class TestDataConfig:
    """Test the DataConfig pydantic model."""

    def test_data_config_empty(self):
        # DataConfig has no required fields; all pipeline-specific fields pass
        # through as extras.
        cfg = DataConfig.model_validate({})
        assert cfg.model_dump() == {}

    def test_data_config_csv_extras(self):
        raw = {'data_path': '/data', 'expt_ids': None, 'seed': 42}
        cfg = DataConfig.model_validate(raw)
        result = cfg.model_dump()
        assert result['data_path'] == '/data'
        assert result['expt_ids'] is None
        assert result['seed'] == 42

    def test_data_config_video_extras(self):
        raw = {'videos_dir': '/videos', 'labels_dir': '/labels'}
        cfg = DataConfig.model_validate(raw)
        result = cfg.model_dump()
        assert result['videos_dir'] == '/videos'
        assert result['labels_dir'] == '/labels'


# ---------------------------------------------------------------------------
# ModelConfig
# ---------------------------------------------------------------------------

class TestModelConfig:
    """Test the ModelConfig pydantic model."""

    def test_model_config_valid_head_types(self):
        valid_heads = [
            'temporal-mlp', 'temporalmlp',
            'tcn', 'dtcn', 'dilatedtcn',
            'lstm', 'gru', 'rnn',
        ]
        for head in valid_heads:
            cfg = ModelConfig.model_validate({'head': head, 'output_size': 4})
            assert cfg.head == head

    def test_model_config_invalid_head(self):
        with pytest.raises(ValidationError, match='head'):
            ModelConfig.model_validate({'head': 'transformer', 'output_size': 4})

    def test_model_config_missing_head(self):
        with pytest.raises(ValidationError, match='head'):
            ModelConfig.model_validate({'output_size': 4})

    def test_model_config_missing_output_size(self):
        with pytest.raises(ValidationError, match='output_size'):
            ModelConfig.model_validate({'head': 'temporalmlp'})

    def test_model_config_defaults(self):
        cfg = ModelConfig.model_validate({'head': 'temporalmlp', 'output_size': 4})
        assert cfg.num_hid_units == 64
        assert cfg.num_layers == 2
        assert cfg.num_lags == 2

    def test_model_config_explicit_values(self):
        raw = {
            'head': 'rnn', 'output_size': 5, 'num_hid_units': 128, 'num_layers': 4, 'num_lags': 5,
        }
        cfg = ModelConfig.model_validate(raw)
        assert cfg.num_hid_units == 128
        assert cfg.num_layers == 4
        assert cfg.num_lags == 5

    def test_model_config_extra_fields_preserved(self):
        raw = {'head': 'rnn', 'output_size': 5, 'seed': 42}
        cfg = ModelConfig.model_validate(raw)
        assert cfg.model_dump()['seed'] == 42


# ---------------------------------------------------------------------------
# OptimizerConfig
# ---------------------------------------------------------------------------

class TestOptimizerConfig:
    """Test the OptimizerConfig pydantic model."""

    def test_optimizer_config_valid_types(self):
        for t in ['Adam', 'AdamW', 'SGD']:
            cfg = OptimizerConfig.model_validate({'lr': 1e-3, 'type': t})
            assert cfg.type == t

    def test_optimizer_config_type_case_insensitive(self):
        # Lowercase variants should also be accepted.
        for t in ['adam', 'adamw', 'sgd']:
            cfg = OptimizerConfig.model_validate({'lr': 1e-3, 'type': t})
            assert cfg.type == t

    def test_optimizer_config_invalid_type(self):
        with pytest.raises(ValidationError, match='Invalid optimizer type'):
            OptimizerConfig.model_validate({'lr': 1e-3, 'type': 'rmsprop'})

    def test_optimizer_config_default_type(self):
        cfg = OptimizerConfig.model_validate({'lr': 1e-3})
        assert cfg.type == 'Adam'

    def test_optimizer_config_missing_lr(self):
        with pytest.raises(ValidationError, match='lr'):
            OptimizerConfig.model_validate({'type': 'Adam'})

    def test_optimizer_config_string_scheduler_preserved(self):
        raw = {'lr': 1e-4, 'type': 'Adam', 'scheduler': 'cosine'}
        cfg = OptimizerConfig.model_validate(raw)
        assert cfg.model_dump()['scheduler'] == 'cosine'

    def test_optimizer_config_dict_scheduler_preserved(self):
        raw = {
            'lr': 1e-4,
            'type': 'Adam',
            'scheduler': {'type': 'CosineAnnealingWarmRestarts', 'T_0': 34},
        }
        cfg = OptimizerConfig.model_validate(raw)
        result = cfg.model_dump()
        assert result['scheduler']['T_0'] == 34


# ---------------------------------------------------------------------------
# TrainingConfig
# ---------------------------------------------------------------------------

class TestTrainingConfig:
    """Test the TrainingConfig pydantic model."""

    def test_training_config_valid_devices(self):
        for device in ['cpu', 'gpu']:
            cfg = TrainingConfig.model_validate({'num_epochs': 10, 'device': device})
            assert cfg.device == device

    def test_training_config_invalid_device(self):
        with pytest.raises(ValidationError, match='Invalid device'):
            TrainingConfig.model_validate({'num_epochs': 10, 'device': 'tpu'})

    def test_training_config_default_device(self):
        cfg = TrainingConfig.model_validate({'num_epochs': 10})
        assert cfg.device == 'cpu'

    def test_training_config_missing_num_epochs(self):
        with pytest.raises(ValidationError, match='num_epochs'):
            TrainingConfig.model_validate({'device': 'cpu'})

    def test_training_config_defaults(self):
        cfg = TrainingConfig.model_validate({'num_epochs': 10})
        assert cfg.sequence_length == 500
        assert cfg.batch_size == 32
        assert cfg.num_workers == 4
        assert cfg.train_probability == pytest.approx(0.9)
        assert cfg.val_probability == pytest.approx(0.1)

    def test_training_config_explicit_values(self):
        raw = {
            'num_epochs': 200,
            'device': 'gpu',
            'sequence_length': 1000,
            'batch_size': 8,
            'num_workers': 2,
            'train_probability': 0.95,
            'val_probability': 0.05,
        }
        cfg = TrainingConfig.model_validate(raw)
        assert cfg.sequence_length == 1000
        assert cfg.batch_size == 8
        assert cfg.num_workers == 2
        assert cfg.train_probability == pytest.approx(0.95)
        assert cfg.val_probability == pytest.approx(0.05)

    def test_training_config_extras_preserved(self):
        raw = {'num_epochs': 100, 'device': 'cpu', 'log_every_n_steps': 10}
        cfg = TrainingConfig.model_validate(raw)
        result = cfg.model_dump()
        assert result['log_every_n_steps'] == 10


# ---------------------------------------------------------------------------
# LightningActionConfig
# ---------------------------------------------------------------------------

class TestLightningActionConfig:
    """Test the top-level LightningActionConfig model."""

    def test_valid_csv_config(self):
        cfg = LightningActionConfig.model_validate(_MINIMAL_CSV_CONFIG)
        result = cfg.model_dump()
        assert result['model']['head'] == 'temporalmlp'
        assert result['optimizer']['lr'] == pytest.approx(1e-4)
        assert result['training']['num_epochs'] == 10

    def test_missing_top_level_section(self):
        for missing_key in ['data', 'model', 'optimizer', 'training']:
            raw = {k: v for k, v in _MINIMAL_CSV_CONFIG.items() if k != missing_key}
            with pytest.raises(ValidationError, match=missing_key):
                LightningActionConfig.model_validate(raw)

    def test_extra_top_level_fields_ignored(self):
        # Pydantic ignores unknown top-level keys by default; this allows
        # reloading saved configs that include the 'version' field added by
        # save_config() during training.
        raw = {**_MINIMAL_CSV_CONFIG, 'version': '1.1.0'}
        cfg = LightningActionConfig.model_validate(raw)
        assert cfg.model.head == 'temporalmlp'

    def test_model_dump_preserves_extras(self):
        raw = dict(_MINIMAL_CSV_CONFIG)
        raw['model'] = {**raw['model'], 'num_lags': 3, 'seed': 7}  # type: ignore[assignment]
        cfg = LightningActionConfig.model_validate(raw)
        result = cfg.model_dump()
        assert result['model']['num_lags'] == 3
        assert result['model']['seed'] == 7

    def test_video_pipeline_config(self):
        raw = {
            'data': {'videos_dir': '/v', 'labels_dir': '/l'},
            'model': {
                'head': 'dtcn',
                'output_size': 3,
                'backbone': 'vitmae',
                'num_hid_units': 32,
            },
            'optimizer': {
                'lr': 1e-3,
                'type': 'Adam',
                'scheduler': {'type': 'CosineAnnealingWarmRestarts', 'T_0': 34},
            },
            'training': {'num_epochs': 100, 'device': 'gpu', 'batch_size': 2},
        }
        cfg = LightningActionConfig.model_validate(raw)
        result = cfg.model_dump()
        assert result['model']['backbone'] == 'vitmae'
        assert result['optimizer']['scheduler']['T_0'] == 34
        assert result['training']['device'] == 'gpu'


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------

class TestLoadConfig:
    """Test the load_config function."""

    def test_load_config_returns_dict(self, tmp_path: Path):
        p = _write_yaml(tmp_path, _MINIMAL_CSV_CONFIG)
        result = load_config(p)
        assert isinstance(result, dict)
        assert 'data' in result
        assert 'model' in result
        assert 'optimizer' in result
        assert 'training' in result

    def test_load_config_file_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match='Config file not found'):
            load_config(tmp_path / 'nonexistent.yaml')

    def test_load_config_accepts_string_path(self, tmp_path: Path):
        p = _write_yaml(tmp_path, _MINIMAL_CSV_CONFIG)
        result = load_config(str(p))
        assert result['model']['head'] == 'temporalmlp'

    def test_load_config_invalid_head_raises(self, tmp_path: Path):
        raw = dict(_MINIMAL_CSV_CONFIG)
        raw['model'] = {**raw['model'], 'head': 'transformer'}  # type: ignore[assignment]
        p = _write_yaml(tmp_path, raw)
        with pytest.raises(ValidationError, match='head'):
            load_config(p)

    def test_load_config_invalid_optimizer_type_raises(self, tmp_path: Path):
        raw = dict(_MINIMAL_CSV_CONFIG)
        raw['optimizer'] = {**raw['optimizer'], 'type': 'rmsprop'}  # type: ignore[assignment]
        p = _write_yaml(tmp_path, raw)
        with pytest.raises(ValidationError, match='Invalid optimizer type'):
            load_config(p)

    def test_load_config_invalid_device_raises(self, tmp_path: Path):
        raw = dict(_MINIMAL_CSV_CONFIG)
        raw['training'] = {**raw['training'], 'device': 'tpu'}  # type: ignore[assignment]
        p = _write_yaml(tmp_path, raw)
        with pytest.raises(ValidationError, match='Invalid device'):
            load_config(p)

    def test_load_config_preserves_extra_fields(self, tmp_path: Path):
        raw = dict(_MINIMAL_CSV_CONFIG)
        raw['model'] = {**raw['model'], 'num_lags': 5}  # type: ignore[assignment]
        raw['training'] = {**raw['training'], 'batch_size': 64}  # type: ignore[assignment]
        p = _write_yaml(tmp_path, raw)
        result = load_config(p)
        assert result['model']['num_lags'] == 5
        assert result['training']['batch_size'] == 64

    def test_load_config_coerces_lr_to_float(self, tmp_path: Path):
        raw = dict(_MINIMAL_CSV_CONFIG)
        raw['optimizer'] = {**raw['optimizer'], 'lr': '0.001'}  # type: ignore[assignment]
        p = _write_yaml(tmp_path, raw)
        result = load_config(p)
        assert isinstance(result['optimizer']['lr'], float)
        assert result['optimizer']['lr'] == pytest.approx(0.001)

    def test_load_config_null_expt_ids_preserved(self, tmp_path: Path):
        raw = dict(_MINIMAL_CSV_CONFIG)
        raw['data'] = {**raw['data'], 'expt_ids': None}  # type: ignore[assignment]
        p = _write_yaml(tmp_path, raw)
        result = load_config(p)
        assert result['data']['expt_ids'] is None


# ---------------------------------------------------------------------------
# Smoke tests against the checked-in example configs
# ---------------------------------------------------------------------------

_CONFIGS_DIR = Path(__file__).parent.parent / 'configs'
_TOP_LEVEL_CONFIGS = sorted(_CONFIGS_DIR.glob('*.yaml'))


class TestExampleConfigs:
    """Smoke-test every top-level YAML in configs/ against the pydantic schema.

    Backbone sub-configs (configs/backbones/) are excluded because they are
    not top-level pipeline configs and lack required sections.
    """

    @pytest.mark.parametrize('config_path', _TOP_LEVEL_CONFIGS, ids=lambda p: p.name)
    def test_example_config_validates(self, config_path: Path):
        result = load_config(config_path)
        assert isinstance(result, dict)
        assert 'data' in result
        assert 'model' in result
        assert 'optimizer' in result
        assert 'training' in result

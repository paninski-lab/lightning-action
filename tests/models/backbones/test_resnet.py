"""Tests for ResNet backbone."""

import pytest
import torch

from lightning_action.models.backbones.resnet import RESNET_HIDDEN_SIZES, ResNetBackbone


@pytest.fixture(scope='module')
def resnet18_backbone():
    """Module-scoped resnet18 backbone to avoid repeated model construction."""
    return ResNetBackbone({'backbone': 'resnet18'})


class TestResNetBackbone:
    """Test the ResNetBackbone class."""

    @pytest.fixture
    def default_config(self):
        """Default config using resnet18 (smallest/fastest)."""
        return {'backbone': 'resnet18'}

    def test_default_initialization(self):
        """Test initialization with default config (resnet50)."""
        backbone = ResNetBackbone()
        assert backbone.backbone_name == 'resnet50'
        assert backbone.hidden_size == 2048

    def test_custom_initialization(self, default_config):
        """Test initialization with custom config."""
        backbone = ResNetBackbone(default_config)
        assert backbone.backbone_name == 'resnet18'
        assert backbone.hidden_size == 512

    def test_all_supported_variants(self):
        """Test that all supported variants can be initialized."""
        for name in RESNET_HIDDEN_SIZES:
            backbone = ResNetBackbone({'backbone': name})
            assert backbone.backbone_name == name
            assert backbone.hidden_size == RESNET_HIDDEN_SIZES[name]

    def test_invalid_backbone(self):
        """Test that unsupported backbone raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported backbone"):
            ResNetBackbone({'backbone': 'resnet999'})

    @pytest.mark.parametrize("backbone_name,expected_hidden_size", RESNET_HIDDEN_SIZES.items())
    def test_properties(self, backbone_name, expected_hidden_size):
        """Test backbone properties for all variants."""
        backbone = ResNetBackbone({'backbone': backbone_name})
        assert backbone.hidden_size == expected_hidden_size
        assert backbone.num_channels == 3
        assert backbone.image_size == 224
        assert backbone.patch_size == 32
        assert backbone.backbone_name == backbone_name
        assert backbone.backbone_type == 'resnet'

    def test_hidden_size_mapping(self):
        """Test hidden size mapping for different variants."""
        # resnet18/34 -> 512
        for name in ['resnet18', 'resnet34']:
            backbone = ResNetBackbone({'backbone': name})
            assert backbone.hidden_size == 512

        # resnet50/101/152 -> 2048
        for name in ['resnet50', 'resnet101', 'resnet152']:
            backbone = ResNetBackbone({'backbone': name})
            assert backbone.hidden_size == 2048

    @pytest.mark.parametrize("backbone_name,expected_hidden_size", RESNET_HIDDEN_SIZES.items())
    def test_forward_pass_shape(self, backbone_name, expected_hidden_size):
        """Test forward pass produces correct output shape for all variants."""
        backbone = ResNetBackbone({'backbone': backbone_name})
        x = torch.randn(2, 3, 224, 224)
        output = backbone(x)

        assert output.shape == (2, expected_hidden_size, 7, 7)
        assert torch.isfinite(output).all()

    def test_forward_channel_validation(self, default_config):
        """Test that wrong input channels raises ValueError."""
        backbone = ResNetBackbone(default_config)
        x = torch.randn(2, 1, 224, 224)  # wrong channels

        with pytest.raises(ValueError, match="Input has 1 channels"):
            backbone(x)

    @pytest.mark.parametrize("backbone_name", RESNET_HIDDEN_SIZES.keys())
    def test_gradient_flow(self, backbone_name):
        """Test that gradients flow through all variants."""
        backbone = ResNetBackbone({'backbone': backbone_name})
        x = torch.randn(1, 3, 224, 224, requires_grad=True)
        output = backbone(x)

        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_load_pretrained_weights_file_not_found(self, default_config):
        """Test that missing checkpoint raises FileNotFoundError."""
        backbone = ResNetBackbone(default_config)
        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            backbone.load_pretrained_weights('/nonexistent/checkpoint.ckpt')

    def test_get_last_layer_params(self, default_config):
        """Test that get_last_layer_params returns parameters from layer4."""
        backbone = ResNetBackbone(default_config)
        params = list(backbone.get_last_layer_params())

        assert len(params) > 0
        # All should be Parameters
        for p in params:
            assert isinstance(p, torch.nn.Parameter)


class TestResNetBackboneLoadPretrainedWeights:
    """Test the load_pretrained_weights method of ResNetBackbone."""

    def test_load_raw_state_dict(self, resnet18_backbone, tmp_path):
        """Test loading a plain state dict with no wrapper key."""
        ckpt = tmp_path / 'model.pt'
        torch.save(resnet18_backbone.backbone.state_dict(), ckpt)
        resnet18_backbone.load_pretrained_weights(ckpt)

    def test_load_state_dict_key(self, resnet18_backbone, tmp_path):
        """Test loading a Lightning-style checkpoint with 'state_dict' wrapper."""
        ckpt = tmp_path / 'model.pt'
        torch.save({'state_dict': resnet18_backbone.backbone.state_dict()}, ckpt)
        resnet18_backbone.load_pretrained_weights(ckpt)

    def test_load_model_state_dict_key(self, resnet18_backbone, tmp_path):
        """Test loading a PyTorch training-loop checkpoint with 'model_state_dict' wrapper."""
        ckpt = tmp_path / 'model.pt'
        torch.save({'model_state_dict': resnet18_backbone.backbone.state_dict()}, ckpt)
        resnet18_backbone.load_pretrained_weights(ckpt)

    def test_load_with_backbone_prefix(self, resnet18_backbone, tmp_path):
        """Test that 'backbone.' key prefix is stripped before matching."""
        prefixed = {
            f'backbone.{k}': v
            for k, v in resnet18_backbone.backbone.state_dict().items()
        }
        ckpt = tmp_path / 'model.pt'
        torch.save(prefixed, ckpt)
        resnet18_backbone.load_pretrained_weights(ckpt)

    def test_fc_keys_are_skipped(self, resnet18_backbone, tmp_path, capsys):
        """Test that 'fc.' keys are skipped and do not cause errors."""
        state_dict = resnet18_backbone.backbone.state_dict()
        state_dict['fc.weight'] = torch.zeros(1000, 512)
        state_dict['fc.bias'] = torch.zeros(1000)
        ckpt = tmp_path / 'model.pt'
        torch.save(state_dict, ckpt)
        resnet18_backbone.load_pretrained_weights(ckpt)

    def test_no_matching_weights_prints_warning(self, resnet18_backbone, tmp_path, capsys):
        """Test that a checkpoint with no matching keys prints a warning."""
        ckpt = tmp_path / 'model.pt'
        torch.save({'completely_wrong_key': torch.zeros(1)}, ckpt)
        resnet18_backbone.load_pretrained_weights(ckpt)
        assert 'Warning' in capsys.readouterr().out

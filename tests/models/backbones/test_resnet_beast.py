"""Tests for ResNet Beast backbone."""

import pytest
import torch

from lightning_action.models.backbones.resnet_beast import (
    BEAST_RESNET_HIDDEN_SIZES,
    BottleneckBlock,
    ResidualBlock,
    ResNetBeast,
    ResNetBeastBackbone,
    get_configs,
)

EXPECTED_CONFIGS = {
    'resnet18': ([2, 2, 2, 2], False),
    'resnet34': ([3, 4, 6, 3], False),
    'resnet50': ([3, 4, 6, 3], True),
    'resnet101': ([3, 4, 23, 3], True),
    'resnet152': ([3, 8, 36, 3], True),
}


class TestGetConfigs:
    """Test the get_configs function."""

    @pytest.mark.parametrize("arch,expected", EXPECTED_CONFIGS.items())
    def test_configs(self, arch, expected):
        """Test config for each architecture variant."""
        layers, bottleneck = get_configs(arch)
        assert layers == expected[0]
        assert bottleneck == expected[1]

    def test_invalid_arch(self):
        """Test that invalid architecture raises ValueError."""
        with pytest.raises(ValueError, match="not a valid ResNet architecture"):
            get_configs('resnet999')


class TestResNetBeastBackbone:
    """Test the ResNetBeastBackbone class."""

    @pytest.fixture
    def default_config(self):
        """Default config using resnet18 (smallest/fastest)."""
        return {'backbone': 'resnet18'}

    def test_default_initialization(self):
        """Test initialization with default config (resnet50)."""
        backbone = ResNetBeastBackbone()
        assert backbone._backbone_name == 'resnet50'
        assert backbone.hidden_size == 2048

    def test_custom_initialization(self, default_config):
        """Test initialization with custom config."""
        backbone = ResNetBeastBackbone(default_config)
        assert backbone._backbone_name == 'resnet18'
        assert backbone.hidden_size == 512

    def test_invalid_backbone(self):
        """Test that unsupported backbone raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported backbone"):
            ResNetBeastBackbone({'backbone': 'resnet999'})

    @pytest.mark.parametrize(
        "backbone_name,expected_hidden_size", BEAST_RESNET_HIDDEN_SIZES.items(),
    )
    def test_properties(self, backbone_name, expected_hidden_size):
        """Test backbone properties for each variant."""
        backbone = ResNetBeastBackbone({'backbone': backbone_name})
        assert backbone.hidden_size == expected_hidden_size
        assert backbone.num_channels == 3
        assert backbone.image_size == 224
        assert backbone.patch_size == 32
        assert backbone.backbone_type == 'resnet-beast'

    @pytest.mark.parametrize(
        "backbone_name,expected_hidden_size", BEAST_RESNET_HIDDEN_SIZES.items(),
    )
    def test_forward_pass_shape(self, backbone_name, expected_hidden_size):
        """Test forward pass produces correct output shape for each variant."""
        backbone = ResNetBeastBackbone({'backbone': backbone_name})
        x = torch.randn(2, 3, 224, 224)
        output = backbone(x)

        assert output.shape == (2, expected_hidden_size, 7, 7)
        assert torch.isfinite(output).all()

    @pytest.mark.parametrize("backbone_name", BEAST_RESNET_HIDDEN_SIZES.keys())
    def test_gradient_flow(self, backbone_name):
        """Test that gradients flow through the model for each variant."""
        backbone = ResNetBeastBackbone({'backbone': backbone_name})
        x = torch.randn(1, 3, 224, 224, requires_grad=True)
        output = backbone(x)

        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_load_pretrained_weights_file_not_found(self, default_config):
        """Test that missing checkpoint raises FileNotFoundError."""
        backbone = ResNetBeastBackbone(default_config)
        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            backbone.load_pretrained_weights('/nonexistent/checkpoint.ckpt')

    def test_get_last_layer_params(self, default_config):
        """Test that get_last_layer_params returns parameters from conv5."""
        backbone = ResNetBeastBackbone(default_config)
        params = list(backbone.get_last_layer_params())

        assert len(params) > 0
        for p in params:
            assert isinstance(p, torch.nn.Parameter)


class TestResNetBeastComponents:
    """Test internal ResNetBeast components."""

    def test_resnet_beast_forward_no_bottleneck(self):
        """Test ResNetBeast forward pass without bottleneck (resnet18-style)."""
        model = ResNetBeast(configs=[2, 2, 2, 2], bottleneck=False)
        x = torch.randn(1, 3, 224, 224)
        output = model(x)

        assert output.shape == (1, 512, 7, 7)

    def test_resnet_beast_forward_bottleneck(self):
        """Test ResNetBeast forward pass with bottleneck (resnet50-style)."""
        model = ResNetBeast(configs=[3, 4, 6, 3], bottleneck=True)
        x = torch.randn(1, 3, 224, 224)
        output = model(x)

        assert output.shape == (1, 2048, 7, 7)

    def test_resnet_beast_invalid_configs(self):
        """Test that invalid config length raises ValueError."""
        with pytest.raises(ValueError, match="Only 4 layers can be configured"):
            ResNetBeast(configs=[2, 2, 2], bottleneck=False)

    def test_residual_block_forward(self):
        """Test ResidualBlock forward pass."""
        block = ResidualBlock(
            in_channels=64, hidden_channels=128, layers=2, downsample_method='conv',
        )
        x = torch.randn(1, 64, 28, 28)
        output = block(x)

        assert output.shape == (1, 128, 14, 14)

    def test_bottleneck_block_forward(self):
        """Test BottleneckBlock forward pass."""
        block = BottleneckBlock(
            in_channels=64, hidden_channels=64, up_channels=256,
            layers=3, downsample_method='pool',
        )
        x = torch.randn(1, 64, 56, 56)
        output = block(x)

        assert output.shape == (1, 256, 28, 28)


@pytest.fixture(scope='module')
def resnet18_beast_backbone():
    """Module-scoped resnet18 ResNetBeast backbone to avoid repeated model construction."""
    return ResNetBeastBackbone({'backbone': 'resnet18'})


class TestResNetBeastBackboneLoadPretrainedWeights:
    """Test the load_pretrained_weights method of ResNetBeastBackbone."""

    def test_load_raw_state_dict(self, resnet18_beast_backbone, tmp_path):
        """Test loading a plain state dict with no wrapper key."""
        ckpt = tmp_path / 'model.pt'
        torch.save(resnet18_beast_backbone.backbone.state_dict(), ckpt)
        resnet18_beast_backbone.load_pretrained_weights(ckpt)

    def test_load_state_dict_key(self, resnet18_beast_backbone, tmp_path):
        """Test loading a Lightning-style checkpoint with 'state_dict' wrapper."""
        ckpt = tmp_path / 'model.pt'
        torch.save({'state_dict': resnet18_beast_backbone.backbone.state_dict()}, ckpt)
        resnet18_beast_backbone.load_pretrained_weights(ckpt)

    def test_load_model_state_dict_key(self, resnet18_beast_backbone, tmp_path):
        """Test loading a PyTorch training-loop checkpoint with 'model_state_dict' wrapper."""
        ckpt = tmp_path / 'model.pt'
        torch.save({'model_state_dict': resnet18_beast_backbone.backbone.state_dict()}, ckpt)
        resnet18_beast_backbone.load_pretrained_weights(ckpt)

    def test_load_with_encoder_prefix(self, resnet18_beast_backbone, tmp_path):
        """Test that 'encoder.' key prefix is stripped before matching."""
        prefixed = {
            f'encoder.{k}': v
            for k, v in resnet18_beast_backbone.backbone.state_dict().items()
        }
        ckpt = tmp_path / 'model.pt'
        torch.save(prefixed, ckpt)
        resnet18_beast_backbone.load_pretrained_weights(ckpt)

    def test_decoder_and_latent_keys_are_skipped(self, resnet18_beast_backbone, tmp_path, capsys):
        """Test that decoder and latent keys are skipped without error."""
        state_dict = resnet18_beast_backbone.backbone.state_dict()
        state_dict['decoder.weight'] = torch.zeros(10, 10)
        state_dict['latent_mapping.weight'] = torch.zeros(10, 10)
        ckpt = tmp_path / 'model.pt'
        torch.save(state_dict, ckpt)
        resnet18_beast_backbone.load_pretrained_weights(ckpt)

    def test_no_matching_weights_prints_warning(self, resnet18_beast_backbone, tmp_path, capsys):
        """Test that a checkpoint with no matching keys prints a warning."""
        ckpt = tmp_path / 'model.pt'
        torch.save({'completely_wrong_key': torch.zeros(1)}, ckpt)
        resnet18_beast_backbone.load_pretrained_weights(ckpt)
        assert 'Warning' in capsys.readouterr().out

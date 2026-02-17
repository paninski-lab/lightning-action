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


class TestGetConfigs:
    """Test the get_configs function."""

    def test_resnet18(self):
        """Test resnet18 config."""
        layers, bottleneck = get_configs('resnet18')
        assert layers == [2, 2, 2, 2]
        assert bottleneck is False

    def test_resnet34(self):
        """Test resnet34 config."""
        layers, bottleneck = get_configs('resnet34')
        assert layers == [3, 4, 6, 3]
        assert bottleneck is False

    def test_resnet50(self):
        """Test resnet50 config."""
        layers, bottleneck = get_configs('resnet50')
        assert layers == [3, 4, 6, 3]
        assert bottleneck is True

    def test_resnet101(self):
        """Test resnet101 config."""
        layers, bottleneck = get_configs('resnet101')
        assert layers == [3, 4, 23, 3]
        assert bottleneck is True

    def test_resnet152(self):
        """Test resnet152 config."""
        layers, bottleneck = get_configs('resnet152')
        assert layers == [3, 8, 36, 3]
        assert bottleneck is True

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

    def test_properties(self, default_config):
        """Test backbone properties."""
        backbone = ResNetBeastBackbone(default_config)
        assert backbone.hidden_size == 512
        assert backbone.num_channels == 3
        assert backbone.image_size == 224
        assert backbone.patch_size == 32
        assert backbone.backbone_type == 'resnet-beast'

    def test_hidden_size_mapping(self):
        """Test hidden size mapping for different variants."""
        for name in ['resnet18', 'resnet34']:
            backbone = ResNetBeastBackbone({'backbone': name})
            assert backbone.hidden_size == 512

        for name in ['resnet50']:
            backbone = ResNetBeastBackbone({'backbone': name})
            assert backbone.hidden_size == 2048

    def test_forward_pass_shape(self, default_config):
        """Test forward pass produces correct output shape."""
        backbone = ResNetBeastBackbone(default_config)
        x = torch.randn(2, 3, 224, 224)
        output = backbone(x)

        assert output.shape == (2, 512, 7, 7)
        assert torch.isfinite(output).all()

    def test_forward_bottleneck_variant(self):
        """Test forward pass with bottleneck variant (resnet50)."""
        backbone = ResNetBeastBackbone({'backbone': 'resnet50'})
        x = torch.randn(2, 3, 224, 224)
        output = backbone(x)

        assert output.shape == (2, 2048, 7, 7)
        assert torch.isfinite(output).all()

    def test_gradient_flow(self, default_config):
        """Test that gradients flow through the model."""
        backbone = ResNetBeastBackbone(default_config)
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

"""Tests for ResNet backbone."""

import pytest
import torch

from lightning_action.models.backbones.resnet import ResNetBackbone, RESNET_HIDDEN_SIZES


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

"""Tests for ViT-MAE backbone."""

import pytest
import torch

from lightning_action.models.backbones.vitmae import ViTMAEBackbone


@pytest.fixture(scope="module")
def vitmae_backbone():
    """Module-scoped ViTMAE backbone to avoid repeated HuggingFace model loading."""
    return ViTMAEBackbone()


class TestViTMAEBackbone:
    """Test the ViTMAEBackbone class."""

    def test_default_initialization(self, vitmae_backbone):
        """Test initialization with default config."""
        assert vitmae_backbone.backbone_type == 'vitmae'
        assert vitmae_backbone.hidden_size == 768

    def test_custom_mask_ratio(self):
        """Test initialization with custom mask_ratio."""
        backbone = ViTMAEBackbone({'mask_ratio': 0.5})
        assert backbone.vit_mae.config.mask_ratio == 0.5

    def test_properties(self, vitmae_backbone):
        """Test backbone properties."""
        assert vitmae_backbone.hidden_size == 768
        assert vitmae_backbone.num_channels == 3
        assert vitmae_backbone.image_size == 224
        assert vitmae_backbone.patch_size == 16
        assert vitmae_backbone.backbone_type == 'vitmae'

    def test_forward_pass_shape(self, vitmae_backbone):
        """Test forward pass produces correct output shape."""
        x = torch.randn(2, 3, 224, 224)
        output = vitmae_backbone(x)

        assert output.shape == (2, 768, 14, 14)
        assert torch.isfinite(output).all()

    def test_forward_channel_validation(self, vitmae_backbone):
        """Test that wrong input channels raises ValueError."""
        x = torch.randn(2, 1, 224, 224)

        with pytest.raises(ValueError, match="Input has 1 channels"):
            vitmae_backbone(x)

    def test_gradient_flow(self, vitmae_backbone):
        """Test that gradients flow through the model."""
        x = torch.randn(1, 3, 224, 224, requires_grad=True)
        output = vitmae_backbone(x)

        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_load_pretrained_weights_file_not_found(self, vitmae_backbone):
        """Test that missing checkpoint raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            vitmae_backbone.load_pretrained_weights('/nonexistent/checkpoint.ckpt')

    def test_get_last_layer_params(self, vitmae_backbone):
        """Test that get_last_layer_params returns parameters."""
        params = list(vitmae_backbone.get_last_layer_params())

        assert len(params) > 0
        for p in params:
            assert isinstance(p, torch.nn.Parameter)


class TestViTMAEBackboneLoadPretrainedWeights:
    """Test the load_pretrained_weights method of ViTMAEBackbone."""

    def test_load_raw_state_dict(self, vitmae_backbone, tmp_path):
        """Test loading a plain state dict with no wrapper key."""
        ckpt = tmp_path / 'model.pt'
        torch.save(vitmae_backbone.vit_mae.state_dict(), ckpt)
        vitmae_backbone.load_pretrained_weights(ckpt)

    def test_load_state_dict_key(self, vitmae_backbone, tmp_path):
        """Test loading a Lightning-style checkpoint with 'state_dict' wrapper."""
        ckpt = tmp_path / 'model.pt'
        torch.save({'state_dict': vitmae_backbone.vit_mae.state_dict()}, ckpt)
        vitmae_backbone.load_pretrained_weights(ckpt)

    def test_load_model_state_dict_key(self, vitmae_backbone, tmp_path):
        """Test loading a PyTorch training-loop checkpoint with 'model_state_dict' wrapper."""
        ckpt = tmp_path / 'model.pt'
        torch.save({'model_state_dict': vitmae_backbone.vit_mae.state_dict()}, ckpt)
        vitmae_backbone.load_pretrained_weights(ckpt)

    def test_load_with_vit_prefix(self, vitmae_backbone, tmp_path):
        """Test that 'vit.' key prefix is stripped before matching."""
        prefixed = {
            f'vit.{k}': v
            for k, v in vitmae_backbone.vit_mae.state_dict().items()
        }
        ckpt = tmp_path / 'model.pt'
        torch.save(prefixed, ckpt)
        vitmae_backbone.load_pretrained_weights(ckpt)

    def test_decoder_and_mask_token_keys_are_skipped(self, vitmae_backbone, tmp_path, capsys):
        """Test that decoder and mask_token keys are skipped without error."""
        state_dict = vitmae_backbone.vit_mae.state_dict()
        state_dict['decoder.weight'] = torch.zeros(10, 10)
        state_dict['mask_token'] = torch.zeros(1, 1, 768)
        ckpt = tmp_path / 'model.pt'
        torch.save(state_dict, ckpt)
        vitmae_backbone.load_pretrained_weights(ckpt)

    def test_no_matching_weights_prints_warning(self, vitmae_backbone, tmp_path, capsys):
        """Test that a checkpoint with no matching keys prints a warning."""
        ckpt = tmp_path / 'model.pt'
        torch.save({'completely_wrong_key': torch.zeros(1)}, ckpt)
        vitmae_backbone.load_pretrained_weights(ckpt)
        assert 'Warning' in capsys.readouterr().out

"""Tests for MultiheadAttentionPooling module.

This module tests the attention-based pooling mechanism used to aggregate
spatial features from vision transformers into fixed-size representations.

Tests cover:
- Initialization and parameter validation
- Output shapes for various configurations
- Attention weight computation
- Optional components (FFN, layer norm)
- Gradient flow for training
"""

import pytest
import torch
import torch.nn as nn


class TestMultiheadAttentionPoolingInit:
    """Test MultiheadAttentionPooling initialization."""
    
    def test_basic_initialization(self):
        """Test basic initialization with default parameters."""
        from lightning_action.models.necks.mha_pooling import MultiheadAttentionPooling
        
        pooling = MultiheadAttentionPooling(
            embed_dim=768,
            num_heads=8,
        )
        
        assert pooling.embed_dim == 768
        assert pooling.num_heads == 8
        assert pooling.num_seeds == 1  # default
        assert pooling.use_ffn is True  # default

    def test_custom_num_seeds(self):
        """Test initialization with custom num_seeds."""
        from lightning_action.models.necks.mha_pooling import MultiheadAttentionPooling
        
        pooling = MultiheadAttentionPooling(
            embed_dim=256,
            num_heads=4,
            num_seeds=4,
        )
        
        assert pooling.num_seeds == 4
        assert pooling.seeds.shape == (1, 4, 256)

    def test_embed_dim_not_divisible_by_num_heads_raises(self):
        """Test that embed_dim must be divisible by num_heads."""
        from lightning_action.models.necks.mha_pooling import MultiheadAttentionPooling
        
        with pytest.raises(ValueError, match="must be divisible by"):
            MultiheadAttentionPooling(
                embed_dim=768,
                num_heads=7,  # 768 % 7 != 0
            )

    def test_seeds_shape(self):
        """Test that seeds parameter has correct shape."""
        from lightning_action.models.necks.mha_pooling import MultiheadAttentionPooling
        
        pooling = MultiheadAttentionPooling(
            embed_dim=512,
            num_heads=8,
            num_seeds=3,
        )
        
        assert pooling.seeds.shape == (1, 3, 512)
        assert isinstance(pooling.seeds, nn.Parameter)

    def test_use_ffn_true(self):
        """Test initialization with FFN enabled."""
        from lightning_action.models.necks.mha_pooling import MultiheadAttentionPooling
        
        pooling = MultiheadAttentionPooling(
            embed_dim=256,
            num_heads=4,
            use_ffn=True,
        )
        
        assert pooling.ffn is not None
        assert isinstance(pooling.ffn, nn.Linear)
        assert pooling.ffn.in_features == 256
        assert pooling.ffn.out_features == 256

    def test_use_ffn_false(self):
        """Test initialization with FFN disabled."""
        from lightning_action.models.necks.mha_pooling import MultiheadAttentionPooling
        
        pooling = MultiheadAttentionPooling(
            embed_dim=256,
            num_heads=4,
            use_ffn=False,
        )
        
        assert pooling.ffn is None

    def test_layer_norm_true(self):
        """Test initialization with layer normalization enabled."""
        from lightning_action.models.necks.mha_pooling import MultiheadAttentionPooling
        
        pooling = MultiheadAttentionPooling(
            embed_dim=256,
            num_heads=4,
            layer_norm=True,
        )
        
        assert isinstance(pooling.norm1, nn.LayerNorm)
        assert isinstance(pooling.norm2, nn.LayerNorm)
        assert pooling.norm1.normalized_shape == (256,)

    def test_layer_norm_false(self):
        """Test initialization with layer normalization disabled."""
        from lightning_action.models.necks.mha_pooling import MultiheadAttentionPooling
        
        pooling = MultiheadAttentionPooling(
            embed_dim=256,
            num_heads=4,
            layer_norm=False,
        )
        
        assert isinstance(pooling.norm1, nn.Identity)
        assert isinstance(pooling.norm2, nn.Identity)

    def test_various_valid_configurations(self):
        """Test various valid embed_dim and num_heads combinations."""
        from lightning_action.models.necks.mha_pooling import MultiheadAttentionPooling
        
        valid_configs = [
            (64, 1),
            (64, 2),
            (64, 4),
            (64, 8),
            (128, 8),
            (256, 8),
            (512, 8),
            (768, 12),
            (1024, 16),
            (2048, 32),
        ]
        
        for embed_dim, num_heads in valid_configs:
            pooling = MultiheadAttentionPooling(
                embed_dim=embed_dim,
                num_heads=num_heads,
            )
            assert pooling.embed_dim == embed_dim
            assert pooling.num_heads == num_heads


class TestMultiheadAttentionPoolingForward:
    """Test MultiheadAttentionPooling forward pass."""
    
    def test_output_shape_basic(self):
        """Test basic output shape."""
        from lightning_action.models.necks.mha_pooling import MultiheadAttentionPooling
        
        pooling = MultiheadAttentionPooling(
            embed_dim=768,
            num_heads=8,
            num_seeds=1,
        )
        
        batch_size = 4
        seq_len = 196  # 14x14 patches from ViT
        x = torch.randn(batch_size, seq_len, 768)
        
        output = pooling(x)
        
        assert output.shape == (batch_size, 1, 768)

    def test_output_shape_multiple_seeds(self):
        """Test output shape with multiple seeds."""
        from lightning_action.models.necks.mha_pooling import MultiheadAttentionPooling
        
        pooling = MultiheadAttentionPooling(
            embed_dim=256,
            num_heads=4,
            num_seeds=4,
        )
        
        batch_size = 8
        seq_len = 100
        x = torch.randn(batch_size, seq_len, 256)
        
        output = pooling(x)
        
        assert output.shape == (batch_size, 4, 256)

    def test_output_shape_various_batch_sizes(self):
        """Test output shape with various batch sizes."""
        from lightning_action.models.necks.mha_pooling import MultiheadAttentionPooling
        
        pooling = MultiheadAttentionPooling(
            embed_dim=512,
            num_heads=8,
            num_seeds=2,
        )
        
        for batch_size in [1, 2, 4, 8, 16, 32]:
            x = torch.randn(batch_size, 64, 512)
            output = pooling(x)
            assert output.shape == (batch_size, 2, 512)

    def test_output_shape_various_sequence_lengths(self):
        """Test output shape with various sequence lengths."""
        from lightning_action.models.necks.mha_pooling import MultiheadAttentionPooling
        
        pooling = MultiheadAttentionPooling(
            embed_dim=256,
            num_heads=4,
            num_seeds=1,
        )
        
        batch_size = 4
        # Output shape should be independent of sequence length
        for seq_len in [1, 10, 49, 100, 196, 256, 1024]:
            x = torch.randn(batch_size, seq_len, 256)
            output = pooling(x)
            assert output.shape == (batch_size, 1, 256)

    def test_return_attention_false(self):
        """Test that return_attention=False returns only output tensor."""
        from lightning_action.models.necks.mha_pooling import MultiheadAttentionPooling
        
        pooling = MultiheadAttentionPooling(
            embed_dim=256,
            num_heads=4,
        )
        
        x = torch.randn(2, 64, 256)
        output = pooling(x, return_attention=False)
        
        assert isinstance(output, torch.Tensor)
        assert output.shape == (2, 1, 256)

    def test_return_attention_true(self):
        """Test that return_attention=True returns tuple of (output, weights)."""
        from lightning_action.models.necks.mha_pooling import MultiheadAttentionPooling
        
        pooling = MultiheadAttentionPooling(
            embed_dim=256,
            num_heads=4,
            num_seeds=2,
        )
        
        batch_size = 4
        seq_len = 64
        x = torch.randn(batch_size, seq_len, 256)
        
        result = pooling(x, return_attention=True)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        
        output, attn_weights = result
        assert output.shape == (batch_size, 2, 256)
        # Attention weights: (batch_size, num_heads, num_seeds, seq_len)
        assert attn_weights.shape == (batch_size, 4, 2, seq_len)

    def test_attention_weights_sum_to_one(self):
        """Test that attention weights sum to 1 along sequence dimension."""
        from lightning_action.models.necks.mha_pooling import MultiheadAttentionPooling
        
        pooling = MultiheadAttentionPooling(
            embed_dim=256,
            num_heads=4,
            num_seeds=1,
        )
        
        x = torch.randn(2, 64, 256)
        _, attn_weights = pooling(x, return_attention=True)
        
        # Sum along last dimension (seq_len) should be ~1
        attn_sum = attn_weights.sum(dim=-1)
        assert torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-5)

    def test_deterministic_output(self):
        """Test that same input produces same output (no randomness in eval mode)."""
        from lightning_action.models.necks.mha_pooling import MultiheadAttentionPooling
        
        pooling = MultiheadAttentionPooling(
            embed_dim=256,
            num_heads=4,
            dropout=0.0,  # No dropout for determinism
        )
        pooling.eval()
        
        x = torch.randn(2, 64, 256)
        
        output1 = pooling(x)
        output2 = pooling(x)
        
        assert torch.allclose(output1, output2)

    def test_batch_independence(self):
        """Test that batches are processed independently."""
        from lightning_action.models.necks.mha_pooling import MultiheadAttentionPooling
        
        pooling = MultiheadAttentionPooling(
            embed_dim=256,
            num_heads=4,
            dropout=0.0,
        )
        pooling.eval()
        
        # Create two different inputs
        x1 = torch.randn(1, 64, 256)
        x2 = torch.randn(1, 64, 256)
        
        # Process separately
        out1_single = pooling(x1)
        out2_single = pooling(x2)
        
        # Process together as batch
        x_batch = torch.cat([x1, x2], dim=0)
        out_batch = pooling(x_batch)
        
        # Results should match
        assert torch.allclose(out1_single, out_batch[0:1], atol=1e-5)
        assert torch.allclose(out2_single, out_batch[1:2], atol=1e-5)


class TestMultiheadAttentionPoolingGradients:
    """Test gradient flow through MultiheadAttentionPooling."""
    
    def test_gradients_flow_to_input(self):
        """Test that gradients flow back to input."""
        from lightning_action.models.necks.mha_pooling import MultiheadAttentionPooling
        
        pooling = MultiheadAttentionPooling(
            embed_dim=256,
            num_heads=4,
        )
        
        x = torch.randn(2, 64, 256, requires_grad=True)
        output = pooling(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert not torch.all(x.grad == 0)

    def test_gradients_flow_to_seeds(self):
        """Test that gradients flow to learnable seed parameters."""
        from lightning_action.models.necks.mha_pooling import MultiheadAttentionPooling
        
        pooling = MultiheadAttentionPooling(
            embed_dim=256,
            num_heads=4,
        )
        
        x = torch.randn(2, 64, 256)
        output = pooling(x)
        loss = output.sum()
        loss.backward()
        
        assert pooling.seeds.grad is not None
        assert pooling.seeds.grad.shape == pooling.seeds.shape
        assert not torch.all(pooling.seeds.grad == 0)

    def test_gradients_flow_to_ffn(self):
        """Test that gradients flow to FFN when enabled."""
        from lightning_action.models.necks.mha_pooling import MultiheadAttentionPooling
        
        pooling = MultiheadAttentionPooling(
            embed_dim=256,
            num_heads=4,
            use_ffn=True,
        )
        
        x = torch.randn(2, 64, 256)
        output = pooling(x)
        loss = output.sum()
        loss.backward()
        
        assert pooling.ffn.weight.grad is not None
        assert pooling.ffn.bias.grad is not None

    def test_no_ffn_gradients_when_disabled(self):
        """Test that FFN has no gradients when disabled."""
        from lightning_action.models.necks.mha_pooling import MultiheadAttentionPooling
        
        pooling = MultiheadAttentionPooling(
            embed_dim=256,
            num_heads=4,
            use_ffn=False,
        )
        
        x = torch.randn(2, 64, 256)
        output = pooling(x)
        loss = output.sum()
        loss.backward()
        
        # FFN should be None when disabled
        assert pooling.ffn is None


class TestMultiheadAttentionPoolingDropout:
    """Test dropout behavior in MultiheadAttentionPooling."""
    
    def test_dropout_affects_training_mode(self):
        """Test that dropout affects output during training."""
        from lightning_action.models.necks.mha_pooling import MultiheadAttentionPooling
        
        pooling = MultiheadAttentionPooling(
            embed_dim=256,
            num_heads=4,
            dropout=0.5,  # High dropout for visibility
        )
        pooling.train()
        
        torch.manual_seed(42)
        x = torch.randn(4, 64, 256)
        
        # Multiple forward passes should give different results due to dropout
        outputs = []
        for _ in range(5):
            outputs.append(pooling(x).clone())
        
        # At least some outputs should differ (with high probability)
        all_same = all(torch.allclose(outputs[0], out) for out in outputs[1:])
        # Note: This could theoretically fail but is extremely unlikely with dropout=0.5
        assert not all_same, "Dropout should cause variation in training mode"

    def test_dropout_no_effect_eval_mode(self):
        """Test that dropout has no effect during evaluation."""
        from lightning_action.models.necks.mha_pooling import MultiheadAttentionPooling
        
        pooling = MultiheadAttentionPooling(
            embed_dim=256,
            num_heads=4,
            dropout=0.5,
        )
        pooling.eval()
        
        x = torch.randn(4, 64, 256)
        
        output1 = pooling(x)
        output2 = pooling(x)
        
        assert torch.allclose(output1, output2)

    def test_zero_dropout_deterministic(self):
        """Test that zero dropout gives deterministic results."""
        from lightning_action.models.necks.mha_pooling import MultiheadAttentionPooling
        
        pooling = MultiheadAttentionPooling(
            embed_dim=256,
            num_heads=4,
            dropout=0.0,
        )
        pooling.train()  # Even in train mode
        
        x = torch.randn(4, 64, 256)
        
        output1 = pooling(x)
        output2 = pooling(x)
        
        assert torch.allclose(output1, output2)


class TestMultiheadAttentionPoolingEdgeCases:
    """Test edge cases for MultiheadAttentionPooling."""
    
    def test_single_token_sequence(self):
        """Test with sequence length of 1."""
        from lightning_action.models.necks.mha_pooling import MultiheadAttentionPooling
        
        pooling = MultiheadAttentionPooling(
            embed_dim=256,
            num_heads=4,
        )
        
        x = torch.randn(2, 1, 256)  # Single token
        output = pooling(x)
        
        assert output.shape == (2, 1, 256)

    def test_batch_size_one(self):
        """Test with batch size of 1."""
        from lightning_action.models.necks.mha_pooling import MultiheadAttentionPooling
        
        pooling = MultiheadAttentionPooling(
            embed_dim=512,
            num_heads=8,
        )
        
        x = torch.randn(1, 64, 512)
        output = pooling(x)
        
        assert output.shape == (1, 1, 512)

    def test_large_num_seeds(self):
        """Test with num_seeds larger than typical."""
        from lightning_action.models.necks.mha_pooling import MultiheadAttentionPooling
        
        pooling = MultiheadAttentionPooling(
            embed_dim=256,
            num_heads=4,
            num_seeds=16,
        )
        
        x = torch.randn(2, 64, 256)
        output = pooling(x)
        
        assert output.shape == (2, 16, 256)

    def test_num_seeds_equals_sequence_length(self):
        """Test when num_seeds equals sequence length."""
        from lightning_action.models.necks.mha_pooling import MultiheadAttentionPooling
        
        pooling = MultiheadAttentionPooling(
            embed_dim=256,
            num_heads=4,
            num_seeds=64,
        )
        
        x = torch.randn(2, 64, 256)  # seq_len == num_seeds
        output = pooling(x)
        
        assert output.shape == (2, 64, 256)

    def test_small_embed_dim(self):
        """Test with small embed_dim."""
        from lightning_action.models.necks.mha_pooling import MultiheadAttentionPooling
        
        pooling = MultiheadAttentionPooling(
            embed_dim=32,
            num_heads=2,
        )
        
        x = torch.randn(4, 16, 32)
        output = pooling(x)
        
        assert output.shape == (4, 1, 32)

    def test_single_head(self):
        """Test with single attention head."""
        from lightning_action.models.necks.mha_pooling import MultiheadAttentionPooling
        
        pooling = MultiheadAttentionPooling(
            embed_dim=256,
            num_heads=1,
        )
        
        x = torch.randn(2, 64, 256)
        output, attn_weights = pooling(x, return_attention=True)
        
        assert output.shape == (2, 1, 256)
        assert attn_weights.shape == (2, 1, 1, 64)  # Single head


class TestMultiheadAttentionPoolingDevices:
    """Test MultiheadAttentionPooling on different devices."""
    
    def test_cpu_forward(self):
        """Test forward pass on CPU."""
        from lightning_action.models.necks.mha_pooling import MultiheadAttentionPooling
        
        pooling = MultiheadAttentionPooling(
            embed_dim=256,
            num_heads=4,
        ).cpu()
        
        x = torch.randn(2, 64, 256, device='cpu')
        output = pooling(x)
        
        assert output.device.type == 'cpu'
        assert output.shape == (2, 1, 256)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_forward(self):
        """Test forward pass on CUDA."""
        from lightning_action.models.necks.mha_pooling import MultiheadAttentionPooling
        
        pooling = MultiheadAttentionPooling(
            embed_dim=256,
            num_heads=4,
        ).cuda()
        
        x = torch.randn(2, 64, 256, device='cuda')
        output = pooling(x)
        
        assert output.device.type == 'cuda'
        assert output.shape == (2, 1, 256)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_gradients(self):
        """Test gradient computation on CUDA."""
        from lightning_action.models.necks.mha_pooling import MultiheadAttentionPooling
        
        pooling = MultiheadAttentionPooling(
            embed_dim=256,
            num_heads=4,
        ).cuda()
        
        x = torch.randn(2, 64, 256, device='cuda', requires_grad=True)
        output = pooling(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.device.type == 'cuda'


class TestMultiheadAttentionPoolingIntegration:
    """Integration tests for MultiheadAttentionPooling."""
    
    def test_typical_vit_usage(self):
        """Test with typical ViT output dimensions."""
        from lightning_action.models.necks.mha_pooling import MultiheadAttentionPooling

        # ViT-Base: 768-dim features, 196 patches (14x14)
        pooling = MultiheadAttentionPooling(
            embed_dim=768,
            num_heads=12,
            num_seeds=1,
        )
        
        batch_size = 8
        num_patches = 196
        x = torch.randn(batch_size, num_patches, 768)
        
        output = pooling(x)
        
        # Should pool 196 patches to 1 vector per sample
        assert output.shape == (batch_size, 1, 768)

    def test_typical_resnet_usage(self):
        """Test with typical ResNet output dimensions."""
        from lightning_action.models.necks.mha_pooling import MultiheadAttentionPooling

        # ResNet-50: 2048-dim features, 7x7=49 spatial locations
        pooling = MultiheadAttentionPooling(
            embed_dim=2048,
            num_heads=16,
            num_seeds=1,
        )
        
        batch_size = 4
        spatial_locs = 49
        x = torch.randn(batch_size, spatial_locs, 2048)
        
        output = pooling(x)
        
        assert output.shape == (batch_size, 1, 2048)

    def test_video_frame_processing(self):
        """Test processing multiple frames (like in video segmentation)."""
        from lightning_action.models.necks.mha_pooling import MultiheadAttentionPooling
        
        pooling = MultiheadAttentionPooling(
            embed_dim=768,
            num_heads=8,
            num_seeds=1,
        )
        
        # Simulate batch of frames, each with patch features
        batch_frames = 32  # 32 frames in batch
        num_patches = 196
        x = torch.randn(batch_frames, num_patches, 768)
        
        output = pooling(x)
        
        # Each frame pooled to single vector
        assert output.shape == (batch_frames, 1, 768)
        # Can squeeze for temporal processing
        output_squeezed = output.squeeze(1)
        assert output_squeezed.shape == (batch_frames, 768)

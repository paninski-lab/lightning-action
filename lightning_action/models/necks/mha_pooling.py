"""Multi-head attention pooling neck for video segmentation.

This module provides attention-based pooling to aggregate spatial features
(e.g., from a ViT backbone) into a fixed-size representation.

A "neck" connects the backbone (image encoder) to the head (temporal model).
This particular neck uses Pooling by Multi-head Attention (PMA) to aggregate
variable-length sequences into fixed-size outputs.

Example usage:
    pooling = MultiheadAttentionPooling(
        embed_dim=768,
        num_heads=8,
        num_seeds=1,  # Pool to single vector
    )

    # Input: patch features from ViT
    x = torch.randn(batch_size, num_patches, 768)

    # Output: pooled representation
    pooled = pooling(x)  # (batch_size, 1, 768)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiheadAttentionPooling(nn.Module):
    """Pooling by Multi-head Attention (PMA).

    Uses learnable seed vectors as queries to pool a variable-length
    sequence into a fixed number of output vectors via cross-attention.

    This is commonly used to aggregate spatial features from vision
    transformers into a single frame-level representation.

    Architecture:
        1. Learnable seed vectors S serve as queries
        2. Input sequence X serves as keys and values
        3. Cross-attention: output = Attention(Q=S, K=X, V=X)
        4. Optional: FFN with residual connection

    For num_seeds=1, this reduces (B, L, D) -> (B, 1, D), effectively
    summarizing all input tokens into one vector.

    Attributes:
        embed_dim: Feature dimension.
        num_heads: Number of attention heads.
        num_seeds: Number of output vectors (learnable queries).
        seeds: Learnable query vectors, shape (1, num_seeds, embed_dim).

    Reference:
        Lee et al., "Set Transformer: A Framework for Attention-based
        Permutation-Invariant Neural Networks", ICML 2019
        https://arxiv.org/abs/1810.00825
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_seeds: int = 1,
        dropout: float = 0.0,
        use_ffn: bool = True,
        layer_norm: bool = False,
    ):
        """Initialize the attention pooling module.

        Args:
            embed_dim: Dimension of input and output features.
            num_heads: Number of attention heads.
            num_seeds: Number of output vectors (learnable queries).
                Use 1 to pool to a single vector per input.
            dropout: Dropout probability for attention weights.
            use_ffn: Whether to include feed-forward network after attention.
            layer_norm: Whether to apply layer normalization.
        """
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_seeds = num_seeds
        self.use_ffn = use_ffn
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Learnable seed vectors (queries for pooling)
        self.seeds = nn.Parameter(torch.empty(1, num_seeds, embed_dim))
        nn.init.xavier_uniform_(self.seeds)

        self.fc_q = nn.Linear(embed_dim, embed_dim)
        self.fc_k = nn.Linear(embed_dim, embed_dim)
        self.fc_v = nn.Linear(embed_dim, embed_dim)

        # Optional layer normalization
        self.norm1 = nn.LayerNorm(embed_dim) if layer_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(embed_dim) if layer_norm else nn.Identity()

        # Optional feed-forward network
        if use_ffn:
            self.ffn = nn.Linear(embed_dim, embed_dim)
        else:
            self.ffn = None

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Pool input sequence using attention.

        Args:
            x: Input features, shape (batch_size, seq_len, embed_dim).
            return_attention: If True, also return attention weights.

        Returns:
            Pooled output, shape (batch_size, num_seeds, embed_dim).
            If return_attention=True, also returns attention weights.
        """
        batch_size = x.size(0)
        seq_len = x.size(1)

        # Expand seeds for batch: (1, num_seeds, D) -> (B, num_seeds, D)
        seeds = self.seeds.expand(batch_size, -1, -1)

        # Project Q, K, V (matching sandbox)
        Q = self.fc_q(seeds)   # (B, num_seeds, embed_dim)
        K = self.fc_k(x)       # (B, seq_len, embed_dim)
        V = self.fc_v(x)       # (B, seq_len, embed_dim)

        # Split into heads: (B, L, D) -> (B*H, L, D/H)
        dim_split = self.embed_dim // self.num_heads

        # Reshape for multi-head: concat along batch dim
        Q_ = torch.cat(Q.split(dim_split, dim=2), dim=0)  # (B*H, num_seeds, D/H)
        K_ = torch.cat(K.split(dim_split, dim=2), dim=0)  # (B*H, seq_len, D/H)
        V_ = torch.cat(V.split(dim_split, dim=2), dim=0)  # (B*H, seq_len, D/H)

        # Compute attention scores
        scores = Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.embed_dim)
        A = torch.softmax(scores, dim=2)  # (B*H, num_seeds, seq_len)
        A = self.dropout(A)

        # Per-head residual with PROJECTED query
        # This preserves query information at each head before concatenation
        out = Q_ + A.bmm(V_)  # (B*H, num_seeds, D/H)

        # Merge heads back: (B*H, num_seeds, D/H) -> (B, num_seeds, D)
        out = torch.cat(out.split(batch_size, dim=0), dim=2)

        # Optional layer norm after attention
        if self.norm1 is not None:
            out = self.norm1(out)

        # FFN with residual
        if self.ffn is not None:
            out = out + F.relu(self.ffn(out))
            if self.norm2 is not None:
                out = self.norm2(out)

        if return_attention:
            # Reshape attention weights: (B*H, num_seeds, seq_len) -> (B, H, num_seeds, seq_len)
            A = A.view(batch_size, self.num_heads, self.num_seeds, seq_len)
            return out, A

        return out

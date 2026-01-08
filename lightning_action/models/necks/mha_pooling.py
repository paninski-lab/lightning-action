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

from typing import Optional, Tuple, Union

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
        
        # Learnable seed vectors (queries for pooling)
        self.seeds = nn.Parameter(torch.empty(1, num_seeds, embed_dim))
        nn.init.xavier_uniform_(self.seeds)
        
        # Multi-head cross-attention
        # Seeds attend to input sequence (seeds=Q, input=K,V)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
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
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Pool input sequence using attention.
        
        Args:
            x: Input features, shape (batch_size, seq_len, embed_dim).
            return_attention: If True, also return attention weights.
        
        Returns:
            Pooled output, shape (batch_size, num_seeds, embed_dim).
            If return_attention=True, also returns attention weights
            of shape (batch_size, num_heads, num_seeds, seq_len).
        """
        batch_size = x.size(0)
        
        # Expand seeds for batch: (1, num_seeds, D) -> (B, num_seeds, D)
        queries = self.seeds.expand(batch_size, -1, -1)
        
        # Cross-attention: seeds attend to input sequence
        attn_output, attn_weights = self.attention(
            query=queries,
            key=x,
            value=x,
            need_weights=return_attention,
            average_attn_weights=False,  # Return per-head weights
        )
        
        # Residual connection and layer norm
        output = self.norm1(queries + attn_output)
        
        # Optional FFN with residual
        if self.ffn is not None:
            output = self.norm2(output + F.relu(self.ffn(output)))
        
        if return_attention:
            return output, attn_weights
        
        return output

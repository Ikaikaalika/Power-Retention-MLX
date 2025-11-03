"""
Large Language Model architecture using Power Retention.

This replaces traditional attention with Power Retention for linear-complexity
sequence processing, enabling efficient training on long contexts.
"""

import mlx.core as mx
import mlx.nn as nn
import sys
import os

# Add parent directory to path to import power_retention
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from power_retention import PowerRetention


class RetentionBlock(nn.Module):
    """
    Transformer-style block with Power Retention replacing attention.

    Architecture:
        x -> LayerNorm -> PowerRetention -> Residual
          -> LayerNorm -> FeedForward -> Residual
    """

    def __init__(
        self,
        dim: int,
        ffn_mult: int = 4,
        chunk_size: int = 128,
        dropout: float = 0.0,
        use_metal: bool = False,
    ):
        super().__init__()

        # Power Retention layer
        self.retention = PowerRetention(
            dim=dim,
            chunk_size=chunk_size,
            use_metal_kernels=use_metal
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_mult * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * dim, dim),
            nn.Dropout(dropout),
        )

    def __call__(self, x: mx.array, log_g: mx.array = None) -> mx.array:
        # Retention with pre-norm and residual
        x = x + self.retention(self.norm1(x), log_g=log_g)

        # FFN with pre-norm and residual
        x = x + self.ffn(self.norm2(x))

        return x


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) for better position encoding.
    Can be integrated into retention layers for long-context understanding.
    """

    def __init__(self, dim: int, max_seq_len: int = 32768, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # Precompute frequencies
        inv_freq = 1.0 / (theta ** (mx.arange(0, dim, 2).astype(mx.float32) / dim))
        t = mx.arange(max_seq_len, dtype=mx.float32)
        freqs = mx.outer(t, inv_freq)

        # Cache cos and sin
        self.cos_cached = mx.cos(freqs)
        self.sin_cached = mx.sin(freqs)

    def __call__(self, seq_len: int) -> tuple:
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


class RetentionLLM(nn.Module):
    """
    Complete Language Model using Power Retention.

    Args:
        vocab_size: Size of vocabulary
        dim: Model dimension (embedding size)
        num_layers: Number of retention blocks
        max_seq_len: Maximum sequence length
        ffn_mult: FFN hidden dimension multiplier
        chunk_size: Chunk size for retention processing
        dropout: Dropout probability
        use_metal: Whether to use Metal kernels (inference only)
        tie_embeddings: Whether to tie input/output embeddings
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int = 1024,
        num_layers: int = 12,
        max_seq_len: int = 4096,
        ffn_mult: int = 4,
        chunk_size: int = 256,
        dropout: float = 0.1,
        use_metal: bool = False,
        tie_embeddings: bool = True,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.tie_embeddings = tie_embeddings

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, dim)

        # Position embeddings (learnable)
        self.position_embedding = nn.Embedding(max_seq_len, dim)

        # Optional: RoPE for better position encoding
        self.rope = RotaryEmbedding(dim, max_seq_len)

        # Stack of retention blocks
        self.blocks = [
            RetentionBlock(
                dim=dim,
                ffn_mult=ffn_mult,
                chunk_size=chunk_size,
                dropout=dropout,
                use_metal=use_metal,
            )
            for _ in range(num_layers)
        ]

        # Final layer norm
        self.norm = nn.LayerNorm(dim)

        # Output head
        if tie_embeddings:
            # Share weights with token embedding
            self.output = None  # Will use token_embedding.weight
        else:
            self.output = nn.Linear(dim, vocab_size, bias=False)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def __call__(
        self,
        tokens: mx.array,
        log_g: mx.array = None,
        return_hidden: bool = False
    ) -> mx.array:
        """
        Forward pass.

        Args:
            tokens: Input token IDs [batch, seq_len]
            log_g: Optional gating values [batch, seq_len]
            return_hidden: Whether to return hidden states

        Returns:
            logits: Output logits [batch, seq_len, vocab_size]
            or (logits, hidden_states) if return_hidden=True
        """
        batch, seq_len = tokens.shape

        # Token embeddings
        x = self.token_embedding(tokens)

        # Add positional embeddings
        positions = mx.arange(seq_len)
        x = x + self.position_embedding(positions)

        # Apply dropout
        x = self.dropout(x)

        # Pass through retention blocks
        for block in self.blocks:
            x = block(x, log_g=log_g)

        # Final normalization
        x = self.norm(x)

        # Compute logits
        if self.tie_embeddings:
            # Project using transposed embedding weights
            logits = mx.matmul(x, self.token_embedding.weight.T)
        else:
            logits = self.output(x)

        if return_hidden:
            return logits, x
        return logits

    def generate(
        self,
        prompt_tokens: mx.array,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
    ) -> mx.array:
        """
        Generate text autoregressively.

        Args:
            prompt_tokens: Starting tokens [batch, prompt_len]
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            repetition_penalty: Penalty for repeating tokens

        Returns:
            generated_tokens: [batch, prompt_len + max_new_tokens]
        """
        batch = prompt_tokens.shape[0]
        tokens = prompt_tokens

        # Reset retention states for all blocks
        for block in self.blocks:
            block.retention.reset_state()

        for _ in range(max_new_tokens):
            # Get logits for last token
            logits = self(tokens)[:, -1, :]  # [batch, vocab_size]

            # Apply temperature
            logits = logits / temperature

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token_id in mx.unique(tokens):
                    logits[:, token_id] /= repetition_penalty

            # Top-k filtering
            if top_k > 0:
                top_k_logits, top_k_indices = mx.topk(logits, top_k)
                logits = mx.full_like(logits, float('-inf'))
                logits = mx.scatter(logits, top_k_indices, top_k_logits, axis=-1)

            # Convert to probabilities
            probs = mx.softmax(logits, axis=-1)

            # Top-p (nucleus) sampling
            if top_p < 1.0:
                sorted_probs, sorted_indices = mx.sort(probs, axis=-1)[::-1]
                cumulative_probs = mx.cumsum(sorted_probs, axis=-1)
                mask = cumulative_probs <= top_p
                filtered_probs = mx.where(mask, sorted_probs, 0.0)
                probs = mx.scatter(mx.zeros_like(probs), sorted_indices, filtered_probs, axis=-1)

            # Sample next token
            next_token = mx.random.categorical(mx.log(probs + 1e-10))

            # Append to sequence
            tokens = mx.concatenate([tokens, next_token.reshape(batch, 1)], axis=1)

        return tokens

    def reset_states(self):
        """Reset retention states in all blocks."""
        for block in self.blocks:
            block.retention.reset_state()


def create_model_config(size: str = "small") -> dict:
    """
    Create model configuration for different sizes.

    Args:
        size: One of 'tiny', 'small', 'medium', 'large', '7b'

    Returns:
        Configuration dictionary
    """
    configs = {
        "tiny": {
            "dim": 256,
            "num_layers": 6,
            "vocab_size": 50257,  # GPT-2 tokenizer
            "max_seq_len": 2048,
            "ffn_mult": 4,
            "chunk_size": 128,
        },
        "small": {
            "dim": 768,
            "num_layers": 12,
            "vocab_size": 50257,
            "max_seq_len": 4096,
            "ffn_mult": 4,
            "chunk_size": 256,
        },
        "medium": {
            "dim": 1024,
            "num_layers": 24,
            "vocab_size": 50257,
            "max_seq_len": 8192,
            "ffn_mult": 4,
            "chunk_size": 512,
        },
        "large": {
            "dim": 1536,
            "num_layers": 36,
            "vocab_size": 50257,
            "max_seq_len": 16384,
            "ffn_mult": 4,
            "chunk_size": 1024,
        },
        "7b": {
            "dim": 4096,
            "num_layers": 32,
            "vocab_size": 50257,
            "max_seq_len": 32768,
            "ffn_mult": 4,
            "chunk_size": 2048,
        },
    }

    if size not in configs:
        raise ValueError(f"Unknown size: {size}. Choose from {list(configs.keys())}")

    return configs[size]

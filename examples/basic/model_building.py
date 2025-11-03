"""
Complete example of building and training models with Power Retention on Apple Silicon.

This demonstrates:
1. Simple sequence model with Power Retention
2. Multi-layer transformer-style architecture
3. Training with MLX's autodiff
4. Inference and evaluation
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from power_retention import PowerRetention


# ==============================================================================
# Example 1: Simple Sequence Model
# ==============================================================================

class SimpleRetentionModel(nn.Module):
    """
    Basic model with Power Retention for sequence processing.
    Use case: Time series prediction, sequence classification
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, chunk_size: int = 128, use_metal: bool = False):
        super().__init__()

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Power Retention layer (replaces attention)
        # Note: use_metal_kernels=False for training (autodiff support)
        self.retention = PowerRetention(dim=hidden_dim, power=2, chunk_size=chunk_size, use_metal_kernels=use_metal)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        # Optional: Learnable gating
        self.gate_proj = nn.Linear(hidden_dim, 1)

    def __call__(self, x: mx.array, use_gating: bool = False) -> mx.array:
        """
        Args:
            x: Input sequence [batch, seq_len, input_dim]
            use_gating: Whether to use learned gating

        Returns:
            Output [batch, seq_len, output_dim]
        """
        # Project input
        h = self.input_proj(x)  # [batch, seq_len, hidden_dim]

        # Optional: Compute gating values
        log_g = None
        if use_gating:
            log_g = self.gate_proj(h).squeeze(-1)  # [batch, seq_len]

        # Apply Power Retention
        h = self.retention(h, log_g=log_g)

        # Project to output
        out = self.output_proj(h)

        return out


# ==============================================================================
# Example 2: Multi-Layer Retention Transformer
# ==============================================================================

class RetentionBlock(nn.Module):
    """Single transformer-style block with Power Retention."""
    def __init__(self, dim: int, chunk_size: int = 128, dropout: float = 0.1, use_metal: bool = False):
        super().__init__()

        self.retention = PowerRetention(dim=dim, power=2, chunk_size=chunk_size, use_metal_kernels=use_metal)
        self.norm1 = nn.LayerNorm(dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def __call__(self, x: mx.array, log_g: mx.array = None) -> mx.array:
        # Retention with residual connection
        h = x + self.retention(self.norm1(x), log_g=log_g)

        # FFN with residual connection
        h = h + self.ffn(self.norm2(h))

        return h


class RetentionTransformer(nn.Module):
    """
    Full transformer-style model using Power Retention.
    Use case: Language modeling, sequence-to-sequence tasks
    """
    def __init__(
        self,
        vocab_size: int,
        dim: int = 512,
        num_layers: int = 6,
        max_seq_len: int = 2048,
        chunk_size: int = 128,
        use_metal: bool = False,
    ):
        super().__init__()
        self.dim = dim

        # Token embeddings
        self.token_emb = nn.Embedding(vocab_size, dim)

        # Positional embeddings (learnable)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        # Stack of retention blocks
        self.blocks = [
            RetentionBlock(dim=dim, chunk_size=chunk_size, use_metal=use_metal)
            for _ in range(num_layers)
        ]

        # Final layer norm
        self.norm = nn.LayerNorm(dim)

        # Output head
        self.output = nn.Linear(dim, vocab_size)

    def __call__(self, tokens: mx.array) -> mx.array:
        """
        Args:
            tokens: Input token IDs [batch, seq_len]

        Returns:
            Logits [batch, seq_len, vocab_size]
        """
        batch, seq_len = tokens.shape

        # Embeddings
        x = self.token_emb(tokens)  # [batch, seq_len, dim]

        # Add positional embeddings
        positions = mx.arange(seq_len)
        x = x + self.pos_emb(positions)

        # Apply retention blocks
        for block in self.blocks:
            x = block(x)

        # Final normalization and projection
        x = self.norm(x)
        logits = self.output(x)

        return logits


# ==============================================================================
# Example 3: Training Functions
# ==============================================================================

def loss_fn(model: nn.Module, x: mx.array, y: mx.array) -> mx.array:
    """
    Compute loss for sequence prediction.

    Args:
        model: The model to train
        x: Input [batch, seq_len, input_dim] or [batch, seq_len] for tokens
        y: Target [batch, seq_len, output_dim] or [batch, seq_len] for tokens

    Returns:
        Loss scalar
    """
    logits = model(x)

    # For classification (e.g., language modeling)
    if len(y.shape) == 2:
        # Cross-entropy loss
        logits_flat = logits.reshape(-1, logits.shape[-1])
        targets_flat = y.reshape(-1)
        loss = nn.losses.cross_entropy(logits_flat, targets_flat)
    else:
        # MSE loss for regression
        loss = mx.mean((logits - y) ** 2)

    return loss


def train_step(
    model: nn.Module,
    optimizer: optim.Optimizer,
    x: mx.array,
    y: mx.array,
) -> float:
    """
    Single training step with gradient computation.

    Returns:
        Loss value
    """
    # Compute loss and gradients
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    loss, grads = loss_and_grad_fn(model, x, y)

    # Update parameters
    optimizer.update(model, grads)

    # Evaluate updated model (important for MLX lazy evaluation)
    mx.eval(model.parameters(), optimizer.state)

    return loss.item()


def train_model(
    model: nn.Module,
    train_data: list,
    num_epochs: int = 10,
    learning_rate: float = 1e-3,
    batch_size: int = 32,
):
    """
    Complete training loop.

    Args:
        model: Model to train
        train_data: List of (x, y) tuples
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        batch_size: Batch size
    """
    # Clear cache before training
    mx.clear_cache()

    # Initialize optimizer
    optimizer = optim.Adam(learning_rate=learning_rate)

    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0

        # Simple batching (in practice, use a DataLoader)
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size]

            # Stack batch
            x_batch = mx.stack([item[0] for item in batch])
            y_batch = mx.stack([item[1] for item in batch])

            # Training step
            loss = train_step(model, optimizer, x_batch, y_batch)

            total_loss += loss
            num_batches += 1

            # Periodic memory cleanup
            if num_batches % 5 == 0:
                mx.clear_cache()

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")


# ==============================================================================
# Example 4: Inference
# ==============================================================================

def generate_sequence(
    model: RetentionTransformer,
    prompt_tokens: mx.array,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
) -> mx.array:
    """
    Generate sequence autoregressively.

    Args:
        model: Trained transformer model
        prompt_tokens: Initial tokens [seq_len]
        max_new_tokens: Number of tokens to generate
        temperature: Sampling temperature

    Returns:
        Generated tokens [seq_len + max_new_tokens]
    """
    tokens = prompt_tokens.tolist()

    for _ in range(max_new_tokens):
        # Get logits for current sequence
        input_tokens = mx.array(tokens).reshape(1, -1)
        logits = model(input_tokens)

        # Get logits for last position
        next_logits = logits[0, -1, :] / temperature

        # Sample next token
        probs = mx.softmax(next_logits)
        next_token = mx.random.categorical(probs).item()

        tokens.append(next_token)

    return mx.array(tokens)


# ==============================================================================
# Demo Scripts
# ==============================================================================

def demo_simple_model():
    """Demo the simple retention model."""
    print("\n" + "="*60)
    print("Demo 1: Simple Retention Model (Training Mode)")
    print("="*60)

    # Create model - use_metal=False enables autodiff for training
    model = SimpleRetentionModel(
        input_dim=10,
        hidden_dim=32,
        output_dim=5,
        chunk_size=16,
        use_metal=False  # Pure MLX for training
    )

    # Generate dummy data
    batch_size, seq_len = 4, 20
    x = mx.random.normal((batch_size, seq_len, 10))
    y = mx.random.normal((batch_size, seq_len, 5))

    # Forward pass
    output = model(x, use_gating=True)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Compute loss
    loss = mx.mean((output - y) ** 2)
    print(f"Initial loss: {loss.item():.4f}")

    # Single training step
    optimizer = optim.Adam(learning_rate=1e-3)
    loss = train_step(model, optimizer, x, y)
    print(f"Loss after one step: {loss:.4f}")


def demo_transformer():
    """Demo the retention transformer."""
    print("\n" + "="*60)
    print("Demo 2: Retention Transformer (Training Mode)")
    print("="*60)

    # Create model
    model = RetentionTransformer(
        vocab_size=1000,
        dim=128,
        num_layers=4,
        max_seq_len=512,
        chunk_size=64,
        use_metal=False  # Pure MLX for training
    )

    # Generate dummy data
    batch_size, seq_len = 2, 50
    tokens = mx.random.randint(0, 1000, (batch_size, seq_len))

    # Forward pass
    logits = model(tokens)
    print(f"Input shape: {tokens.shape}")
    print(f"Output shape: {logits.shape}")

    # Generate text
    prompt = mx.random.randint(0, 1000, (10,))
    generated = generate_sequence(model, prompt, max_new_tokens=20, temperature=1.0)
    print(f"Generated {len(generated)} tokens from {len(prompt)} prompt tokens")


def demo_time_series():
    """Demo time series prediction."""
    print("\n" + "="*60)
    print("Demo 3: Time Series Prediction (Full Training)")
    print("="*60)

    # Create model
    model = SimpleRetentionModel(
        input_dim=1,
        hidden_dim=64,
        output_dim=1,
        chunk_size=32,
        use_metal=False  # Pure MLX for training
    )

    # Generate synthetic sine wave data
    num_samples = 50  # Reduced for memory efficiency
    train_data = []
    for _ in range(num_samples):
        t = mx.random.uniform(0, 10, (1,))
        x = mx.sin(mx.arange(30).reshape(30, 1) * 0.1 + t)  # Shorter sequences
        y = mx.sin(mx.arange(30).reshape(30, 1) * 0.1 + t + 0.1)  # Predict one step ahead
        train_data.append((x, y))

    print(f"Training on {len(train_data)} time series samples")

    # Train for a few epochs
    train_model(model, train_data, num_epochs=3, learning_rate=1e-3, batch_size=8)


if __name__ == "__main__":
    print("Power Retention Model Building Examples on Apple Silicon")
    print("\nNOTE: These demos use use_metal_kernels=False for training.")
    print("Metal kernels provide 2-3x speedup but don't support autodiff yet.")
    print("For inference-only, set use_metal_kernels=True for maximum performance.\n")

    # Run demos
    demo_simple_model()
    demo_transformer()
    demo_time_series()

    print("\n" + "="*60)
    print("All demos completed successfully!")
    print("="*60)
    print("\nTIP: For production inference, create models with use_metal=True")
    print("     for 2-3x faster forward passes using custom GPU kernels!")

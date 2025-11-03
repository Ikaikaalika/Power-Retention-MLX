# Quick Start: Building Models with Power Retention

Complete guide to building and training models with Power Retention on your Mac.

## Table of Contents
1. [Basic Setup](#basic-setup)
2. [Simple Model Example](#simple-model-example)
3. [Training Your Model](#training-your-model)
4. [Multi-Layer Transformer](#multi-layer-transformer)
5. [Metal Kernels: Training vs Inference](#metal-kernels-training-vs-inference)
6. [Common Use Cases](#common-use-cases)

---

## Basic Setup

```bash
# Install dependencies
pip install mlx mlx-nn

# Install Power Retention
pip install -e .

# Or just import directly from the repo
cd Power-Retention-MLX
python3
```

```python
import mlx.core as mx
import mlx.nn as nn
from power_retention import PowerRetention
```

---

## Simple Model Example

Create a basic sequence model with Power Retention:

```python
class MyModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Power Retention layer
        # use_metal_kernels=False for training (supports autodiff)
        self.retention = PowerRetention(
            dim=hidden_dim,
            chunk_size=128,
            use_metal_kernels=False
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def __call__(self, x):
        h = self.input_proj(x)
        h = self.retention(h)
        return self.output_proj(h)

# Create model
model = MyModel(input_dim=10, hidden_dim=64, output_dim=5)

# Forward pass
x = mx.random.normal((batch, seq_len, 10))
output = model(x)  # [batch, seq_len, 5]
```

---

## Training Your Model

Complete training loop with MLX's autodiff:

```python
import mlx.optimizers as optim

def train_step(model, optimizer, x, y):
    """Single training step with gradient computation."""

    # Define loss function
    def loss_fn(model, x, y):
        predictions = model(x)
        return mx.mean((predictions - y) ** 2)  # MSE loss

    # Compute loss and gradients
    loss_and_grad = nn.value_and_grad(model, loss_fn)
    loss, grads = loss_and_grad(model, x, y)

    # Update parameters
    optimizer.update(model, grads)

    # Evaluate (important for MLX lazy evaluation)
    mx.eval(model.parameters(), optimizer.state)

    return loss.item()

# Training loop
optimizer = optim.Adam(learning_rate=1e-3)

for epoch in range(num_epochs):
    for x_batch, y_batch in dataloader:
        loss = train_step(model, optimizer, x_batch, y_batch)
    print(f"Epoch {epoch}, Loss: {loss:.4f}")
```

---

## Multi-Layer Transformer

Build a full transformer-style model:

```python
class RetentionBlock(nn.Module):
    """Transformer block with Power Retention."""
    def __init__(self, dim, chunk_size=128):
        super().__init__()
        self.retention = PowerRetention(dim, chunk_size=chunk_size, use_metal_kernels=False)
        self.norm1 = nn.LayerNorm(dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def __call__(self, x):
        # Retention with residual
        h = x + self.retention(self.norm1(x))
        # FFN with residual
        h = h + self.ffn(self.norm2(h))
        return h

class MyTransformer(nn.Module):
    """Full transformer with Power Retention."""
    def __init__(self, vocab_size, dim=512, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.blocks = [RetentionBlock(dim) for _ in range(num_layers)]
        self.norm = nn.LayerNorm(dim)
        self.output = nn.Linear(dim, vocab_size)

    def __call__(self, tokens):
        x = self.embedding(tokens)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.output(x)

# Usage
model = MyTransformer(vocab_size=10000, dim=512, num_layers=6)
tokens = mx.random.randint(0, 10000, (2, 100))
logits = model(tokens)  # [2, 100, 10000]
```

---

## Metal Kernels: Training vs Inference

### Understanding the Two Modes

Power Retention has **two implementations**:

1. **Pure MLX** (`use_metal_kernels=False`):
   - Supports training with autodiff
   - Slightly slower
   - Use for training

2. **Metal Kernels** (`use_metal_kernels=True`):
   - 2-3x faster forward passes
   - Custom GPU kernels
   - Does NOT support autodiff yet
   - Use for inference

### When to Use Each

```python
# FOR TRAINING
model = MyModel(
    hidden_dim=64,
    use_metal=False  # Enables backpropagation
)
train_loop(model)  # Works!

# FOR INFERENCE (after training)
model = MyModel(
    hidden_dim=64,
    use_metal=True  # Fast Metal kernels
)
model.load_weights("trained_model.safetensors")
predictions = model(test_data)  # 2-3x faster!
```

### Switching After Training

```python
# Train with pure MLX
train_model = MyModel(hidden_dim=64, use_metal=False)
train_loop(train_model)
save_weights(train_model, "model.safetensors")

# Load for inference with Metal kernels
inference_model = MyModel(hidden_dim=64, use_metal=True)
load_weights(inference_model, "model.safetensors")
fast_predictions = inference_model(data)  # Fast!
```

---

## Common Use Cases

### 1. Time Series Prediction

```python
class TimeSeriesModel(nn.Module):
    def __init__(self, input_features, hidden_dim):
        super().__init__()
        self.proj_in = nn.Linear(input_features, hidden_dim)
        self.retention = PowerRetention(hidden_dim, use_metal_kernels=False)
        self.proj_out = nn.Linear(hidden_dim, input_features)

    def __call__(self, x):
        # x: [batch, time_steps, features]
        h = self.proj_in(x)
        h = self.retention(h)  # Linear-time processing
        return self.proj_out(h)

# Train on sine waves
model = TimeSeriesModel(input_features=1, hidden_dim=64)
optimizer = optim.Adam(learning_rate=1e-3)

for epoch in range(100):
    # Generate synthetic data
    t = mx.arange(100).reshape(1, 100, 1) * 0.1
    x = mx.sin(t)
    y = mx.sin(t + 0.1)  # Predict one step ahead

    loss = train_step(model, optimizer, x, y)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
```

### 2. Sequence Classification

```python
class SequenceClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
        )
        self.retention = PowerRetention(hidden_dim, use_metal_kernels=False)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def __call__(self, x):
        # x: [batch, seq_len, input_dim]
        h = self.encoder(x)
        h = self.retention(h)

        # Global average pooling
        h = mx.mean(h, axis=1)  # [batch, hidden_dim]

        return self.classifier(h)  # [batch, num_classes]

# Usage
model = SequenceClassifier(input_dim=128, hidden_dim=256, num_classes=10)
```

### 3. Language Model

```python
class LanguageModel(nn.Module):
    def __init__(self, vocab_size, dim, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(2048, dim)  # Max 2048 tokens

        self.blocks = [
            RetentionBlock(dim, chunk_size=256)
            for _ in range(num_layers)
        ]

        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size)

    def __call__(self, tokens):
        # tokens: [batch, seq_len]
        batch, seq_len = tokens.shape

        # Embeddings
        x = self.embedding(tokens)
        positions = mx.arange(seq_len)
        x = x + self.pos_emb(positions)

        # Process through retention blocks
        for block in self.blocks:
            x = block(x)

        # Output logits
        x = self.norm(x)
        return self.lm_head(x)

# Train on text data
model = LanguageModel(vocab_size=50000, dim=512, num_layers=8)

def train_language_model(model, text_data):
    optimizer = optim.Adam(learning_rate=3e-4)

    for epoch in range(num_epochs):
        for tokens in text_data:
            # Prepare inputs and targets
            inputs = tokens[:, :-1]   # All but last
            targets = tokens[:, 1:]   # All but first

            # Training step with cross-entropy loss
            def loss_fn(model, x, y):
                logits = model(x)
                logits_flat = logits.reshape(-1, logits.shape[-1])
                targets_flat = y.reshape(-1)
                return nn.losses.cross_entropy(logits_flat, targets_flat)

            loss_and_grad = nn.value_and_grad(model, loss_fn)
            loss, grads = loss_and_grad(model, inputs, targets)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
```

### 4. Recurrent Inference (Streaming)

For real-time applications, use recurrent mode:

```python
# Initialize model for streaming
model = PowerRetention(dim=64, use_metal_kernels=True)  # Fast inference
model.reset_state()  # Clear internal state

# Process tokens one at a time
outputs = []
for token in input_sequence:
    output = model.inference_step(token)
    outputs.append(output)

# Or process with gating
for token, gate_value in zip(input_sequence, gate_values):
    output = model.inference_step(token, log_g=gate_value)
    outputs.append(output)
```

---

## Performance Tips

### 1. Chunk Size Tuning

```python
# Smaller chunks: Lower memory, slightly slower
retention = PowerRetention(dim=64, chunk_size=32)

# Larger chunks: Higher memory, faster
retention = PowerRetention(dim=64, chunk_size=512)

# Default (good balance)
retention = PowerRetention(dim=64, chunk_size=128)
```

### 2. Memory Management

```python
import mlx.core as mx

# Clear cache periodically during training
for epoch in range(num_epochs):
    train_epoch()
    mx.clear_cache()  # Free unused memory
```

### 3. Batch Size Selection

```python
# Power Retention has O(n) complexity
# You can use larger sequences than attention!

# Attention: batch=32, seq=512 (limited by O(n²))
# Retention: batch=32, seq=2048 (same memory!)

model = PowerRetention(dim=128, chunk_size=256)
x = mx.random.normal((32, 2048, 128))  # Works great!
```

---

## Next Steps

- See [model_example.py](model_example.py) for complete working examples
- Check [METAL_KERNELS.md](METAL_KERNELS.md) for kernel implementation details
- Read the [main README](README.md) for more information

## Troubleshooting

**Problem**: Training fails with "Not implemented for CustomKernel"
**Solution**: Set `use_metal_kernels=False` for training mode

**Problem**: Out of memory during training
**Solution**: Reduce batch size, chunk size, or call `mx.clear_cache()` periodically

**Problem**: Slow inference
**Solution**: Set `use_metal_kernels=True` after training for 2-3x speedup

**Problem**: NaN values in output
**Solution**: Check learning rate (try 1e-4), add gradient clipping, or normalize inputs

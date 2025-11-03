"""
Quick Start: Train a Tiny LLM with Power Retention

This script demonstrates the complete training pipeline on synthetic data.
Perfect for testing and understanding the system before scaling up.

Usage:
    python quick_start.py
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import sys
import os
from pathlib import Path
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from llm_training.models import RetentionLLM, create_model_config
from llm_training.data import create_synthetic_data
from llm_training.training.utils import (
    save_checkpoint,
    TrainingLogger,
    compute_perplexity,
    format_time
)


def train_step(model, optimizer, batch):
    """Single training step with gradient computation."""

    def loss_fn(model, input_ids, labels):
        # Forward pass
        logits = model(input_ids)  # [batch, seq, vocab]

        # Compute cross-entropy loss
        # Reshape for loss computation
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.reshape(-1, vocab_size)
        labels_flat = labels.reshape(-1)

        # Mask padding tokens (-100)
        mask = labels_flat != -100
        logits_masked = logits_flat[mask]
        labels_masked = labels_flat[mask]

        # Cross-entropy loss
        loss = nn.losses.cross_entropy(logits_masked, labels_masked)
        return loss

    # Compute loss and gradients
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    loss, grads = loss_and_grad_fn(model, batch['input_ids'], batch['labels'])

    # Update parameters
    optimizer.update(model, grads)

    # Evaluate (important for MLX's lazy evaluation)
    mx.eval(model.parameters(), optimizer.state)

    return loss.item()


def main():
    print("="*60)
    print("Quick Start: Training Tiny LLM with Power Retention")
    print("="*60)
    print()

    # Configuration
    config = create_model_config("tiny")  # 256dim, 6 layers, ~23M params
    num_steps = 100
    batch_size = 4
    seq_len = 128
    learning_rate = 3e-4
    log_interval = 10
    save_interval = 50

    print("Configuration:")
    print(f"  Model: {config['num_layers']} layers, {config['dim']} dim")
    print(f"  Params: ~23M")
    print(f"  Steps: {num_steps}")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Learning rate: {learning_rate}")
    print()

    # Create model
    print("Creating model...")
    model = RetentionLLM(
        **config,
        use_metal=False  # Training mode (supports gradients)
    )
    print(f"✓ Model created with {sum(p.size for p in model.parameters())} parameters")
    print()

    # Create optimizer
    optimizer = optim.Adam(learning_rate=learning_rate)

    # Create logger
    logger = TrainingLogger(log_dir="logs", log_interval=log_interval)

    # Generate synthetic training data
    print("Generating synthetic training data...")
    total_samples = num_steps * batch_size
    data = create_synthetic_data(num_samples=total_samples, seq_len=seq_len)

    # Split into batches
    batches = []
    for i in range(0, total_samples, batch_size):
        batch = {
            'input_ids': data['input_ids'][i:i+batch_size],
            'labels': data['labels'][i:i+batch_size],
        }
        batches.append(batch)

    print(f"✓ Created {len(batches)} batches")
    print()

    # Training loop
    print("Starting training...")
    print("-"*60)

    start_time = time.time()
    best_loss = float('inf')

    for step, batch in enumerate(batches):
        step_start = time.time()

        # Training step
        loss = train_step(model, optimizer, batch)

        # Update best loss
        if loss < best_loss:
            best_loss = loss

        # Compute metrics
        perplexity = compute_perplexity(loss)
        step_time = time.time() - step_start

        # Log metrics
        logger.log(step, {
            "loss": loss,
            "perplexity": perplexity,
            "learning_rate": learning_rate,
            "step_time": step_time
        })

        # Save checkpoint
        if (step + 1) % save_interval == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=0,
                step=step + 1,
                loss=loss,
                save_dir="checkpoints/quick_start",
                config=config
            )

    # Training complete
    total_time = time.time() - start_time

    print("-"*60)
    print()
    print("Training Complete!")
    print("="*60)
    print(f"Total time: {format_time(total_time)}")
    print(f"Final loss: {loss:.4f}")
    print(f"Final perplexity: {perplexity:.2f}")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Avg step time: {total_time/num_steps:.3f}s")
    print()

    # Test generation
    print("Testing text generation...")
    print("-"*60)

    # Reset retention states
    model.reset_states()

    # Generate from random prompt
    prompt = mx.random.randint(0, config['vocab_size'], (1, 10))
    print(f"Prompt tokens: {prompt.tolist()[0]}")

    generated = model.generate(
        prompt,
        max_new_tokens=20,
        temperature=1.0,
        top_k=50
    )

    print(f"Generated tokens: {generated.tolist()[0]}")
    print()

    # Summary
    summary = logger.get_summary()
    print("Training Summary:")
    print(f"  Total steps: {summary['total_steps']}")
    print(f"  Final loss: {summary['final_loss']:.4f}")
    print(f"  Best loss: {summary['min_loss']:.4f}")
    print(f"  Avg loss: {summary['avg_loss']:.4f}")
    print()

    print("Checkpoints saved to: checkpoints/quick_start/")
    print("Logs saved to: logs/")
    print()
    print("="*60)
    print("Next Steps:")
    print("1. Try training on real data with data/data_processor.py")
    print("2. Scale up to 'small' or 'medium' model size")
    print("3. Run longer training (1000+ steps)")
    print("4. Fine-tune on instruction data")
    print("="*60)


if __name__ == "__main__":
    main()

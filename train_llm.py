#!/usr/bin/env python3
"""
Train a Language Model with Power Retention

Quick start script to train an LLM on your Mac using Power Retention.
Uses synthetic data by default for immediate testing, with optional real data support.

Usage:
    python3 train_llm.py                    # Train with synthetic data (fast)
    python3 train_llm.py --real-data        # Train with real data (requires dependencies)
    python3 train_llm.py --model large      # Use larger model
    python3 train_llm.py --steps 10000      # Train for more steps
"""

import sys
import os
import time
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

print("="*70)
print("🚀 Power Retention LLM Training")
print("="*70)
print()

# Import LLM components
try:
    from llm.models import RetentionLLM, create_model_config
    print("✓ LLM models loaded")
except ImportError as e:
    print(f"❌ Error importing LLM models: {e}")
    print("\nTrying alternative import...")
    try:
        sys.path.insert(0, str(Path(__file__).parent / "llm"))
        from models import RetentionLLM, create_model_config
        print("✓ LLM models loaded (alternative path)")
    except ImportError:
        print(f"❌ Could not import LLM models")
        print("Make sure you're in the Power-Retention-MLX directory")
        sys.exit(1)

try:
    from llm.data import DataProcessor, DataConfig, create_synthetic_data
    print("✓ LLM data modules loaded")
except ImportError as e:
    print(f"❌ Error importing LLM data: {e}")
    print("\nTrying alternative import...")
    try:
        from data import DataProcessor, DataConfig, create_synthetic_data
        print("✓ LLM data modules loaded (alternative path)")
    except ImportError:
        print(f"❌ Could not import LLM data modules")
        print("Make sure you're in the Power-Retention-MLX directory")
        sys.exit(1)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train LLM with Power Retention")

    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default="tiny",
        choices=["tiny", "small", "medium", "large", "7b"],
        help="Model size (default: tiny for quick testing)"
    )

    # Training configuration
    parser.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Number of training steps (default: 500)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size (default: 2)"
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=128,
        help="Sequence length (default: 128)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate (default: 3e-4)"
    )

    # Data configuration
    parser.add_argument(
        "--real-data",
        action="store_true",
        help="Use real data instead of synthetic (requires transformers/datasets)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="fineweb-edu",
        help="Dataset to use with --real-data (default: fineweb-edu)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10000,
        help="Number of samples to load (default: 10000)"
    )

    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/llm_training",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Steps between logging (default: 10)"
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=100,
        help="Steps between checkpoints (default: 100)"
    )

    return parser.parse_args()


def train_step(model, optimizer, batch):
    """Single training step with gradient computation."""

    def loss_fn(model, input_ids, labels):
        # Forward pass
        logits = model(input_ids)

        # Compute cross-entropy loss
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.reshape(-1, vocab_size)
        labels_flat = labels.reshape(-1)

        # Cross-entropy loss (MLX handles -100 labels internally)
        loss = nn.losses.cross_entropy(logits_flat, labels_flat, reduction='mean')
        return loss

    # Compute loss and gradients
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    loss, grads = loss_and_grad_fn(model, batch['input_ids'], batch['labels'])

    # Update parameters
    optimizer.update(model, grads)

    # Evaluate (important for MLX's lazy evaluation)
    mx.eval(model.parameters(), optimizer.state)

    return loss.item()


def format_time(seconds):
    """Format seconds as human-readable time."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def save_checkpoint(model, optimizer, step, loss, output_dir):
    """Save model checkpoint."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save weights
    weights_file = output_path / f"model_step{step}.safetensors"
    mx.save_safetensors(str(weights_file), dict(model.parameters()))

    # Save metadata
    import json
    meta_file = output_path / f"model_step{step}_meta.json"
    with open(meta_file, 'w') as f:
        json.dump({
            "step": step,
            "loss": float(loss),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)

    # Update latest
    latest_file = output_path / "latest.txt"
    with open(latest_file, 'w') as f:
        f.write(f"model_step{step}")

    print(f"    💾 Checkpoint saved: {weights_file.name}")


def prepare_synthetic_data(num_samples, batch_size, seq_len, vocab_size):
    """Prepare synthetic training data."""
    print(f"  Generating {num_samples} synthetic samples...")

    data = create_synthetic_data(num_samples=num_samples, seq_len=seq_len)

    # Create batches
    batches = []
    for i in range(0, num_samples, batch_size):
        if i + batch_size > num_samples:
            break

        batch = {
            'input_ids': data['input_ids'][i:i+batch_size],
            'labels': data['labels'][i:i+batch_size],
        }
        batches.append(batch)

    print(f"  ✓ Created {len(batches)} batches")
    return batches


def prepare_real_data(dataset_name, num_samples, batch_size, seq_len):
    """Prepare real training data."""
    print(f"  Loading {dataset_name}...")

    try:
        processor = DataProcessor(DataConfig(
            max_seq_len=seq_len,
            batch_size=batch_size
        ))

        # Load and prepare data
        data_iter = processor.prepare_training_data(
            dataset_name=dataset_name,
            num_samples=num_samples,
            filter_quality=True
        )

        # Collect batches
        batches = []
        for batch in data_iter:
            batches.append(batch)
            if len(batches) * batch_size >= num_samples:
                break

        print(f"  ✓ Loaded {len(batches)} batches from {dataset_name}")
        return batches

    except Exception as e:
        print(f"  ❌ Error loading real data: {e}")
        print("  💡 Falling back to synthetic data")
        vocab_size = 50257  # GPT-2 vocab size
        return prepare_synthetic_data(num_samples, batch_size, seq_len, vocab_size)


def main():
    """Main training function."""
    args = parse_args()

    print("Configuration:")
    print(f"  Model size: {args.model}")
    print(f"  Training steps: {args.steps}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Sequence length: {args.seq_len}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Data: {'Real' if args.real_data else 'Synthetic'}")
    print()

    # Create model
    print("🏗️  Creating model...")
    config = create_model_config(args.model)

    # Update config with args
    config['max_seq_len'] = args.seq_len

    model = RetentionLLM(**config, use_metal=False)  # Training mode

    # Count parameters
    def count_params(params):
        if isinstance(params, dict):
            return sum(count_params(v) for v in params.values())
        elif isinstance(params, (list, tuple)):
            return sum(count_params(v) for v in params)
        elif hasattr(params, 'size'):
            return params.size
        else:
            return 0

    num_params = count_params(model.parameters())
    print(f"  ✓ Model created: {args.model}")
    print(f"  ✓ Parameters: {num_params:,}")
    print(f"  ✓ Dimensions: {config['dim']}")
    print(f"  ✓ Layers: {config['num_layers']}")
    print()

    # Create optimizer
    print("⚙️  Setting up optimizer...")
    optimizer = optim.Adam(learning_rate=args.learning_rate)
    print(f"  ✓ Adam optimizer (lr={args.learning_rate})")
    print()

    # Prepare data
    print("📊 Preparing training data...")

    if args.real_data:
        batches = prepare_real_data(
            args.dataset,
            args.num_samples,
            args.batch_size,
            args.seq_len
        )
    else:
        batches = prepare_synthetic_data(
            args.num_samples,
            args.batch_size,
            args.seq_len,
            config['vocab_size']
        )

    if not batches:
        print("❌ No training data available!")
        return

    print()

    # Training loop
    print("🎯 Starting training...")
    print("="*70)
    print()

    start_time = time.time()
    losses = []

    for step in range(args.steps):
        step_start = time.time()

        # Get batch (cycle through data)
        batch = batches[step % len(batches)]

        # Training step
        loss = train_step(model, optimizer, batch)
        losses.append(loss)

        # Compute metrics
        step_time = time.time() - step_start
        avg_loss = sum(losses[-100:]) / len(losses[-100:])  # Last 100 steps

        # Log progress
        if (step + 1) % args.log_interval == 0:
            elapsed = time.time() - start_time
            tokens_per_sec = (step + 1) * args.batch_size * args.seq_len / elapsed

            print(f"Step {step+1:4d}/{args.steps} | "
                  f"Loss: {loss:6.4f} | "
                  f"Avg: {avg_loss:6.4f} | "
                  f"Time: {step_time:.3f}s | "
                  f"Tokens/s: {tokens_per_sec:.0f}")

        # Save checkpoint
        if (step + 1) % args.save_interval == 0:
            save_checkpoint(model, optimizer, step + 1, loss, args.output_dir)

        # Clear cache periodically
        if (step + 1) % 50 == 0:
            mx.clear_cache()

    # Training complete
    total_time = time.time() - start_time

    print()
    print("="*70)
    print("✅ Training Complete!")
    print("="*70)
    print()
    print("📊 Training Summary:")
    print(f"  Total time: {format_time(total_time)}")
    print(f"  Final loss: {losses[-1]:.4f}")
    print(f"  Best loss: {min(losses):.4f}")
    print(f"  Average loss: {sum(losses)/len(losses):.4f}")
    print(f"  Total tokens: {args.steps * args.batch_size * args.seq_len:,}")
    print()

    # Save final checkpoint
    print("💾 Saving final checkpoint...")
    save_checkpoint(model, optimizer, args.steps, losses[-1], args.output_dir)
    print()

    print(f"✓ Checkpoints saved to: {args.output_dir}/")
    print()

    # Test generation
    print("🎭 Testing text generation...")
    print("-"*70)

    model.reset_states()
    prompt = mx.random.randint(0, config['vocab_size'], (1, 10))

    print(f"Prompt tokens: {prompt.tolist()[0][:10]}")

    generated = model.generate(
        prompt,
        max_new_tokens=20,
        temperature=0.8
    )

    print(f"Generated tokens: {generated.tolist()[0][10:30]}")
    print()
    print("-"*70)
    print()

    print("🎉 Training pipeline complete!")
    print()
    print("Next steps:")
    print("  1. Fine-tune on instruction data")
    print("  2. Evaluate on benchmarks")
    print("  3. Use Metal kernels for faster inference")
    print("  4. Scale up model size")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print("Checkpoint saved at last save interval")
    except Exception as e:
        print(f"\n\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

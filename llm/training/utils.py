"""
Training utilities: checkpointing, logging, metrics.
"""

import mlx.core as mx
import mlx.nn as nn
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def save_checkpoint(
    model: nn.Module,
    optimizer: Any,
    epoch: int,
    step: int,
    loss: float,
    save_dir: str,
    config: Optional[Dict] = None,
):
    """
    Save model checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        step: Current step
        loss: Current loss
        save_dir: Directory to save checkpoint
        config: Model configuration
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Save model weights
    weights_path = save_path / f"checkpoint_epoch{epoch}_step{step}.safetensors"
    mx.savez(str(weights_path), **dict(model.parameters()))

    # Save optimizer state
    opt_path = save_path / f"optimizer_epoch{epoch}_step{step}.npz"
    if hasattr(optimizer, 'state'):
        mx.savez(str(opt_path), **optimizer.state)

    # Save metadata
    metadata = {
        "epoch": epoch,
        "step": step,
        "loss": float(loss),
        "timestamp": datetime.now().isoformat(),
        "config": config or {},
    }
    meta_path = save_path / f"checkpoint_epoch{epoch}_step{step}_meta.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Save latest checkpoint reference
    latest_path = save_path / "latest.txt"
    with open(latest_path, 'w') as f:
        f.write(f"checkpoint_epoch{epoch}_step{step}")

    logger.info(f"Saved checkpoint to {weights_path}")


def load_checkpoint(
    model: nn.Module,
    checkpoint_dir: str,
    checkpoint_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load model checkpoint.

    Args:
        model: Model to load weights into
        checkpoint_dir: Directory containing checkpoints
        checkpoint_name: Specific checkpoint name (or 'latest')

    Returns:
        Metadata dictionary
    """
    checkpoint_path = Path(checkpoint_dir)

    if checkpoint_name is None or checkpoint_name == "latest":
        # Load latest checkpoint
        latest_file = checkpoint_path / "latest.txt"
        if not latest_file.exists():
            raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

        with open(latest_file, 'r') as f:
            checkpoint_name = f.read().strip()

    # Load weights
    weights_file = checkpoint_path / f"{checkpoint_name}.safetensors"
    if not weights_file.exists():
        weights_file = checkpoint_path / f"{checkpoint_name}.npz"

    if not weights_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {weights_file}")

    weights = mx.load(str(weights_file))
    model.load_weights(list(weights.items()))

    # Load metadata
    meta_file = checkpoint_path / f"{checkpoint_name}_meta.json"
    if meta_file.exists():
        with open(meta_file, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}

    logger.info(f"Loaded checkpoint from {weights_file}")
    return metadata


class TrainingLogger:
    """Simple training logger."""

    def __init__(self, log_dir: str, log_interval: int = 10):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.log_interval = log_interval
        self.logs = []

        # Create log file
        self.log_file = self.log_dir / f"train_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

    def log(self, step: int, metrics: Dict[str, float]):
        """Log metrics."""
        log_entry = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            **metrics
        }

        self.logs.append(log_entry)

        # Write to file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

        # Print to console
        if step % self.log_interval == 0:
            metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
            logger.info(f"Step {step}: {metrics_str}")

    def get_summary(self) -> Dict[str, Any]:
        """Get training summary statistics."""
        if not self.logs:
            return {}

        losses = [log['loss'] for log in self.logs if 'loss' in log]

        return {
            "total_steps": len(self.logs),
            "final_loss": losses[-1] if losses else None,
            "min_loss": min(losses) if losses else None,
            "avg_loss": sum(losses) / len(losses) if losses else None,
        }


def compute_perplexity(loss: float) -> float:
    """Compute perplexity from cross-entropy loss."""
    return mx.exp(mx.array(loss)).item()


def estimate_tokens_per_second(
    batch_size: int,
    seq_len: int,
    time_elapsed: float,
) -> float:
    """Estimate training throughput."""
    tokens = batch_size * seq_len
    return tokens / time_elapsed


def format_time(seconds: float) -> str:
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

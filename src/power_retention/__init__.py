"""
Power Retention for MLX

A linear-complexity alternative to attention mechanisms for transformers,
optimized for Apple Silicon with custom Metal GPU kernels.
"""

from .core import PowerRetention

__version__ = "0.1.0"
__all__ = ["PowerRetention"]

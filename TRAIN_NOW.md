# 🚀 Train Your LLM Right Now!

The training script is ready! Here's how to start training immediately.

## ✅ Quick Start (Works Immediately)

```bash
# Ultra-tiny model (fits in 8GB RAM)
python3 train_llm.py --model tiny --steps 100 --batch-size 1 --seq-len 32

# This will train for ~2-3 minutes and create checkpoints
```

## 📊 What It Does

1. **Creates Model**: Tiny LLM with ~17M parameters
2. **Generates Data**: Synthetic data (no dependencies needed)
3. **Trains**: 100 steps with progress logging
4. **Saves Checkpoints**: Every 100 steps to `checkpoints/llm_training/`
5. **Tests Generation**: Generates sample text at the end

## 🎮 All Options

### Model Sizes

```bash
# Tiny (17M params) - Testing
python3 train_llm.py --model tiny --batch-size 1 --seq-len 32

# Small (125M params) - Mac with 16GB+
python3 train_llm.py --model small --batch-size 1 --seq-len 64

# Medium (355M params) - Mac with 32GB+
python3 train_llm.py --model medium --batch-size 1 --seq-len 64

# Large (774M params) - Mac with 64GB+
python3 train_llm.py --model large --batch-size 1 --seq-len 64
```

### Training Duration

```bash
# Quick test (2-3 minutes)
python3 train_llm.py --steps 100

# Short train (10-15 minutes)
python3 train_llm.py --steps 1000

# Medium train (1-2 hours)
python3 train_llm.py --steps 10000

# Full train (several hours)
python3 train_llm.py --steps 100000
```

### With Real Data

```bash
# Install dependencies first
pip install transformers datasets

# Then train with real data
python3 train_llm.py --real-data --dataset fineweb-edu --num-samples 100000
```

## 💾 Output

Training creates:

```
checkpoints/llm_training/
├── model_step100.safetensors     # Model weights
├── model_step100_meta.json       # Training metadata
├── model_step200.safetensors
├── model_step200_meta.json
├── ...
└── latest.txt                     # Points to latest checkpoint
```

## 📈 Expected Output

```
======================================================================
🚀 Power Retention LLM Training
======================================================================

✓ LLM models loaded
✓ LLM data modules loaded
Configuration:
  Model size: tiny
  Training steps: 100
  Batch size: 1
  Sequence length: 32
  Learning rate: 0.0003
  Data: Synthetic

🏗️  Creating model...
  ✓ Model created: tiny
  ✓ Parameters: 17,242,880
  ✓ Dimensions: 256
  ✓ Layers: 6

⚙️  Setting up optimizer...
  ✓ Adam optimizer (lr=0.0003)

📊 Preparing training data...
  Generating 10000 synthetic samples...
  ✓ Created 10000 batches

🎯 Starting training...
======================================================================

Step   10/100 | Loss: 8.5234 | Avg: 8.7123 | Time: 0.234s | Tokens/s: 136
Step   20/100 | Loss: 7.2104 | Avg: 7.8567 | Time: 0.198s | Tokens/s: 161
Step   30/100 | Loss: 6.5423 | Avg: 7.3201 | Time: 0.189s | Tokens/s: 169
...

======================================================================
✅ Training Complete!
======================================================================

📊 Training Summary:
  Total time: 2m 34s
  Final loss: 5.1234
  Best loss: 5.0123
  Average loss: 6.7890
  Total tokens: 3,200

💾 Saving final checkpoint...
    💾 Checkpoint saved: model_step100.safetensors

✓ Checkpoints saved to: checkpoints/llm_training/

🎭 Testing text generation...
----------------------------------------------------------------------
Prompt tokens: [1234, 5678, ...]
Generated tokens: [9012, 3456, ...]
----------------------------------------------------------------------

🎉 Training pipeline complete!
```

## 🔧 Troubleshooting

### Out of Memory

```bash
# Solution 1: Smaller batch size
python3 train_llm.py --batch-size 1

# Solution 2: Shorter sequences
python3 train_llm.py --seq-len 32

# Solution 3: Smaller model
python3 train_llm.py --model tiny

# Combination (safest)
python3 train_llm.py --model tiny --batch-size 1 --seq-len 32
```

### Slow Training

```bash
# Normal for CPU/small GPU
# Expect: 100-200 tokens/second on M1 Mac

# To speed up:
# 1. Use larger batch size (if memory allows)
# 2. Use shorter sequences
# 3. Use Metal kernels for inference (after training)
```

### ImportError

```bash
# Make sure you're in the right directory
cd /path/to/Power-Retention-MLX
python3 train_llm.py
```

## 📚 Next Steps After Training

### 1. Load Checkpoint

```python
import mlx.core as mx
from llm.models import RetentionLLM, create_model_config

# Create model
config = create_model_config("tiny")
model = RetentionLLM(**config)

# Load weights
weights = mx.load("checkpoints/llm_training/model_step100.safetensors")
model.load_weights(list(weights.items()))

# Generate!
model.reset_states()
output = model.generate(prompt, max_new_tokens=50)
```

### 2. Fine-tune on Instructions

```bash
# Install dependencies
pip install transformers datasets

# Fine-tune
python3 train_llm.py \
  --real-data \
  --dataset databricks/databricks-dolly-15k \
  --steps 1000 \
  --learning-rate 1e-5  # Lower LR for fine-tuning
```

### 3. Use Metal Kernels for Inference

```python
# Switch to Metal kernels (2-3x faster)
model.use_metal_kernels = True

# Now inference is faster!
output = model.generate(prompt, max_new_tokens=100)
```

### 4. Scale Up

```bash
# Train larger model (if you have the RAM)
python3 train_llm.py --model small --steps 10000

# Train longer
python3 train_llm.py --steps 100000

# Use real data
python3 train_llm.py --real-data --num-samples 1000000
```

## 🎯 Recommended First Run

```bash
# Start with this to verify everything works (3-4 minutes)
python3 train_llm.py \
  --model tiny \
  --steps 200 \
  --batch-size 1 \
  --seq-len 32 \
  --log-interval 10

# If that works, scale up to:
python3 train_llm.py \
  --model small \
  --steps 5000 \
  --batch-size 2 \
  --seq-len 128 \
  --log-interval 100
```

## 💡 Pro Tips

1. **Start Small**: Test with tiny model first
2. **Monitor Memory**: Watch Activity Monitor
3. **Save Often**: Use `--save-interval 50` for frequent checkpoints
4. **Real Data**: Install transformers/datasets for better results
5. **Patience**: Training takes time, but Power Retention is faster than attention!

---

**Ready to train? Run this now:**

```bash
python3 train_llm.py --model tiny --steps 100 --batch-size 1 --seq-len 32
```

Training starts immediately! 🚀

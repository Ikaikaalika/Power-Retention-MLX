# LLM Training with Power Retention

Complete pipeline for training Large Language Models using Power Retention on Apple Silicon.

## Overview

This directory contains a production-ready system for:
1. **Pretraining**: Training from scratch on large text corpora
2. **Fine-tuning (SFT)**: Instruction tuning on task-specific data
3. **RLHF**: Alignment using reinforcement learning from human feedback

### Why Power Retention for LLMs?

- **Linear Complexity**: O(n) vs O(n²) for attention → train on 10x longer contexts
- **Fixed Memory**: State size doesn't grow with sequence length
- **Efficient on Mac**: Optimized for Apple Silicon with Metal kernels
- **Scalable**: Train 1B-7B models on your Mac, larger models on cloud

## Quick Start

### Installation

```bash
# Install dependencies
pip install mlx mlx-nn transformers datasets

# Optional: for quality filtering
pip install detoxify

# Optional: for RLHF
pip install trl peft
```

### 1. Train a Small Model (Testing)

```bash
cd llm_training
python examples/quick_start.py
```

This trains a tiny model (256dim, 6 layers) on synthetic data in ~5 minutes.

### 2. Pretrain from Scratch

```python
from models import RetentionLLM, create_model_config
from data import DataProcessor, get_dataset

# Create model
config = create_model_config("small")  # 768dim, 12 layers
model = RetentionLLM(**config, use_metal=False)  # Training mode

# Load data
processor = DataProcessor()
data_iter = processor.prepare_training_data(
    dataset_name="HuggingFaceFW/fineweb-edu",
    num_samples=1_000_000
)

# Train (see training/pretrain.py for full script)
for batch in data_iter:
    loss = train_step(model, optimizer, batch)
```

### 3. Fine-tune on Instructions

```python
from training import finetune

finetune(
    model_path="checkpoints/pretrained",
    dataset_name="databricks/databricks-dolly-15k",
    output_dir="checkpoints/finetuned"
)
```

## Architecture

### Model Sizes

```python
from models import create_model_config

# Available configs:
tiny   = create_model_config("tiny")    # 23M params, testing
small  = create_model_config("small")   # 125M params, Mac-friendly
medium = create_model_config("medium")  # 355M params, M1 Max+
large  = create_model_config("large")   # 774M params, M1 Ultra+
7b     = create_model_config("7b")      # 7B params, cloud recommended
```

### Components

```
llm_training/
├── models/
│   └── retention_llm.py      # LLM architecture
├── data/
│   ├── data_processor.py     # Data loading & processing
│   └── data_sources.py       # Curated dataset catalog
├── training/
│   ├── pretrain.py           # Pretraining script
│   ├── sft.py                # Supervised fine-tuning
│   ├── rlhf.py               # RLHF with PPO
│   └── utils.py              # Checkpointing, logging
├── configs/
│   └── training_configs.yaml # Training configurations
└── examples/
    ├── quick_start.py        # Simple training demo
    ├── pretrain_small.py     # Pretrain 125M model
    └── full_pipeline.py      # Complete training pipeline
```

## Data Sources

### Pretraining (High-Quality Curated)

| Dataset | Size | Quality | Domain | License |
|---------|------|---------|--------|---------|
| **FineWeb-Edu** | 1.3T tokens | High | General | ODC-By |
| **Dolma** | 3T tokens | High | General | ODC-By |
| **The Stack V2** | 900B tokens | High | Code | OpenRAIL-M |
| **Proof-Pile 2** | 55B tokens | High | Math | MIT |

Access via:
```python
from data import get_dataset, list_datasets

# Get specific dataset
dataset = get_dataset("fineweb-edu")
print(dataset.hf_path)  # "HuggingFaceFW/fineweb-edu"

# List all available
for name, ds in list_datasets("pretrain").items():
    print(f"{name}: {ds.size_tokens} tokens")
```

### Fine-tuning

| Dataset | Examples | Domain |
|---------|----------|--------|
| **Alpaca** | 52K | Instructions |
| **Dolly 15K** | 15K | Instructions |
| **OpenAssistant** | 88K | Conversations |
| **FLAN V2** | 20M | Multi-task |

### RLHF

| Dataset | Type | Quality |
|---------|------|---------|
| **HH-RLHF** | Preferences | High (Anthropic) |
| **UltraFeedback** | Preferences | High |
| **RewardBench** | Evaluation | High (AI2) |

## Training Pipeline

### Stage 1: Pretraining

**Goal**: Learn language understanding from raw text

```bash
python training/pretrain.py \
  --model_size small \
  --dataset fineweb-edu \
  --num_tokens 10B \
  --batch_size 8 \
  --seq_len 2048 \
  --learning_rate 3e-4 \
  --output_dir checkpoints/pretrained
```

**Time estimates** (on M1 Max):
- Tiny (23M): 10B tokens → ~6 hours
- Small (125M): 10B tokens → ~2 days
- Medium (355M): 10B tokens → ~5 days

### Stage 2: Supervised Fine-Tuning

**Goal**: Teach instruction-following

```bash
python training/sft.py \
  --model_path checkpoints/pretrained/latest \
  --dataset databricks-dolly-15k \
  --epochs 3 \
  --batch_size 4 \
  --learning_rate 1e-5 \
  --output_dir checkpoints/finetuned
```

**Time**: ~1-2 hours for 15K examples

### Stage 3: RLHF (Optional)

**Goal**: Align with human preferences

```bash
# 1. Train reward model
python training/rlhf.py train-reward \
  --dataset Anthropic/hh-rlhf \
  --output_dir checkpoints/reward_model

# 2. Run PPO
python training/rlhf.py ppo \
  --model_path checkpoints/finetuned/latest \
  --reward_model checkpoints/reward_model \
  --dataset allenai/reward-bench \
  --output_dir checkpoints/rlhf
```

**Time**: ~4-8 hours depending on iterations

## Performance Tips

### Memory Optimization

```python
# Use smaller batch size
batch_size = 2  # vs 8

# Use gradient accumulation
accumulation_steps = 4  # Effective batch_size = 2 * 4 = 8

# Clear cache periodically
import mlx.core as mx
mx.clear_cache()
```

### Speed Optimization

```python
# Use Metal kernels for inference (not training)
model = RetentionLLM(**config, use_metal=True)

# Increase chunk size
chunk_size = 512  # vs 128

# Use mixed precision (coming soon in MLX)
```

### Training on Cloud

For larger models (>1B params), use cloud GPUs:

```bash
# Example: RunPod with A6000
runpod create \
  --gpu A6000 \
  --image pytorch/pytorch:latest \
  --volume 100GB

# Then: upload code and run training scripts
```

**Cost estimates**:
- 1B model, 10B tokens: ~$50 (A6000, 20 hours)
- 7B model, 100B tokens: ~$500 (A100, 100 hours)

## Monitoring

### Training Metrics

```python
from training.utils import TrainingLogger

logger = TrainingLogger(log_dir="logs")

# Log during training
logger.log(step=100, metrics={
    "loss": 2.45,
    "perplexity": 11.59,
    "lr": 3e-4
})

# Get summary
summary = logger.get_summary()
print(f"Final loss: {summary['final_loss']}")
```

### Checkpoints

Checkpoints saved every N steps:
```
checkpoints/
├── checkpoint_epoch0_step1000.safetensors
├── checkpoint_epoch0_step1000_meta.json
├── optimizer_epoch0_step1000.npz
└── latest.txt  # Points to most recent
```

Load checkpoint:
```python
from training.utils import load_checkpoint

metadata = load_checkpoint(model, "checkpoints", "latest")
print(f"Resuming from epoch {metadata['epoch']}")
```

## Evaluation

### Perplexity

```python
from training.utils import compute_perplexity

loss = 2.5
perplexity = compute_perplexity(loss)  # ~12.18
# Lower is better; GPT-2 small: ~30, GPT-3: ~20
```

### Generation Quality

```python
model.reset_states()
prompt = tokenizer("The meaning of life is")
output = model.generate(
    prompt,
    max_new_tokens=50,
    temperature=0.7,
    top_p=0.9
)
print(tokenizer.decode(output))
```

## Troubleshooting

### Out of Memory

```python
# Reduce batch size
batch_size = 1

# Reduce sequence length
seq_len = 1024

# Use gradient checkpointing (if available)
model.gradient_checkpointing = True
```

### NaN Loss

```python
# Reduce learning rate
learning_rate = 1e-4  # vs 3e-4

# Add gradient clipping
mx.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Check data quality
# - Remove toxic/malformed samples
# - Verify tokenization
```

### Slow Training

```python
# Use Metal kernels (inference only)
model.use_metal_kernels = True

# Increase batch size (if memory allows)
batch_size = 16

# Profile bottlenecks
import time
start = time.time()
# ... training code ...
print(f"Step time: {time.time() - start:.2f}s")
```

## Advanced Topics

### Multi-GPU Training

MLX supports multi-GPU training on M1 Ultra:
```python
# Coming soon in MLX
```

### Curriculum Learning

Train on easier examples first:
```python
# Sort by length
data = sorted(data, key=lambda x: len(x['text']))

# Train in stages
train(data[:100000])  # Short sequences
train(data)           # All sequences
```

### Continual Learning

Update model without catastrophic forgetting:
```python
# Use EWC or similar
# Mix old and new data
# Lower learning rate
```

## Citation

If you use this code, please cite:

```bibtex
@software{power_retention_mlx,
  title = {Power Retention for MLX},
  year = {2025},
  url = {https://github.com/YOUR_USERNAME/power-retention-mlx}
}
```

## License

MIT License - See [LICENSE](../LICENSE) file.

## Resources

- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [Power Retention Paper](https://github.com/m-a-n-i-f-e-s-t/retention)
- [LLM Training Best Practices](https://github.com/Hannibal046/Awesome-LLM)
- [Dataset Catalog](https://huggingface.co/datasets)

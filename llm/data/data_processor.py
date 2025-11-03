"""
Data processing utilities for LLM training.

Handles:
- Loading datasets from Hugging Face
- Tokenization
- Quality filtering
- Batch preparation
"""

import mlx.core as mx
from dataclasses import dataclass
from typing import List, Optional, Dict, Iterator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Configuration for data processing."""
    tokenizer_name: str = "gpt2"
    max_seq_len: int = 2048
    batch_size: int = 8
    min_text_length: int = 50
    max_toxicity: float = 0.1
    num_workers: int = 4


class DataProcessor:
    """
    Process and prepare data for LLM training.

    Features:
    - Multi-source dataset loading
    - Quality filtering (toxicity, length)
    - Efficient tokenization
    - Batch preparation for MLX
    """

    def __init__(self, config: Optional[DataConfig] = None):
        self.config = config or DataConfig()

        # Lazy import heavy dependencies
        self._tokenizer = None
        self._detoxify = None

    @property
    def tokenizer(self):
        """Lazy load tokenizer."""
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.config.tokenizer_name
                )
                # Add pad token if missing
                if self._tokenizer.pad_token is None:
                    self._tokenizer.pad_token = self._tokenizer.eos_token
            except ImportError:
                logger.error("transformers not installed. Run: pip install transformers")
                raise
        return self._tokenizer

    @property
    def detoxify_model(self):
        """Lazy load toxicity detector."""
        if self._detoxify is None:
            try:
                from detoxify import Detoxify
                self._detoxify = Detoxify('original', device='cpu')
            except ImportError:
                logger.warning("detoxify not installed. Toxicity filtering disabled.")
                logger.warning("Install with: pip install detoxify")
                self._detoxify = None
        return self._detoxify

    def load_dataset(
        self,
        dataset_name: str,
        split: str = 'train',
        num_samples: Optional[int] = None,
        streaming: bool = True,
    ) -> Iterator[str]:
        """
        Load dataset from Hugging Face.

        Args:
            dataset_name: HF dataset name (e.g., 'HuggingFaceFW/fineweb')
            split: Dataset split
            num_samples: Number of samples to load (None = all)
            streaming: Use streaming mode for large datasets

        Yields:
            text: Text samples
        """
        try:
            from datasets import load_dataset
        except ImportError:
            logger.error("datasets not installed. Run: pip install datasets")
            raise

        logger.info(f"Loading dataset: {dataset_name}")

        try:
            dataset = load_dataset(
                dataset_name,
                split=split,
                streaming=streaming
            )

            if num_samples:
                dataset = dataset.take(num_samples)

            count = 0
            for example in dataset:
                # Handle different text field names
                text = None
                for key in ['text', 'content', 'article', 'document']:
                    if key in example:
                        text = example[key]
                        break

                if text:
                    count += 1
                    if count % 10000 == 0:
                        logger.info(f"Loaded {count} samples")
                    yield text

            logger.info(f"Finished loading {count} samples")

        except Exception as e:
            logger.error(f"Error loading dataset {dataset_name}: {e}")
            raise

    def filter_quality(self, texts: List[str]) -> List[str]:
        """
        Filter texts by quality metrics.

        Filters:
        - Minimum length
        - Toxicity (if detoxify available)
        - Duplicates (simple check)

        Args:
            texts: List of text samples

        Returns:
            filtered_texts: Quality-filtered texts
        """
        filtered = []
        seen_hashes = set()

        for text in texts:
            # Length filter
            if len(text) < self.config.min_text_length:
                continue

            # Simple deduplication
            text_hash = hash(text[:100])  # Hash first 100 chars
            if text_hash in seen_hashes:
                continue
            seen_hashes.add(text_hash)

            # Toxicity filter (if available)
            if self.detoxify_model is not None:
                try:
                    result = self.detoxify_model.predict(text[:512])  # Check first 512 chars
                    if result['toxicity'] > self.config.max_toxicity:
                        continue
                except Exception as e:
                    logger.warning(f"Toxicity check failed: {e}")

            filtered.append(text)

        logger.info(f"Filtered {len(texts)} -> {len(filtered)} samples")
        return filtered

    def tokenize_texts(
        self,
        texts: List[str],
        return_tensors: str = "mlx"
    ) -> Dict[str, mx.array]:
        """
        Tokenize texts for training.

        Args:
            texts: List of text samples
            return_tensors: 'mlx' or 'np'

        Returns:
            Dictionary with 'input_ids' and 'labels'
        """
        # Tokenize
        encoded = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.config.max_seq_len,
            padding='max_length',
            return_tensors='np'
        )

        if return_tensors == "mlx":
            # Convert to MLX arrays
            input_ids = mx.array(encoded['input_ids'], dtype=mx.int32)

            # Create labels (shifted input_ids for next-token prediction)
            labels = mx.array(encoded['input_ids'], dtype=mx.int32)
            # Shift labels left by 1
            labels = mx.concatenate([labels[:, 1:], mx.full((labels.shape[0], 1), -100, dtype=mx.int32)], axis=1)

            return {
                'input_ids': input_ids,
                'labels': labels,
                'attention_mask': mx.array(encoded['attention_mask'], dtype=mx.int32)
            }
        else:
            return encoded

    def prepare_training_data(
        self,
        dataset_name: str,
        num_samples: int = 100000,
        filter_quality: bool = True,
    ) -> Iterator[Dict[str, mx.array]]:
        """
        Complete pipeline: load -> filter -> tokenize -> batch.

        Args:
            dataset_name: HF dataset name
            num_samples: Number of samples to process
            filter_quality: Whether to apply quality filtering

        Yields:
            Batches ready for training
        """
        # Load data in chunks
        chunk_size = self.config.batch_size * 100
        chunk = []

        for text in self.load_dataset(dataset_name, num_samples=num_samples):
            chunk.append(text)

            if len(chunk) >= chunk_size:
                # Filter
                if filter_quality:
                    chunk = self.filter_quality(chunk)

                # Tokenize in batches
                for i in range(0, len(chunk), self.config.batch_size):
                    batch_texts = chunk[i:i + self.config.batch_size]
                    if len(batch_texts) < self.config.batch_size:
                        continue  # Skip incomplete batches

                    batch_data = self.tokenize_texts(batch_texts)
                    yield batch_data

                chunk = []

        # Process remaining
        if chunk:
            if filter_quality:
                chunk = self.filter_quality(chunk)

            for i in range(0, len(chunk), self.config.batch_size):
                batch_texts = chunk[i:i + self.config.batch_size]
                if len(batch_texts) == self.config.batch_size:
                    batch_data = self.tokenize_texts(batch_texts)
                    yield batch_data


def create_synthetic_data(num_samples: int = 1000, seq_len: int = 128) -> Dict[str, mx.array]:
    """
    Create synthetic data for testing.

    Args:
        num_samples: Number of samples
        seq_len: Sequence length

    Returns:
        Dictionary with random token IDs
    """
    vocab_size = 50257  # GPT-2 vocab size

    input_ids = mx.random.randint(0, vocab_size, (num_samples, seq_len))
    labels = mx.concatenate([input_ids[:, 1:], mx.full((num_samples, 1), -100, dtype=mx.int32)], axis=1)

    return {
        'input_ids': input_ids,
        'labels': labels,
    }

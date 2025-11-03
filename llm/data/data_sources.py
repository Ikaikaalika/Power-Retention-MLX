"""
Curated high-quality data sources for LLM training.

Data sources are organized by quality tier and purpose.
"""

from typing import Dict, List
from dataclasses import dataclass


@dataclass
class DataSource:
    """Metadata for a data source."""
    name: str
    hf_path: str
    description: str
    size_tokens: str
    quality_tier: str  # 'high', 'medium', 'low'
    domain: str  # 'general', 'code', 'math', 'books', etc.
    license: str


# Curated data sources for pretraining
PRETRAIN_SOURCES = {
    "fineweb": DataSource(
        name="FineWeb",
        hf_path="HuggingFaceFW/fineweb",
        description="High-quality deduplicated web crawl (2024)",
        size_tokens="630B",
        quality_tier="high",
        domain="general",
        license="ODC-By"
    ),
    "fineweb-edu": DataSource(
        name="FineWeb-Edu",
        hf_path="HuggingFaceFW/fineweb-edu",
        description="Educational subset of FineWeb",
        size_tokens="1.3T",
        quality_tier="high",
        domain="general",
        license="ODC-By"
    ),
    "dolma": DataSource(
        name="Dolma",
        hf_path="allenai/dolma",
        description="AI2 curated dataset (books, code, web)",
        size_tokens="3T",
        quality_tier="high",
        domain="general",
        license="ODC-By"
    ),
    "redpajama-v2": DataSource(
        name="RedPajama V2",
        hf_path="togethercomputer/RedPajama-Data-V2",
        description="Deduplicated multi-source dataset",
        size_tokens="30T",
        quality_tier="medium",
        domain="general",
        license="Apache-2.0"
    ),
    "the-stack-v2": DataSource(
        name="The Stack V2",
        hf_path="bigcode/the-stack-v2",
        description="Deduplicated code from GitHub",
        size_tokens="900B",
        quality_tier="high",
        domain="code",
        license="BigCode OpenRAIL-M"
    ),
    "proof-pile-2": DataSource(
        name="Proof-Pile 2",
        hf_path="EleutherAI/proof-pile-2",
        description="Mathematical proofs and formal mathematics",
        size_tokens="55B",
        quality_tier="high",
        domain="math",
        license="MIT"
    ),
    "c4": DataSource(
        name="C4",
        hf_path="allenai/c4",
        description="Colossal Clean Crawled Corpus",
        size_tokens="750B",
        quality_tier="medium",
        domain="general",
        license="ODC-By"
    ),
}

# Data sources for fine-tuning
FINETUNE_SOURCES = {
    "alpaca": DataSource(
        name="Alpaca",
        hf_path="tatsu-lab/alpaca",
        description="52K instruction-following examples",
        size_tokens="10M",
        quality_tier="high",
        domain="instruction",
        license="CC BY NC 4.0"
    ),
    "dolly-15k": DataSource(
        name="Dolly 15K",
        hf_path="databricks/databricks-dolly-15k",
        description="15K instruction-response pairs",
        size_tokens="3M",
        quality_tier="high",
        domain="instruction",
        license="CC BY SA 3.0"
    ),
    "oasst1": DataSource(
        name="OpenAssistant",
        hf_path="OpenAssistant/oasst1",
        description="Conversation trees for assistant training",
        size_tokens="20M",
        quality_tier="high",
        domain="conversation",
        license="Apache-2.0"
    ),
    "flan-v2": DataSource(
        name="FLAN V2",
        hf_path="conceptofmind/FLAN_2022",
        description="Multi-task instruction tuning collection",
        size_tokens="50M",
        quality_tier="high",
        domain="instruction",
        license="Apache-2.0"
    ),
}

# Data sources for RLHF
RLHF_SOURCES = {
    "hh-rlhf": DataSource(
        name="HH-RLHF",
        hf_path="Anthropic/hh-rlhf",
        description="Anthropic's helpful & harmless preference data",
        size_tokens="10M",
        quality_tier="high",
        domain="preferences",
        license="MIT"
    ),
    "reward-bench": DataSource(
        name="RewardBench",
        hf_path="allenai/reward-bench",
        description="Benchmark for reward models",
        size_tokens="5M",
        quality_tier="high",
        domain="preferences",
        license="Apache-2.0"
    ),
    "ultrafeedback": DataSource(
        name="UltraFeedback",
        hf_path="openbmb/UltraFeedback",
        description="Large-scale preference learning dataset",
        size_tokens="20M",
        quality_tier="high",
        domain="preferences",
        license="MIT"
    ),
}


def get_dataset(name: str) -> DataSource:
    """
    Get data source by name.

    Args:
        name: Dataset name (key from any of the source dicts)

    Returns:
        DataSource object

    Raises:
        ValueError if dataset not found
    """
    # Check all source dictionaries
    all_sources = {**PRETRAIN_SOURCES, **FINETUNE_SOURCES, **RLHF_SOURCES}

    if name not in all_sources:
        available = list(all_sources.keys())
        raise ValueError(f"Dataset '{name}' not found. Available: {available}")

    return all_sources[name]


def list_datasets(stage: str = "all") -> Dict[str, DataSource]:
    """
    List available datasets by training stage.

    Args:
        stage: 'pretrain', 'finetune', 'rlhf', or 'all'

    Returns:
        Dictionary of data sources
    """
    if stage == "pretrain":
        return PRETRAIN_SOURCES
    elif stage == "finetune":
        return FINETUNE_SOURCES
    elif stage == "rlhf":
        return RLHF_SOURCES
    elif stage == "all":
        return {**PRETRAIN_SOURCES, **FINETUNE_SOURCES, **RLHF_SOURCES}
    else:
        raise ValueError(f"Unknown stage: {stage}")


def get_recommended_mix(budget_tokens: str = "10B") -> List[tuple]:
    """
    Get recommended data mix for different token budgets.

    Args:
        budget_tokens: Total token budget ('1B', '10B', '100B', '1T')

    Returns:
        List of (dataset_name, proportion) tuples
    """
    mixes = {
        "1B": [  # Small model, focus on quality
            ("fineweb-edu", 0.6),
            ("the-stack-v2", 0.2),
            ("proof-pile-2", 0.1),
            ("dolma", 0.1),
        ],
        "10B": [  # Medium model, balanced mix
            ("fineweb", 0.5),
            ("the-stack-v2", 0.2),
            ("dolma", 0.15),
            ("proof-pile-2", 0.1),
            ("c4", 0.05),
        ],
        "100B": [  # Large model, diverse sources
            ("fineweb", 0.4),
            ("dolma", 0.2),
            ("the-stack-v2", 0.15),
            ("redpajama-v2", 0.15),
            ("proof-pile-2", 0.05),
            ("c4", 0.05),
        ],
        "1T": [  # Very large model, maximize diversity
            ("fineweb", 0.3),
            ("redpajama-v2", 0.3),
            ("dolma", 0.15),
            ("the-stack-v2", 0.15),
            ("c4", 0.05),
            ("proof-pile-2", 0.05),
        ],
    }

    if budget_tokens not in mixes:
        raise ValueError(f"Unknown budget: {budget_tokens}. Choose from {list(mixes.keys())}")

    return mixes[budget_tokens]

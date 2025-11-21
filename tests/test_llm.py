import mlx.core as mx
import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llm.models.retention_llm import RetentionLLM, create_model_config

@pytest.fixture
def tiny_config():
    config = create_model_config("ultra-tiny")
    config['vocab_size'] = 100 # Small vocab for testing
    config['dim'] = 32
    config['num_layers'] = 2
    config['max_seq_len'] = 64
    return config

def test_llm_creation_power2(tiny_config):
    tiny_config['power'] = 2
    model = RetentionLLM(**tiny_config)
    assert model.dim == 32
    assert len(model.blocks) == 2

def test_llm_creation_power1(tiny_config):
    tiny_config['power'] = 1
    model = RetentionLLM(**tiny_config)
    assert model.dim == 32
    assert len(model.blocks) == 2

def test_forward_pass_power2(tiny_config):
    tiny_config['power'] = 2
    model = RetentionLLM(**tiny_config)
    
    batch_size = 2
    seq_len = 10
    tokens = mx.random.randint(0, tiny_config['vocab_size'], (batch_size, seq_len))
    
    logits = model(tokens)
    assert logits.shape == (batch_size, seq_len, tiny_config['vocab_size'])

def test_forward_pass_power1(tiny_config):
    tiny_config['power'] = 1
    model = RetentionLLM(**tiny_config)
    
    batch_size = 2
    seq_len = 10
    tokens = mx.random.randint(0, tiny_config['vocab_size'], (batch_size, seq_len))
    
    logits = model(tokens)
    assert logits.shape == (batch_size, seq_len, tiny_config['vocab_size'])

def test_generation(tiny_config):
    tiny_config['power'] = 1
    model = RetentionLLM(**tiny_config)
    
    prompt = mx.array([[1, 2, 3]])
    generated = model.generate(prompt, max_new_tokens=5)
    
    assert generated.shape[1] == 3 + 5

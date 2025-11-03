import mlx.core as mx
from power_retention import PowerRetention
import pytest

@pytest.fixture
def pr():
    return PowerRetention(dim=4, power=2, chunk_size=2)

def test_forward(pr):
    x = mx.random.normal((1, 3, 4))
    outputs = pr(x)
    assert outputs.shape == (1, 3, 4)
    assert not mx.any(mx.isnan(outputs))

def test_inference_step(pr):
    pr.reset_state()
    x = mx.random.normal((4,))
    y = pr.inference_step(x)
    assert y.shape == (4,)
    assert not mx.any(mx.isnan(y))

def test_reset_state(pr):
    pr(mx.random.normal((1, 3, 4)))
    old_state = mx.array(pr._state)
    pr.reset_state()
    assert mx.all(pr._state == 0)
    assert not mx.all(old_state == 0)

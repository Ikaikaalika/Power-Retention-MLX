import mlx.core as mx
from power_retention import PowerRetention

dim = 4
pr = PowerRetention(dim)

# Batch forward
x = mx.random.normal((2, 3, dim))  # batch=2, seq=3
log_g = mx.array([[-0.693, 0.0, 0.0], [0.0, 0.0, 0.0]])
outputs = pr(x, log_g)
print("Forward outputs:", outputs)

# Recurrent
pr.reset_state()
for b in range(2):
    for i in range(3):
        y = pr.inference_step(x[b, i], log_g[b, i])
        print(f"Batch {b}, Step {i} output:", y)

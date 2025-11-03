import mlx.core as mx
import mlx.nn as nn
from power_retention import PowerRetention
import random  # Mock env

class RLPolicy:
    def __init__(self, obs_dim, act_dim, hidden_dim):
        self.pr = PowerRetention(hidden_dim)
        self.w_obs = nn.Linear(obs_dim, hidden_dim)  # Embed obs
        self.w_out = nn.Linear(hidden_dim, act_dim)

    def sample_action(self, obs):
        emb = self.w_obs(mx.array(obs))
        y = self.pr.inference_step(emb.squeeze(0))
        logits = self.w_out(y)
        probs = mx.softmax(logits)
        return random.choices(range(len(probs)), weights=probs)[0]  # Mock sample

# Usage
policy = RLPolicy(4, 2, 8)
obs = [0.1, 0.2, 0.3, 0.4]
action = policy.sample_action(obs)
print("Sample action:", action)

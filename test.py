from model import GPT, GPTConfig
from model_pt import GPT as GPT_PT
import jax.numpy as jnp
import jax
import torch

model_config = GPTConfig()
model_config.vocab_size = 50257 # openai's model vocabulary
model_config.block_size = 1024  # openai's model block_size (i.e. input context length)

model_pt = GPT_PT(model_config)
model = GPT(model_config)

idx = jnp.ones((2, 10), dtype=jnp.int32)
key = jax.random.PRNGKey(0)
print(model.tabulate(key, idx, train=False, depth=1))
"""
Sample from a trained model
"""
import os
import pickle
import optax
import tiktoken
from model import GPTConfig, GPT
from model_pt import GPT as GPT_PT
from pathlib import Path
from flax.training import checkpoints, train_state
from flax import serialization
import jax.numpy as jnp
import jax
import torch
import utils

from utils import print_compiling

# -----------------------------------------------------------------------------
start = "I have a cat named" # or "<|endoftext|>" or whatever you like
num_samples = 10 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # higher temperature (up to 1) is more random, lower (down to 0) means more greedy
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
model_type = 'gpt2' # 'gpt2' or 'gpt2-medium' or 'gpt2-large' or 'gpt2-xl'
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

# model jax
override_args = dict(dropout=0.0)
model_jax, params = GPT.from_pretrained(model_type, override_args)

# model pytorch
model_pt = GPT_PT.from_pretrained(model_type, override_args)
model_pt.eval()

with utils.activate_logger():
    logits_jax, _ = model_jax.apply({'params': params}, jnp.ones((1, 1), dtype=jnp.int32) * 500, train=False)
    logits_pt, _ = model_pt(torch.ones((1, 1), dtype=torch.long) * 500)

print(utils.get_logs('h_0.mlp.c_proj', lambda x: x[:, :, :5]))
# print(utils.all_close('h_0.attn.c_attn', atol=1e-5))
print(utils.all_different(atol=1e-3))
# print(logits_jax)
# print(logits_pt)
exit()

enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# run generation
with torch.no_grad():
    for k in range(num_samples):
        y = model_pt.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
        print(decode(y[0].tolist()))
        print('---------------')
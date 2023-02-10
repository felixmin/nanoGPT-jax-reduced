"""
Sample from a trained model
"""
import os
import pickle
import tiktoken
from model import GPTConfig, GPT
from pathlib import Path
from flax import serialization
import jax.numpy as jnp
import jax
import orbax.checkpoint as orbax

from utils import print_compiling

# -----------------------------------------------------------------------------
out_dir = 'out'
start = "\n" # or "<|endoftext|>" or whatever you like
num_samples = 10 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # higher temperature (up to 1) is more random, lower (down to 0) means more greedy
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
dtype = 'bfloat16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------
checkpoint_path = Path(out_dir, 'checkpoint')
checkpoint_manager = orbax.CheckpointManager(
    checkpoint_path,
    checkpointers=orbax.Checkpointer(orbax.PyTreeCheckpointHandler()),
)
latest_step = checkpoint_manager.latest_step()
assert latest_step is not None, "No checkpoint found in out_dir!"

# model
checkpoint = checkpoint_manager.restore(latest_step, items=None)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
params = checkpoint['state']['params']

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = Path('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = meta_path.exists()
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
start_ids = encode(start)
x = jnp.array(start_ids, dtype=jnp.int32)[None]
key = jax.random.PRNGKey(seed)

@jax.jit
@print_compiling
def _sample(params, key, tokens) -> jax.Array:
    return model.generate(
        key, params, tokens, max_new_tokens=max_new_tokens, top_k=top_k, temperature=temperature)

def sample(params, key, tokens) -> str:
    tokens = _sample(params, key, tokens)
    return decode(tokens[0])

# run generation
for k in range(num_samples):
    step_key = jax.random.fold_in(key, k)
    sample_str = sample(params, step_key, x)
    print(sample_str)
    print('---------------')

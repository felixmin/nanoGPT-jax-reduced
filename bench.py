"""
A much shorter version of train.py for benchmarking
"""
from functools import partial
import os
from contextlib import nullcontext
from typing import Tuple
import jax
import jax.numpy as jnp
import numpy as np
import time
import torch
from model import GPTConfig, GPT
from utils import print_compiling
from flax.training.train_state import TrainState

# -----------------------------------------------------------------------------
batch_size = 8
block_size = 1024
seed = 1337
dtype = 'bfloat16' # 'float32' or 'bfloat16' or 'float16'
compile = True # use PyTorch 2.0 to compile the model to be faster
real_data = False
profile = False # use pytorch profiler, or just simple benchmarking?
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------
key = jax.random.PRNGKey(seed)
key_x, key_y = jax.random.split(key, 2)
# data loading init
if real_data:
    dataset = 'openwebtext'
    data_dir = os.path.join('data', dataset)
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    def get_batch(split) -> Tuple[jax.Array, jax.Array]:
        ...
        # data = train_data # note ignore split in benchmarking script
        # ix = torch.randint(len(data) - block_size, (batch_size,))
        # x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        # y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        # x, y = x.to(device), y.to(device)
        # return x, y
else:
    # alternatively, if fixed data is desired to not care about data loading
    x = jax.random.randint(key_x, (batch_size, block_size), 0, 50257, dtype=jnp.int32)
    y = jax.random.randint(key_y, (batch_size, block_size), 0, 50257, dtype=jnp.int32)
    get_batch = lambda split: (x, y)

# model init
gptconf = GPTConfig(
    block_size = block_size, # how far back does the model look? i.e. context size
    n_layer = 12, n_head = 12, n_embd = 768, # size of the model
    dropout = 0, # for determinism
)
model = GPT(gptconf)
params = model.init(key, jnp.ones((1, 1), dtype=jnp.int32), train=False)['params']
state = model.create_state(
    param=params, weight_decay=1e-2, learning_rate=1e-4, beta1=0.9, beta2=0.95)

@partial(jax.jit, donate_argnums=(0,))
@print_compiling
def train_step(state: TrainState, batch):
    inputs, labels = batch
    def loss_fn(params):
        logits, loss = state.apply_fn(
            {'params': params}, inputs, targets=labels, train=True)
        return loss
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(state.params)
    state = state.apply_gradients(grads=grad)
    return loss, state

def block_until_ready(a):
    return jax.tree_map(lambda x: x.block_until_ready(), a)

if profile:
    # useful docs on pytorch profiler:
    # - tutorial https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html
    # - api https://pytorch.org/docs/stable/profiler.html#torch.profiler.profile
    wait, warmup, active = 5, 5, 5
    num_steps = wait + warmup + active
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./bench_log'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True, # incurs an additional overhead, disable if not needed
        with_flops=True,
        with_modules=False, # only for torchscript models atm
    ) as prof:

        for k in range(num_steps):
            X, Y = get_batch('train')
            with ctx:
                logits, loss = model(X, Y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            lossf = loss.item()
            print(f"{k}/{num_steps} loss: {lossf:.4f}")

            prof.step() # notify the profiler at end of each step

else:

    # warmup
    batch = get_batch('train')
    loss, state = train_step(state, batch)

    # simple benchmarking
    for stage, num_steps in enumerate([10, 20]): # burnin, then benchmark
        t0 = time.time()
        for k in range(num_steps):
            batch = get_batch('train')
            loss, state = train_step(state, batch)
            print(f"{k}/{num_steps} loss: {loss:.4f}")
        loss, state = block_until_ready((loss, state))
        t1 = time.time()
        if stage == 1:
            print(f"time per iteration: {(t1-t0)/num_steps*1000:.4f}ms")

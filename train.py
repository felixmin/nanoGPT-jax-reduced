# %%
"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run in debug mode example:
$ python train.py --batch_size=32 --other=args

To run DDP on 4 gpus on one node, example:
$ torchrun --standalone --nproc_per_node=4 train.py
"""

import os
import time
import math
import pickle
from typing import Tuple
from flax.training import train_state
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import optax
import flax.training.checkpoints
import orbax.checkpoint as orbax
import tiktoken

from model import GPTConfig, GPT
from utils import print_compiling

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'shakespeare'
batch_size = 12
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-2
beta1 = 0.9
beta2 = 0.95
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' # 'float32' or 'bfloat16'
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# poor man's data loader, TODO evaluate need for actual DataLoader
data_dir = os.path.join('data', dataset)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = np.random.randint(len(data) - block_size, size=(batch_size,))
    x = np.stack([data[i:i+block_size].astype(np.int32) for i in ix])
    y = np.stack([data[i+1:i+1+block_size].astype(np.int32) for i in ix])

    # random samples
    # data = np.random.randint(0, block_size, size=(batch_size, 51), dtype=np.int32)
    # x = data[:, :-1]
    # y = data[:, 1:]
    # x, y = jax.device_put((x, y))

    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    vocab_size = meta['vocab_size']
    print(f"vocab_size = {vocab_size} (from {meta_path})")
else:
    print(f"vocab_size not found in {meta_path}, using GPT-2 default of 50257")
    vocab_size = 50257

# model init
model_args = dict(n_layer = n_layer, n_head = n_head, n_embd = n_embd, block_size = block_size, dropout = dropout, vocab_size = vocab_size)
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    # initialize weights
    variables = model.init(jax.random.PRNGKey(0), jnp.ones((1, 2), dtype=jnp.int32), train=False)
    params = variables['params']
elif init_from == 'resume':
    raise RuntimeError("resuming not supported yet")
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    for k, v in model_args.items():
        assert checkpoint_model_args[k] == v, "for now"
        # TODO: think through how passed in params should interact with checkpoint params
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    raise RuntimeError(f"Initializing from GPT-2 weights not supported yet")
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off and override the GPT sizing model args from the model config
    model_args['n_layer'] = model.config.n_layer
    model_args['n_head'] = model.config.n_head
    model_args['n_embd'] = model.config.n_embd
else:
    raise RuntimeError(f"init_from={init_from} not supported")
# crop down the model block size if desired
if block_size < model.config.block_size:
    params = model.crop_block_size(params, block_size)

# learning rate decay scheduler (cosine with warmup)
if decay_lr:
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=learning_rate,
        warmup_steps=warmup_iters, decay_steps=lr_decay_iters,
        end_value=min_lr,
    )
else:
    lr_schedule = learning_rate

# %%
# optimizer
tx = model.configure_optimizers(params, weight_decay, lr_schedule, (beta1, beta2))
state = train_state.TrainState.create(
    apply_fn=model.apply, params=params, tx=tx)

# %%
@partial(jax.jit, static_argnames=('train',))
@print_compiling
def forward(state, batch, *, train: bool):
    inputs, labels = batch
    rngs = {}
    if train and dropout > 0.0:
        rngs['dropout'] = jax.random.fold_in(
            jax.random.PRNGKey(0), state.step)
    return state.apply_fn(
        {'params': state.params}, 
         inputs, train=train, targets=labels, rngs=rngs)

@partial(jax.jit, donate_argnums=(0,))
@print_compiling
def train_step(state: train_state.TrainState, batch):
    def loss_fn(params):
        state_ = state.replace(params=params)
        logits, loss = forward(state_, batch, train=True)
        return loss
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(state.params)
    state = state.apply_gradients(grads=grad)
    return loss, state

def estimate_loss():
    out = {}
    for split in ['train', 'val']:
        losses = np.zeros(eval_iters)
        for k in range(eval_iters):
            batch = get_batch(split)
            logits, loss = forward(state, batch, train=False)
            losses[k] = float(loss)
        out[split] = losses.mean()
    return out

@jax.jit
@print_compiling
def _sample(params, key, tokens) -> jax.Array:
    return model.generate(key, params, tokens, max_new_tokens=10)

tokenizer = tiktoken.get_encoding("gpt2")

def sample(params, key, tokens) -> str:
    tokens = _sample(params, key, tokens)
    return tokenizer.decode(tokens[0])


# logging
if wandb_log:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# %%
val_batch = get_batch('val')
# training loop
t0 = time.time()
checkpointer = orbax.Checkpointer(orbax.PyTreeCheckpointHandler())
while True:
    if iter_num % eval_interval == 0:
        print("evaluating...")
        sample_str = sample(state.params, jax.random.PRNGKey(0), tokens=val_batch[0][0:1,:5])
        print(f"sample: {sample_str}")
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "loss/train": losses['train'],
                "loss/val": losses['val'],
                "lr": float(lr_schedule(iter_num)) if callable(lr_schedule) else lr_schedule,
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                print(f"saving checkpoint to {out_dir}")
                flax.training.checkpoints.save_checkpoint(
                    ckpt_dir=os.path.join(out_dir, 'checkpoint'),
                    target={
                        'state': state,
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': config,
                    },
                    step=iter_num,
                    orbax_checkpointer=checkpointer,
                )
    if iter_num == 0 and eval_only:
        break

    loss, state = train_step(state, get_batch('train'))

    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0:
        lossf = loss.item() # loss as float. TODO CPU-GPU sync: profile, make sure not slow af
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
    iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

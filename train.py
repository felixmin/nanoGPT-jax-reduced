# %%
"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run in debug mode example:
$ python train.py --batch_size=32 --other=args

To run DDP on 4 gpus on one node, example:
$ torchrun --standalone --nproc_per_node=4 train.py
"""


from flax.training import train_state
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import orbax.checkpoint as orbax
import optax

from model import GPTConfig, GPT
from utils import print_compiling

from gpt_jax import GPTLanguageModel, generate, create_train_state

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
out_dir = 'out'
eval_interval = 200
eval_iters = 200
eval_only = False # if True, script exits right after the first eval

# data
dataset = 'shakespeare'
batch_size = 64
block_size = 256
# model
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+

# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 5000 # total number of training iterations

# sampling
max_new_tokens = 100 # number of tokens generated in each sample
temperature = 0.8 # higher temperature (up to 1) is more random, lower (down to 0) means more greedy
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

with open('tinyshakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string




# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = jnp.array(encode(text))
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = np.random.randint(len(data) - block_size, size=(batch_size,))
    x = jnp.stack([data[i:i+block_size] for i in ix])
    y = jnp.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

vocab_size = 65

# model init
model_args = dict(n_layer = n_layer, n_head = n_head, n_embd = n_embd, block_size = block_size, dropout = dropout, vocab_size = vocab_size)

# init a new model from scratch
print("Initializing a new model from scratch")
#gptconf = GPTConfig(**model_args)
#model = GPT(gptconf)
model = GPTLanguageModel()

# initialize weights
xb, yb = get_batch('train')

state = create_train_state(jax.random.PRNGKey(234), model, xb)
params = state.params


#@partial(jax.jit, static_argnames=('train',))
#@print_compiling
def forward(state: train_state, xb, *, train: bool):
    rngs = {}
    # if train and dropout > 0.0:
    rngs['dropout'] = jax.random.fold_in(
        jax.random.PRNGKey(0), state.step)
    return model.apply(
        state.params, 
         xb, deterministic=not train, rngs=rngs)
    #return state.apply_fn(
    #    {'params': state.params}, 
    #     inputs, train=train, targets=labels, rngs=rngs)


def make_loss_fn(xb, yb):
    def loss_fn(params):
        state_ = state.replace(params=params)
        logits = forward(state_, xb, train=True)
        return optax.softmax_cross_entropy_with_integer_labels(logits, yb).mean()
    return loss_fn

#@partial(jax.jit, donate_argnums=(0,))
#@print_compiling
def train_step(state: train_state.TrainState, batch):
    xb, yb = batch
    loss_fn = make_loss_fn(xb, yb)
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(state.params)
    state = state.apply_gradients(grads=grad)
    return loss, state


def estimate_loss():
    out = {}
    for split in ['train', 'val']:
        losses = np.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            #logits = forward(state, xb, train=False)
            loss_fn = make_loss_fn(xb, yb)
            loss = loss_fn(state.params)
            losses[k] = float(loss)
        out[split] = losses.mean()
    return out

#@jax.jit
#@print_compiling
def _sample(params, key, tokens) -> jax.Array:
    key, gen_key = jax.random.split(key)
    return generate(gen_key, params, model, tokens, max_new_tokens=max_new_tokens)
    #return model.generate(
    #    key, params, tokens, max_new_tokens=max_new_tokens, top_k=top_k, temperature=temperature)


def sample(params, key, tokens) -> str:
    tokens = _sample(params, key, tokens)
    return decode(tokens[0].tolist())


# %%
val_batch = get_batch('val')

while True:
    if iter_num % eval_interval == 0:
        print("evaluating...")
        sample_str = sample(state.params, jax.random.PRNGKey(0), tokens=val_batch[0][0:1,:5])
        print(f"sample: {sample_str}")
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    if iter_num == 0 and eval_only:
        break

    loss, state = train_step(state, get_batch('train'))

    lossf = loss.item() # loss as float. TODO CPU-GPU sync: profile, make sure not slow af
    print(f"iter {iter_num}: loss {lossf:.4f}")
    
    iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

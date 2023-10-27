from flax.linen import MultiHeadDotProductAttention, SelfAttention, Embed, Sequential
from flax import linen as nn
import optax
from jax import numpy as jnp
from jax import random
import jax
from flax.training import train_state

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
#learning_rate = 3e-4
eval_iters = 200
#n_embd = 384 # dimension to which tokens and position is embedded
#n_head = 6 # number of parallel attention heads per block
#n_layer = 6 # number of transformer blocks
#dropout = 0.2
# ------------

key = random.PRNGKey(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('tinyshakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

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
def get_batch(key, split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    key, cur_key = random.split(key)
    ix = random.randint(cur_key, (batch_size,), 0, len(data) - block_size)
    x = jnp.stack([data[i:i+block_size] for i in ix])
    y = jnp.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

def estimate_loss(key, params):
    out = {}
    for split in ['train', 'val']:
        losses = jnp.zeros(eval_iters)
        for k in range(eval_iters):
            key, batch_key, loss_fn_key = random.split(key, 3)
            xb, yb = get_batch(batch_key, split)
            loss_fn = make_loss_fn(loss_fn_key, xb, yb)
            loss = loss_fn(params)
            losses = losses.at[k].set(loss)
        out[split] = losses.mean()
    return out

class CausalSelfAttention(nn.Module):
    n_head: int
    n_embd: int
    dropout: float

    def setup(self):
        assert self.n_embd % self.n_head == 0
        head_size = self.n_embd // self.n_head
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Dense(self.n_embd * 3)
        # output projection
        self.c_proj = nn.Dense(self.n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)


    def __call__(self, x: jax.Array, *, deterministic: bool) -> jax.Array:
        B, T, C = x.shape # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(3, axis=-1)
        q = q.reshape(B, T, self.n_head, C // self.n_head).swapaxes(1, 2) # (B, nh, T, hs)
        k = k.reshape(B, T, self.n_head, C // self.n_head).swapaxes(1, 2) # (B, nh, T, hs)
        v = v.reshape(B, T, self.n_head, C // self.n_head).swapaxes(1, 2) # (B, nh, T, hs)

        mask = jnp.tril(jnp.ones((T, T))).reshape((1, 1, T, T))
        
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.swapaxes(-2, -1)) * (1.0 / jnp.sqrt(k.shape[-1]))
        att = jnp.where(mask == 0, float('-inf'), att)
        att = nn.softmax(att, axis=-1)
        att = self.attn_dropout(att, deterministic=deterministic)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.swapaxes(1, 2).reshape(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.resid_dropout(self.c_proj(y), deterministic=deterministic)
        return y

class FeedForward(nn.Module):
    n_embd: int
    dropout: float
    
    def setup(self):
        self.net = nn.Sequential([
            nn.Dense(4 * self.n_embd),
            nn.relu,
            nn.Dense(self.n_embd)
        ])
        #self.dropout = nn.Dropout(self.dropout)

    def __call__(self, x, deterministic: bool):
        return self.net(x)
        #return self.dropout(x, deterministic=deterministic)

class TransformerBlock(nn.Module):
    n_embd: int
    n_head: int
    dropout: float

    def setup(self):
        #head_size = self.n_embd // n_head
        #self.sa = MultiHeadAttention(n_head, head_size)
        self.ln1 = nn.LayerNorm(epsilon=1e-5)
        self.sa = CausalSelfAttention(self.n_head, self.n_embd, self.dropout)
        #self.sa = SelfAttention(self.n_head, dropout_rate=self.dropout) # head size is calculated internally
        self.ln2 = nn.LayerNorm(epsilon=1e-5)
        self.ffwd = FeedForward(self.n_embd, self.dropout)

    def __call__(self, x, deterministic):
        x = x + self.sa(self.ln1(x), deterministic=deterministic)
        x = x + self.ffwd(self.ln2(x), deterministic=deterministic)
        return x

class GPTLanguageModel(nn.Module):
    """ This block performs the embedding of the tokens and the positional embedding, it feeds the result
    into the transformer blocks, normalizes the result and feeds it into the final linear layer."""
    block_size = 256 # what is the maximum context length for predictions?
    n_embd = 384 # dimension to which tokens and position is embedded
    n_head = 6 # number of parallel attention heads per block
    n_layer = 6 # number of transformer blocks
    dropout = 0.2
    vocab_size = 65

    def setup(self):
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = Embed(self.vocab_size, self.n_embd)
        self.position_embedding_table = Embed(self.block_size, self.n_embd)
        self.blocks = [TransformerBlock(self.n_embd, self.n_head, self.dropout) for _ in range(self.n_layer)]
        self.ln_f = nn.LayerNorm(epsilon=1e-5) # final layer norm
        self.lm_head = nn.Dense(self.vocab_size)

    def __call__(self, xb, deterministic=True):
        B, T = xb.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(xb) # (B,T,C)
        pos_emb = self.position_embedding_table(jnp.arange(T)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        for block in self.blocks:
            x = block(x, deterministic) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        return logits

model = GPTLanguageModel()
key, batch_key, params_key, dropout_rng = random.split(key, 4)
xb, yb = get_batch(batch_key, 'train')

model_key = {'params': params_key, 'dropout': dropout_rng}


# print the number of parameters in the model
# print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

def create_train_state(key, model, xb_sample, learning_rate=3e-4):
    key, dropout_key, params_key = jax.random.split(key, 3)
    rngs = {'dropout': dropout_key, 'params': params_key}
    params = model.init(rngs, xb_sample, deterministic=True)

    # create an optimizer
    optimizer = optax.adamw(learning_rate=learning_rate)
    #opt_state = optimizer.init(params)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)


# params = model.init(model_key, xb, deterministic=True)
key, init_key = jax.random.split(key)
trn_state = create_train_state(key, model, xb)

key, dropout_key = random.split(key)

def make_loss_fn(key, xb, yb):
    def loss_fn(params):
        yb_inner = yb

        logits = model.apply(params, xb, deterministic=False, rngs={'dropout':key})
        B, T, C = logits.shape
        logits = logits.reshape(B*T, C)
        yb_inner = yb_inner.reshape(B*T)

        #yb_inner_one_hot = jnp.eye(C)[yb_inner]  # One-hot encoding of targets
        yb_inner_one_hot = jax.nn.one_hot(yb_inner, C)

        loss = optax.softmax_cross_entropy_with_integer_labels(logits, yb_inner).mean()
        return loss
    return loss_fn

for iter in range(max_iters):

    key, batch_key, loss_fn_key, est_loss_key = random.split(key, 4)

    # sample a batch of data
    xb, yb = get_batch(batch_key, 'train')

    loss_fn = make_loss_fn(loss_fn_key, xb, yb)

    loss, grads = jax.value_and_grad(loss_fn)(trn_state.params)
    trn_state = trn_state.apply_gradients(grads=grads)

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(est_loss_key, trn_state.params)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")



def generate(key, params, model, idx, max_new_tokens):
    loc_key = key
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):
        loc_key, sample_key, dropout_key = random.split(loc_key, 3)
        # crop idx to the last block_size tokens
        idx_cond = idx[:, -block_size:]
        # get the predictions
        logits = model.apply(params, idx_cond, deterministic=True, rngs={'dropout' : dropout_key})
        # focus only on the last time step
        logits = logits[:, -1, :] # becomes (B, C)
        # apply softmax to get probabilities
        # probs = nn.softmax(logits) # (B, C) TODO
        # sample from the distribution
        idx_next = random.categorical(sample_key, logits) # (B, 1)

        idx_next = jnp.reshape(idx_next, (1,-1))
        # append sampled index to the running sequence
        idx = jnp.concatenate((idx, idx_next), axis=1) # (B, T+1)
    return idx

# def generate(key, params, model, input_tokens, max_new_tokens, temperature=1.0, top_k=None):
#     """
#     Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
#     the sequence max_new_tokens times, feeding the predictions back into the model each time.
#     Most likely you'll want to make sure to be in model.eval() mode of operation for this.
#     """
#     B, T = input_tokens.shape
#     padding = jnp.zeros((B, max_new_tokens), dtype=jnp.int32)
#     tokens = jnp.concatenate([input_tokens, padding], axis=-1)
#     indexes = jnp.arange(T, T + max_new_tokens)

#     # tokens index -> tokens None
#     def scan_f(tokens, i):
#         # l: x y
#         # t: a b - -
#         # i: 0 1 2 3
#         step_key = jax.random.fold_in(key, i)
#         # if the sequence context is growing too long we must crop it at block_size
#         # idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
#         # forward the model to get the logits for the index in the sequence
#         logits = model.apply(params, tokens, deterministic=True)
#         # pluck the logits at the final step and scale by desired temperature
#         logits = logits[:, i - 1, :] / temperature
#         # optionally crop the logits to only the top k options
#         # sample from the distribution
#         if top_k is not None:
#             top_logits, top_tokens = jax.lax.top_k(logits, min(top_k, logits.shape[-1]))
#             token_idx = jax.random.categorical(step_key, top_logits, axis=-1)
#             next_token = jnp.take_along_axis(top_tokens, token_idx[:, None], axis=-1).squeeze(-1)
#         else:
#             next_token = jax.random.categorical(step_key, logits, axis=-1)
#             # logits = jnp.where(logits < v[:, -1:], float('-inf'), logits)
#         # append sampled index to the running sequence and continue
#         tokens = tokens.at[:, i].set(next_token)

#         return tokens, None

#     tokens, _ = jax.lax.scan(scan_f, tokens, indexes)

#     return tokens

key, generate_key = random.split(key)

# generate from the model
context = jnp.zeros((1, 1), dtype=jnp.int32)
print(decode(generate(generate_key, trn_state.params, model, context, max_new_tokens=500)[0].tolist()))
open('more.txt', 'w').write(decode(generate(generate_key, trn_state.params, model, context, max_new_tokens=10000)[0].tolist()))


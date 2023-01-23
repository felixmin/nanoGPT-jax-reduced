"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.traverse_util import path_aware_map
from flax.core import freeze

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    config: GPTConfig

    def setup(self):
        config = self.config
        assert config.n_embd % config.n_head == 0
        head_size = config.n_embd // config.n_head
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.DenseGeneral((config.n_head, 3 * head_size))
        # output projection
        self.c_proj = nn.Dense(config.n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def __call__(self, x: jax.Array, *, train: bool) -> jax.Array:
        B, T, C = x.shape # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        c_attn = self.c_attn(x) # (B, T, hs, 3 * hs)
        c_attn = c_attn.swapaxes(1, 2) # (B, nh, T, 3 * hs)
        q, k, v = jnp.split(c_attn, 3, axis=-1) # (B, nh, T, hs)

        with jax.ensure_compile_time_eval():
            mask = jnp.tril(jnp.ones((T, T))).reshape((1, 1, T, T))
        
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.swapaxes(-2, -1)) * (1.0 / jnp.sqrt(k.shape[-1]))
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = jnp.where(mask == 0, float('-inf'), att)
        att = nn.softmax(att, axis=-1)
        att = self.attn_dropout(att, deterministic=not train)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.swapaxes(1, 2).reshape(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y), deterministic=not train)
        return y

class MLP(nn.Module):
    config: GPTConfig

    def setup(self):
        config = self.config
        self.c_fc    = nn.Dense(4 * config.n_embd)
        self.c_proj  = nn.Dense(config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def __call__(self, x: jax.Array, *, train: bool) -> jax.Array:
        x = self.c_fc(x)
        x = nn.gelu(x, approximate=True)
        x = self.c_proj(x)
        x = self.dropout(x, deterministic=not train)
        return x

class Block(nn.Module):
    config: GPTConfig

    def setup(self):
        config = self.config
        self.ln_1 = nn.LayerNorm()
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm()
        self.mlp = MLP(config)

    def __call__(self, x: jax.Array, *, train: bool) -> jax.Array:
        x = x + self.attn(self.ln_1(x), train=train)
        x = x + self.mlp(self.ln_2(x), train=train)
        return x



class GPT(nn.Module):
    config: GPTConfig

    def setup(self):
        config = self.config
        assert config.vocab_size is not None
        assert config.block_size is not None

        self.wte = nn.Embed(config.vocab_size, config.n_embd)
        self.wpe = nn.Embed(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.h = [Block(config) for _ in range(config.n_layer)]
        self.ln_f = nn.LayerNorm()

    def __call__(self, idx: jax.Array, *, train: bool, targets: Optional[jax.Array] = None):
        b, t = idx.shape
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = jnp.arange(0, t, dtype=jnp.int32)[None] # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = self.drop(tok_emb + pos_emb, deterministic=not train)
        for block in self.h:
            x = block(x, train=train)
        x = self.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.wte.attend(x)
            # loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits, targets).mean()
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.wte.attend(x[:, -1:, :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self, params, block_size: int):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        

        assert block_size <= self.config.block_size
        self.config.block_size = block_size

        # self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        def crop_weights(path: Tuple[str, ...], x):
            if path[-2:] == ("wpe", "embedding"):
                return x[:block_size]
            return x

        return freeze(path_aware_map(crop_weights, params))
        

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        # we can override the dropout rate
        if 'dropout' in override_args:
            config_args['dropout'] = override_args['dropout']
        # block_size is always 1024 for GPT model checkpoints
        # if one wants a lower block_size it has to be done through model surgery
        # later, by calling crop_block_shape

        # create a from-scratch initialized minGPT model
        config = GPTConfig(block_size=1024, **config_args)
        model = GPT(config)
        sd = model.state_dict()

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        keys = [k for k in sd_hf if not k.endswith('attn.masked_bias')] # ignore these
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(keys) == len(sd)
        for k in keys:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, params, weight_decay, learning_rate, betas):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        def get_optimizer(decay):
            return optax.adamw(
                learning_rate=learning_rate, b1=betas[0], b2=betas[1],
                weight_decay=decay)
        
        def partition_fn(path: Tuple[str, ...], x) -> str:
            if path[-1] in ('bias', 'scale', 'embedding'):
                return 'no_decay'
            elif path[-1] in ('kernel',):
                return 'decay'
            else:
                raise ValueError(f"Unrecognized parameter: {path}")

        partition_optimizers = {    
            'decay': get_optimizer(weight_decay), 
            'no_decay': get_optimizer(0.0)}
        param_partitions = freeze(path_aware_map(partition_fn, params))
        tx = optax.multi_transform(partition_optimizers, param_partitions)

        return tx

    # @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

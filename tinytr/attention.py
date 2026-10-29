import jax
import jax.numpy as jnp

from tinytr.ops import softmax
from functools import partial

def scaled_dot_product_attention(q, k, v, causal: bool = False):
    """Compute scaled dot-product attention.

    Args:
        q: Queries of shape [S, d_k]
        k: Keys of shape [S, d_k]
        v: Values of shape [S, d_k]
        causal: If True, apply a causal mask to prevent attending to future tokens.

    Returns:
        Output of shape [S, d_k] after applying attention.
    """
    # Compute attention scores
    scores = (q @ k.T) / jnp.sqrt(k.shape[-1])  # [S, S]

    if causal:
        # Create a causal mask
        S = q.shape[0]
        mask = jnp.tril(jnp.ones((S, S), dtype=bool))
        # Apply the mask to the attention scores
        scores = jnp.where(mask, scores, -jnp.inf)

    # Compute attention weights using softmax
    weights = softmax(scores)  # [S, S]

    # Compute the output as a weighted sum of the values
    output = weights @ v  # [S, d_k]
    return output

def attention(x, wq, wk, wv, causal: bool = False):
    # S = sequence length, D = model width, d_k = query/key/value width
    # x: input tensor of shape [S, D] = [8, 64] 
    # wq, wk, wv: weight matrices for query, key, and value projections of shape [D, d_k] = [64, 16]

    # project input x into query, key, and value spaces
    q = x @ wq # [S, d_k] = [8, 16]
    k = x @ wk # [S, d_k] = [8, 16]
    v = x @ wv # [S, d_k] = [8, 16]

    return scaled_dot_product_attention(q, k, v, causal=causal)

def multi_head_attention(x, wq, wk, wv, wo, n_heads, causal: bool = False):
    # x: [S, D]   wq, wk, wv: [D, D] (= [D, H*d_k])   wo: [D, D]
    S, D = x.shape
    d_k = D // n_heads

    # project with the FULL input, then split the projected columns into heads
    q = (x @ wq).reshape(S, n_heads, d_k)   # [S, H, d_k]
    k = (x @ wk).reshape(S, n_heads, d_k)   # [S, H, d_k]
    v = (x @ wv).reshape(S, n_heads, d_k)   # [S, H, d_k]

    # run the single-head core once per head, mapped over axis 1 (the head axis)
    core = partial(scaled_dot_product_attention, causal=causal)
    heads = jax.vmap(core, in_axes=(1, 1, 1), out_axes=1)(q, k, v)   # [S, H, d_k]

    # concatenating the heads = merging the last two axes back into D
    return heads.reshape(S, D) @ wo         # [S, D]

def init_attention_weights(key, d_model, d_k):
    """Three [d_model, d_k] weight matrices, scaled so q and k have unit variance."""
    kq, kk, kv = jax.random.split(key, 3)
    scale = 1 / jnp.sqrt(d_model)
    return tuple(jax.random.normal(k_, (d_model, d_k)) * scale for k_ in (kq, kk, kv))

def init_mha_weights(key, d_model):
    # multi-head attention weights
    """wq, wk, wv, wo: [d_model, d_model], each scaled by 1/sqrt(d_model)."""
    keys = jax.random.split(key, 4)
    scale = 1 / jnp.sqrt(d_model)
    return tuple(jax.random.normal(k_, (d_model, d_model)) * scale for k_ in keys)

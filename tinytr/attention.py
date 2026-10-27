import jax
import jax.numpy as jnp
from tinytr.ops import softmax

def attention(x, wq, wk, wv):
    # S = sequence length, D = model width, d_k = query/key/value width
    # x: input tensor of shape [S, D] = [8, 64] 
    # wq, wk, wv: weight matrices for query, key, and value projections of shape [D, d_k] = [64, 16]

    # project input x into query, key, and value spaces
    q = x @ wq # [S, d_k] = [8, 16]
    k = x @ wk # [S, d_k] = [8, 16]
    v = x @ wv # [S, d_k] = [8, 16]

    # compute attention scores and weights
    attention_scores = (q @ k.T) / jnp.sqrt(k.shape[-1]) # [S, S] = [8, 8]

    # compute attention weights using the numerically stable softmax along the last axis
    attention_weights = softmax(attention_scores) # [S, S] = [8, 8]

    # compute the output as a weighted sum of the values 
    output = attention_weights @ v # [S, d_k] = [8, 16]
    return output

def init_attention_weights(key, d_model, d_k):
    """Three [d_model, d_k] weight matrices, scaled so q and k have unit variance."""
    kq, kk, kv = jax.random.split(key, 3)
    scale = 1 / jnp.sqrt(d_model)
    return tuple(jax.random.normal(k_, (d_model, d_k)) * scale for k_ in (kq, kk, kv))
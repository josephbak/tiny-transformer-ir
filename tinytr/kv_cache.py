import jax
import jax.numpy as jnp

from tinytr.ops import softmax, layer_norm
from tinytr.block import ffn
from tinytr.model import embed, output_head


def init_cache(cfg):
    """Empty KV cache for every layer: {"k": [L, S_max, H, d_k], "v": [L, S_max, H, d_k]}."""
    L, S_max, H = cfg["n_layers"], cfg["max_len"], cfg["n_heads"]
    d_k = cfg["d_model"] // H                      # Q1: derived from which two config values?
    shape = (L, S_max, H, d_k)
    return {"k": jnp.zeros(shape), "v": jnp.zeros(shape)}    # Q2: what fills an empty cache? zeros?


def cached_attention(x_new, wq, wk, wv, wo, k_cache, v_cache, pos, n_heads):
    """x_new: [T_new, D] new tokens starting at position pos. k_cache/v_cache: [S_max, H, d_k] (one layer).
    Returns: out [T_new, D], updated k_cache, updated v_cache."""
    T_new, D = x_new.shape
    S_max = k_cache.shape[0]
    d_k = D // n_heads

    # 1. project the new tokens, split into heads
    q = (x_new @ wq).reshape(T_new, n_heads, d_k)     # [T_new, H, d_k]
    k = (x_new @ wk).reshape(T_new, n_heads, d_k)     # same for k
    v = (x_new @ wv).reshape(T_new, n_heads, d_k)     # same for v

    # 2. write the new rows into the cache, starting at row `pos`
    k_cache = jax.lax.dynamic_update_slice(k_cache, k, (pos, 0, 0))
    v_cache = jax.lax.dynamic_update_slice(v_cache, v, (pos, 0, 0))

    # 3. scores: every new query against every cached key, per head
    scores = jnp.einsum("thd,shd->hts", q, k_cache) / jnp.sqrt(d_k)   # [H, T_new, S_max]

    # 4. mask: query t (at position pos + t) may see key s only if s <= pos + t
    q_pos = pos + jnp.arange(T_new)                   # [T_new]
    k_pos = jnp.arange(S_max)                         # [S_max]
    mask = k_pos[None, :] <= q_pos[:, None]          # [1, S_max] vs [T_new, 1] → [T_new, S_max]
    scores = jnp.where(mask, scores, -jnp.inf)        # broadcasts over H

    # 5. weights, mix values, merge heads, output projection
    weights = softmax(scores, axis=-1)
    out = jnp.einsum("hts,shd->thd", weights, v_cache)   # [T_new, H, d_k]
    return out.reshape(T_new, D) @ wo, k_cache, v_cache

def cached_block(x_new, params, k_cache, v_cache, pos, n_heads):
    ln1, attn, ln2 = params["ln1"], params["attn"], params["ln2"]

    # sublayer 1: attention, with a residual
    a = layer_norm(x_new, ln1["gamma"], ln1["beta"])
    attn_out, k_cache, v_cache = cached_attention(a, attn["wq"], attn["wk"], attn["wv"], attn["wo"],
                                            k_cache, v_cache, pos, n_heads)
    h = x_new + attn_out

    # sublayer 2: FFN, with a residual
    f = layer_norm(h, ln2["gamma"], ln2["beta"])
    return h + ffn(f, params["ffn"]), k_cache, v_cache

def prefill(params, tokens, cache, cfg):
    pos = 0 # 0 for starting position

    x = embed(params, tokens, jnp.arange(tokens.shape[0]))  # [P, D]

    for i in range(cfg["n_layers"]):
        block_i = jax.tree.map(lambda a: a[i], params["blocks"])     # layer i's slice of every leaf
        x, k_i, v_i = cached_block(x, block_i, cache["k"][i], cache["v"][i], pos, cfg["n_heads"])
        cache = {"k": cache["k"].at[i].set(k_i), "v": cache["v"].at[i].set(v_i)}

    logits = output_head(params, x[-1:])  # [n, V], n = 1 for the last token
    return logits, cache

def decode_step(params, token, cache, pos, cfg):
    x = embed(params, token, pos + jnp.arange(1))   # positions = [pos], shape [1]

    for i in range(cfg["n_layers"]):
        block_i = jax.tree.map(lambda a: a[i], params["blocks"])     # layer i's slice of every leaf
        x, k_i, v_i = cached_block(x, block_i, cache["k"][i], cache["v"][i], pos, cfg["n_heads"])
        cache = {"k": cache["k"].at[i].set(k_i), "v": cache["v"].at[i].set(v_i)}

    logits = output_head(params, x)  # [1, V]
    return logits, cache
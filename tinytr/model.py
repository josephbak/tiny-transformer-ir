"""The full decoder-only Transformer: embeddings → L blocks → final LayerNorm → logits."""

import jax
import jax.numpy as jnp

from tinytr.block import init_block_params, transformer_block
from tinytr.ops import layer_norm

# all model sizes in one place
CONFIG = {
    "vocab": 100,      # V: number of distinct tokens
    "max_len": 16,     # S_max: context length (rows of the position table)
    "d_model": 64,     # D: width of every token vector
    "n_heads": 4,      # H
    "d_ff": 256,       # F: FFN hidden width
    "n_layers": 2,     # L: number of stacked blocks
}


def init_model_params(key, cfg):
    """All model parameters as one pytree."""
    V, S_max, D, F, L = cfg["vocab"], cfg["max_len"], cfg["d_model"], cfg["d_ff"], cfg["n_layers"]

    # one key for each of: embed, pos, unembed, and every block
    keys = jax.random.split(key, L + 3)
    k_embed, k_pos, k_unembed, block_keys = keys[0], keys[1], keys[2], keys[3:]

    return {
        "embed": jax.random.normal(k_embed, (V, D)) * 0.02,           # [V, D]: one row per token
        "pos": jax.random.normal(k_pos, (S_max, D)) * 0.02,           # [S_max, D]: one row per position
        # "blocks": [init_block_params(k, D, F) for k in block_keys],   # list of L block dicts
        "blocks": stack_blocks([init_block_params(k, D, F) for k in block_keys]),  # leaves: [L, ...]
        "ln_f": {"gamma": jnp.ones((D,)), "beta": jnp.zeros((D,))},   # final LayerNorm
        "unembed": jax.random.normal(k_unembed, (D, V)) / jnp.sqrt(D),  # [D, V]
    }


# def forward(ids, params, cfg):
#     """ids: [S] integer token ids → logits: [S, V]. Row i predicts token i+1."""
#     S = ids.shape[0]
#     x = params["embed"][ids] + params["pos"][:S]          # [S, D]: what the token is + where it sits
#     for block_params in params["blocks"]:                 # Python loop: unrolled at trace time
#         x = transformer_block(x, block_params, cfg["n_heads"], causal=True)   # [S, D]
#     x = layer_norm(x, params["ln_f"]["gamma"], params["ln_f"]["beta"])       # [S, D]
#     return x @ params["unembed"]                          # [S, V]



def forward(ids, params, cfg):
    """Loop version. params["blocks"] is stacked ([L, ...] leaves); unrolled at trace time."""
    S = ids.shape[0]
    x = params["embed"][ids] + params["pos"][:S]                          # [S, D]

    for i in range(cfg["n_layers"]):
        block_params = jax.tree.map(lambda a: a[i], params["blocks"])     # layer i's slice of every leaf
        x = transformer_block(x, block_params, cfg["n_heads"], causal=True)

    x = layer_norm(x, params["ln_f"]["gamma"], params["ln_f"]["beta"])
    return x @ params["unembed"]                                          # [S, V]


def forward_scan(ids, params, cfg):
    """Scan version. params["blocks"] is already stacked, so scan uses it directly."""
    S = ids.shape[0]
    x = params["embed"][ids] + params["pos"][:S]                          # [S, D]

    def step(x, block_params):                                            # one layer
        return transformer_block(x, block_params, cfg["n_heads"], causal=True), None

    x, _ = jax.lax.scan(step, x, params["blocks"])                        # one block in the IR, looped L times
    x = layer_norm(x, params["ln_f"]["gamma"], params["ln_f"]["beta"])
    return x @ params["unembed"]                                          # [S, V]


def count_params(params):
    """Total number of scalar parameters in the pytree."""
    return sum(leaf.size for leaf in jax.tree.leaves(params))


def stack_blocks(block_list):
    """List of L block pytrees → one pytree whose leaves have a leading axis L."""
    return jax.tree.map(lambda *leaves: jnp.stack(leaves), *block_list)


# def forward_scan(ids, params, cfg):
#     """Same as forward, but the blocks run inside jax.lax.scan (one copy in the IR)."""
#     S = ids.shape[0]
#     x = params["embed"][ids] + params["pos"][:S]                     # [S, D]

#     def step(x, block_params):                                       # one layer
#         return transformer_block(x, block_params, cfg["n_heads"], causal=True), None

#     x, _ = jax.lax.scan(step, x, stack_blocks(params["blocks"]))     # loop over L layers
#     x = layer_norm(x, params["ln_f"]["gamma"], params["ln_f"]["beta"])
#     return x @ params["unembed"]                                     # [S, V]


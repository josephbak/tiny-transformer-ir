import jax
import jax.numpy as jnp

from tinytr.ops import linear, layer_norm
from tinytr.attention import multi_head_attention, init_mha_weights

def ffn(x, params):
    """
    Feedforward network (FFN) for transformers.

    Args:
        x: Input tensor of shape [..., d_model].
        params: A dictionary containing the weights and biases for the FFN.

    Returns:
        Output tensor of shape [..., d_model].
    """
    # First linear layer
    w1 = params['w1']  # Shape: (d_model, d_ff)
    b1 = params['b1']  # Shape: (d_ff,)
    h0 = linear(x, w1, b1)  # Shape: (batch_size, seq_length, d_ff)

    # Activation function (ReLU)
    h1 = jnp.maximum(0, h0)  # Shape: (batch_size, seq_length, d_ff)

    # Second linear layer
    w2 = params['w2']  # Shape: (d_ff, d_model)
    b2 = params['b2']  # Shape: (d_model,)
    y = linear(h1, w2, b2)  # Shape: (batch_size, seq_length, d_model)

    return y

def init_ffn_params(key, d_model, d_ff):
    """FFN parameters: weights scaled by 1/sqrt(fan_in), biases zero."""
    k1, k2 = jax.random.split(key)
    return {
        "w1": jax.random.normal(k1, (d_model, d_ff)) / jnp.sqrt(d_model),   # [D, F]: sums over D
        "b1": jnp.zeros((d_ff,)),                                           # [F]
        "w2": jax.random.normal(k2, (d_ff, d_model)) / jnp.sqrt(d_ff),      # [F, D]: sums over F
        "b2": jnp.zeros((d_model,)),                                        # [D]
    }


def transformer_block(x, params, n_heads, causal=True):
    ln1, attn, ln2 = params["ln1"], params["attn"], params["ln2"]

    # sublayer 1: attention, with a residual
    a = layer_norm(x, ln1["gamma"], ln1["beta"])
    h = x + multi_head_attention(a, attn["wq"], attn["wk"], attn["wv"], attn["wo"],
                                 n_heads=n_heads, causal=causal)

    # sublayer 2: FFN, with a residual
    f = layer_norm(h, ln2["gamma"], ln2["beta"])
    return h + ffn(f, params["ffn"])


def init_block_params(key, d_model, d_ff):
    """All parameters for one Transformer block, as a nested dict (a pytree)."""
    k_attn, k_ffn = jax.random.split(key)

    wq, wk, wv, wo = init_mha_weights(k_attn, d_model)     # four [D, D] matrices

    return {
        "ln1":  {"gamma": jnp.ones((d_model,)), "beta": jnp.zeros((d_model,))},
        "attn": {"wq": wq, "wk": wk, "wv": wv, "wo": wo},
        "ln2":  {"gamma": jnp.ones((d_model,)), "beta": jnp.zeros((d_model,))},
        "ffn":  init_ffn_params(k_ffn, d_model, d_ff),     # already a dict: w1, b1, w2, b2
    }
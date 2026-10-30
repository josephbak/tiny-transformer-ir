import jax
import jax.numpy as jnp
import numpy as np

from tinytr.block import ffn, init_ffn_params, init_block_params, transformer_block

S, D, F = 8, 64, 256


def _ffn_inputs(seed=0):
    kx, kp = jax.random.split(jax.random.key(seed))
    return jax.random.normal(kx, (S, D)), init_ffn_params(kp, D, F)


def test_ffn_shape():
    x, params = _ffn_inputs()
    assert ffn(x, params).shape == (S, D)


def test_ffn_tokens_independent():
    # change ONLY token t: only output row t may change; every other row must be identical
    x, params = _ffn_inputs()
    t = 3
    x2 = x.at[t].set(x[t] + 1.0)
    y1, y2 = ffn(x, params), ffn(x2, params)
    others = jnp.arange(S) != t
    np.testing.assert_array_equal(y1[others], y2[others])   # exactly equal, not just close
    assert not np.allclose(y1[t], y2[t])

H = 4


def _block_inputs(seed=0):
    kx, kp = jax.random.split(jax.random.key(seed))
    return jax.random.normal(kx, (S, D)), init_block_params(kp, D, F)


def test_block_shape():
    x, params = _block_inputs()
    assert transformer_block(x, params, n_heads=H).shape == (S, D)


def test_block_causal():
    # LayerNorm and FFN act per token, so only attention could leak the future
    x, params = _block_inputs()
    t = 5
    x2 = x.at[t].set(x[t] + 10.0)
    y1 = transformer_block(x, params, n_heads=H, causal=True)
    y2 = transformer_block(x2, params, n_heads=H, causal=True)
    np.testing.assert_allclose(y1[:t], y2[:t], rtol=1e-6)


def test_block_residual_identity():
    # zero the LAST layer of each sublayer: both sublayers then output exactly 0,
    # so the block must return x unchanged; proves the skip paths are wired right
    x, params = _block_inputs()
    params = {
        **params,
        "attn": {**params["attn"], "wo": jnp.zeros((D, D))},
        "ffn": {**params["ffn"], "w2": jnp.zeros((F, D)), "b2": jnp.zeros((D,))},
    }
    np.testing.assert_array_equal(transformer_block(x, params, n_heads=H), x)
import jax
import numpy as np

from tinytr.attention import attention, init_attention_weights
from tinytr.reference import attention_loop

S, D, DK = 8, 64, 16


def _inputs(seed=0):
    kx, kw = jax.random.split(jax.random.key(seed))
    x = jax.random.normal(kx, (S, D))
    wq, wk, wv = init_attention_weights(kw, D, DK)   # scaled, so attention actually mixes
    return x, wq, wk, wv


def test_attention_shape():
    x, wq, wk, wv = _inputs()
    assert attention(x, wq, wk, wv).shape == (S, DK)


def test_attention_matches_loop_reference():
    # the two versions sum in different orders, so allow small float32 differences
    x, wq, wk, wv = _inputs()
    np.testing.assert_allclose(
        attention(x, wq, wk, wv),
        attention_loop(x, wq, wk, wv),
        rtol=1e-5,
        atol=1e-6,
    )
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


T = 5  # the token we perturb

def _perturbed(x):
    return x.at[T].set(x[T] + 10.0)   # change token T only (arrays are immutable: .at[].set)


def test_causal_earlier_outputs_ignore_later_tokens():
    x, wq, wk, wv = _inputs()
    out1 = attention(x, wq, wk, wv, causal=True)
    out2 = attention(_perturbed(x), wq, wk, wv, causal=True)
    np.testing.assert_allclose(out1[:T], out2[:T], rtol=1e-6)   # tokens 0..T-1: unchanged
    assert not np.allclose(out1[T:], out2[T:])                   # tokens T..: did change


def test_noncausal_earlier_outputs_do_change():
    # sanity check: without the mask, every token sees token T
    x, wq, wk, wv = _inputs()
    out1 = attention(x, wq, wk, wv, causal=False)
    out2 = attention(_perturbed(x), wq, wk, wv, causal=False)
    assert not np.allclose(out1[:T], out2[:T])
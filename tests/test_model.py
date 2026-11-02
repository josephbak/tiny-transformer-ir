import jax
import jax.numpy as jnp
import numpy as np

from tinytr.model import CONFIG, count_params, forward, init_model_params

S = 8


def _params(seed=0):
    return init_model_params(jax.random.key(seed), CONFIG)


def _ids(seed=1):
    return jax.random.randint(jax.random.key(seed), (S,), 0, CONFIG["vocab"])   # S random token ids


def test_logits_shape():
    assert forward(_ids(), _params(), CONFIG).shape == (S, CONFIG["vocab"])


def test_param_count():
    # embed 6400 + pos 1024 + unembed 6400 + ln_f 128 + 2 blocks × 49728
    assert count_params(_params()) == 113_408


def test_model_causal():
    # change the token at position t: logits for earlier positions must not change,
    # through every block (attention is masked in all of them)
    params, ids = _params(), _ids()
    t = 5
    ids2 = ids.at[t].set((ids[t] + 1) % CONFIG["vocab"])   # a different token at position t
    y1, y2 = forward(ids, params, CONFIG), forward(ids2, params, CONFIG)
    np.testing.assert_allclose(y1[:t], y2[:t], rtol=1e-6)
    assert not np.allclose(y1[t:], y2[t:])


def test_position_matters():
    # the same token at two positions gets different logits: the position table is used
    params = _params()
    ids = jnp.full((S,), 7)                 # token 7 everywhere
    y = forward(ids, params, CONFIG)
    assert not np.allclose(y[0], y[1])
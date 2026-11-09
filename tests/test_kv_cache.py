"""KV-cache equivalence tests (M9 step 6).

One idea, checked at increasing depth: the cached path (prefill + decode_step)
must produce the same logits as the full-sequence `forward` on the same tokens.

Signatures assumed (from tinytr):
    forward(ids, params, cfg)                      -> logits [S, V]
    init_cache(cfg)                                -> {"k": [L,S_max,H,d_k], "v": same}
    prefill(params, tokens, cache, cfg)            -> (logits [1, V], cache)
    decode_step(params, tokens, cache, pos, cfg)   -> (logits [1, V], cache)
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tinytr.model import CONFIG, init_model_params, forward
from tinytr.kv_cache import init_cache, prefill, decode_step

S_MAX = CONFIG["max_len"]          # 16
TOL = dict(rtol=1e-5, atol=1e-5)   # float32: reduction order differs between the two paths


def setup():
    """Fixed weights and a fixed 16-token sequence, so every test sees the same data."""
    params = init_model_params(jax.random.key(0), CONFIG)
    ids = jax.random.randint(jax.random.key(1), (S_MAX,), 0, CONFIG["vocab"])   # [16] int ids
    return params, ids


# ---------------------------------------------------------------------------
# 1. Prefill alone
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("P", [1, 5, 16])
def test_prefill_matches_forward(P):
    """Prefill's single logits row must equal forward's last row on the same prompt.

    Checks: embed (positions 0..P-1), the layer loop, cached_attention with pos=0,
    and output_head on x[-1:]. P=1 and P=16 cover the edges (one token, full cache).
    """
    params, ids = setup()
    logits, _ = prefill(params, ids[:P], init_cache(CONFIG), CONFIG)   # [1, V]
    ref = forward(ids[:P], params, CONFIG)[-1]                          # [V], row P-1

    assert logits.shape == (1, CONFIG["vocab"])
    np.testing.assert_allclose(logits[0], ref, **TOL)


# ---------------------------------------------------------------------------
# 2. Prefill + one decode step
# ---------------------------------------------------------------------------
def test_one_decode_step_matches_forward():
    """Decode at pos 5 consumes token 5 and must match forward's row 5 on ids[:6].

    Checks: decode_step's embed (position [5]), the position-based mask
    (q at pos 5 sees slots 0..5), and the slot-5 write in both layers.
    """
    params, ids = setup()
    P = 5
    _, cache = prefill(params, ids[:P], init_cache(CONFIG), CONFIG)

    logits, _ = decode_step(params, ids[P:P + 1], cache, P, CONFIG)     # token at position 5
    ref = forward(ids[:P + 1], params, CONFIG)[-1]                      # row 5

    np.testing.assert_allclose(logits[0], ref, **TOL)


# ---------------------------------------------------------------------------
# 3. Prefill + decode all the way to S_max  (the important one)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("P", [1, 5, 15])
def test_decode_to_s_max_matches_forward(P):
    """Decode every remaining position, keeping the returned cache each time.

    Teacher forcing: each step feeds the known next id (not a sampled one), so
    both paths see exactly the same sequence. At every pos, the decode logits
    must equal forward's row pos on ids[:pos+1].

    Checks write-back across calls: if any level drops a returned cache slice,
    the next step reads zeros where a k, v row should be, and this fails.
    """
    params, ids = setup()
    _, cache = prefill(params, ids[:P], init_cache(CONFIG), CONFIG)

    for pos in range(P, S_MAX):
        logits, cache = decode_step(params, ids[pos:pos + 1], cache, pos, CONFIG)
        ref = forward(ids[:pos + 1], params, CONFIG)[-1]
        np.testing.assert_allclose(logits[0], ref, **TOL, err_msg=f"mismatch at pos={pos}")


# ---------------------------------------------------------------------------
# 4. Cache contents after prefill
# ---------------------------------------------------------------------------
def test_prefill_writes_only_prompt_slots():
    """After prefill on 5 tokens: slots 0..4 filled, slots 5..15 still exactly zero.

    Checked for k and v, in every layer. Catches a write at the wrong pos,
    into the wrong layer, or a layer that was never written back.
    """
    params, ids = setup()
    P = 5
    _, cache = prefill(params, ids[:P], init_cache(CONFIG), CONFIG)

    for name in ("k", "v"):
        c = np.asarray(cache[name])                                   # [L, S_max, H, d_k]
        assert c.shape == (CONFIG["n_layers"], S_MAX, CONFIG["n_heads"],
                           CONFIG["d_model"] // CONFIG["n_heads"])
        for layer in range(CONFIG["n_layers"]):
            filled = np.abs(c[layer, :P]).sum(axis=(-2, -1))         # one number per slot
            assert np.all(filled > 0), f'{name} layer {layer}: a prompt slot is empty'
            assert np.all(c[layer, P:] == 0), f'{name} layer {layer}: slot >= {P} was written'


# ---------------------------------------------------------------------------
# 5. The test can see the fall-through
# ---------------------------------------------------------------------------
def test_dropped_cache_is_detected():
    """Sanity check on the tests themselves: a stale cache must give wrong logits.

    Decode pos 5 but throw away the returned cache, then decode pos 6 with the
    old one. Slot 5 is still zeros but the pos-6 mask lets it be seen, so the
    logits must differ from forward. If this test failed, test 3 could not
    catch a dropped write-back.
    """
    params, ids = setup()
    P = 5
    _, cache = prefill(params, ids[:P], init_cache(CONFIG), CONFIG)

    _ = decode_step(params, ids[P:P + 1], cache, P, CONFIG)            # returned cache dropped
    stale_logits, _ = decode_step(params, ids[P + 1:P + 2], cache, P + 1, CONFIG)
    ref = forward(ids[:P + 2], params, CONFIG)[-1]

    assert not np.allclose(stale_logits[0], ref, **TOL)


# ---------------------------------------------------------------------------
# 6. jit: one compile for every decode position
# ---------------------------------------------------------------------------
def test_jitted_decode_traces_once():
    """pos is a value, not a shape, so the jitted decode step compiles once.

    The counter lives in a Python wrapper. Python code inside a jitted function
    runs only while JAX traces it, so the counter counts traces, not calls.
    cfg is closed over in the wrapper because a dict can't be a static jit argument.
    """
    params, ids = setup()
    P = 5
    n_traces = {"n": 0}

    def step(params, token, cache, pos):
        n_traces["n"] += 1                         # runs at trace time only
        return decode_step(params, token, cache, pos, CONFIG)

    step_jit = jax.jit(step)

    _, cache = prefill(params, ids[:P], init_cache(CONFIG), CONFIG)
    for pos in range(P, S_MAX):                    # 11 calls, pos = 5..15
        logits, cache = step_jit(params, ids[pos:pos + 1], cache, jnp.int32(pos))
        ref = forward(ids[:pos + 1], params, CONFIG)[-1]
        np.testing.assert_allclose(logits[0], ref, **TOL, err_msg=f"jit mismatch at pos={pos}")

    assert n_traces["n"] == 1, f"decode_step was traced {n_traces['n']} times"
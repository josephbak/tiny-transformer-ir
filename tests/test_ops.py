import jax
import jax.numpy as jnp
import numpy as np

from tinytr.ops import linear, softmax
from tinytr.ops import layer_norm

# float32 carries ~7 significant digits; 1e-6 allows for 1-ulp rounding differences
RTOL = 1e-6


# ---------- linear ----------

def test_linear_shape_batched():
    # [..., n] @ [n, m] + [m] -> [..., m]: leading dims pass through unchanged
    x = jnp.ones((2, 8, 64))
    w = jnp.ones((64, 16))
    b = jnp.zeros((16,))
    assert linear(x, w, b).shape == (2, 8, 16)


def test_linear_small_numbers():
    # hand-checkable: [1, 2] @ [[1, 2], [3, 4]] + [10, 20]
    #   = [1*1 + 2*3, 1*2 + 2*4] + [10, 20] = [7, 10] + [10, 20] = [17, 30]
    x = jnp.array([[1.0, 2.0]])
    w = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    b = jnp.array([10.0, 20.0])
    np.testing.assert_allclose(linear(x, w, b), [[17.0, 30.0]], rtol=RTOL)


# ---------- softmax ----------

def test_softmax_matches_reference():
    # rows differ, so a bug that normalizes over the wrong axis would be caught
    x = jnp.array([[1.0, 2.0, 3.0], [10.0, 0.0, -5.0]])
    np.testing.assert_allclose(softmax(x), jax.nn.softmax(x), rtol=RTOL)


def test_softmax_rows_sum_to_one():
    x = jax.random.normal(jax.random.key(0), (4, 10))
    sums = jnp.sum(softmax(x), axis=-1)
    np.testing.assert_allclose(sums, jnp.ones(4), rtol=RTOL)


def test_softmax_large_inputs_no_overflow():
    # naive exp(1000) overflows to inf -> inf/inf = nan
    out = softmax(jnp.array([1000.0, 1001.0]))
    assert jnp.all(jnp.isfinite(out))
    np.testing.assert_allclose(out, jax.nn.softmax(jnp.array([0.0, 1.0])), rtol=RTOL)


def test_softmax_very_negative_inputs_no_nan():
    # naive exp(-1000) underflows to 0 -> 0/0 = nan
    out = softmax(jnp.array([-1000.0, -999.0]))
    assert jnp.all(jnp.isfinite(out))
    np.testing.assert_allclose(out, jax.nn.softmax(jnp.array([0.0, 1.0])), rtol=RTOL)


def test_softmax_shift_invariance():
    # softmax(x + c) == softmax(x): the e^c factor cancels
    x = jax.random.normal(jax.random.key(1), (3, 5))
    np.testing.assert_allclose(softmax(x + 7.5), softmax(x), rtol=RTOL)


def test_softmax_other_axis():
    # normalizing over axis 0 must make each COLUMN sum to 1
    x = jax.random.normal(jax.random.key(2), (4, 6))
    out = softmax(x, axis=0)
    np.testing.assert_allclose(jnp.sum(out, axis=0), jnp.ones(6), rtol=RTOL)


# ---------- layer_norm ----------

def test_layer_norm_rows_standardized():
    x = jax.random.normal(jax.random.key(3), (8, 64)) * 5.0 + 2.0   # arbitrary scale and offset
    out = layer_norm(x, jnp.ones(64), jnp.zeros(64))
    np.testing.assert_allclose(jnp.mean(out, axis=-1), jnp.zeros(8), atol=1e-5)
    np.testing.assert_allclose(jnp.var(out, axis=-1), jnp.ones(8), rtol=1e-4)


def test_layer_norm_hand_example():
    # mean 5, variance 5 → (x - 5) / sqrt(5)
    out = layer_norm(jnp.array([[2.0, 4.0, 6.0, 8.0]]), jnp.ones(4), jnp.zeros(4))
    np.testing.assert_allclose(out, [[-1.3416, -0.4472, 0.4472, 1.3416]], rtol=1e-4)


def test_layer_norm_gamma_beta():
    # with constant gamma=g, beta=b: each row's mean becomes b, its variance g**2
    x = jax.random.normal(jax.random.key(4), (8, 64))
    out = layer_norm(x, jnp.full(64, 3.0), jnp.full(64, 0.5))
    np.testing.assert_allclose(jnp.mean(out, axis=-1), jnp.full(8, 0.5), atol=1e-5)
    np.testing.assert_allclose(jnp.var(out, axis=-1), jnp.full(8, 9.0), rtol=1e-4)
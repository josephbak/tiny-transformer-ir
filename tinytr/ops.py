import jax
import jax.numpy as jnp

def linear(x: jax.Array, w: jax.Array, b: jax.Array) -> jax.Array: # [..., n] → [..., m]
    return x @ w + b
    # return jnp.dot(x, w) + b
    

def softmax(x: jax.Array, axis: int = -1) -> jax.Array: # [..., n]
    # Step 1: Find the maximum value along the target axis
    x_max = jnp.max(x, axis=axis, keepdims=True)
    # Step 2: Shift inputs and compute the exponentials
    shifted_exp = jnp.exp(x - x_max)
    # Step 3: Compute the sum of the exponentials along the axis
    exp_sum = jnp.sum(shifted_exp, axis=axis, keepdims=True)
    # Step 4: Normalize to get final probabilities
    return shifted_exp / exp_sum


def softmax_naive(x: jax.Array, axis: int = -1) -> jax.Array: # [..., n]
    x_exp = jnp.exp(x)
    exp_sum = jnp.sum(x_exp, axis=axis, keepdims=True)
    return x_exp / exp_sum

def layer_norm(x, gamma, beta, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    # variance = jnp.var(x, axis=-1, keepdims=True) # this calculates mean again
    variance = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    normalized_x = (x - mean) / jnp.sqrt(variance + eps)
    return gamma * normalized_x + beta
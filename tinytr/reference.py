"""Slow, loop-based reference implementations.

These follow the per-token definitions directly, so they are easy to check
against the math by eye. They are only for tests and experiments, never for speed.
"""

import jax.numpy as jnp

from tinytr.ops import softmax


def attention_loop(x, wq, wk, wv):
    """Single-head attention, one token at a time.  x: [S, D] -> [S, d_k]"""
    q = x @ wq                      # [S, d_k]
    k = x @ wk                      # [S, d_k]
    v = x @ wv                      # [S, d_k]
    S, d_k = q.shape

    out_rows = []
    for i in range(S):                                  # each query token i
        scores_i = jnp.stack(
            [jnp.dot(q[i], k[j]) / jnp.sqrt(d_k) for j in range(S)]
        )                                               # [S]: token i vs every token j
        weights_i = softmax(scores_i)                   # [S]: sums to 1
        out_i = sum(weights_i[j] * v[j] for j in range(S))  # [d_k]: weighted mix of values
        out_rows.append(out_i)

    return jnp.stack(out_rows)                          # [S, d_k]
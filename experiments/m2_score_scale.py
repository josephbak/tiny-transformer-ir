import jax
import jax.numpy as jnp

from tinytr.ops import softmax

D, DK, S = 64, 16, 8


def attention_weights(x, wq, wk):
    q, k = x @ wq, x @ wk
    scores = (q @ k.T) / jnp.sqrt(DK)
    return scores, softmax(scores)


def main():
    kx, kq, kk = jax.random.split(jax.random.key(0), 3)
    x = jax.random.normal(kx, (S, D))
    wq_raw = jax.random.normal(kq, (D, DK))
    wk_raw = jax.random.normal(kk, (D, DK))

    for label, scale in [("unscaled weights", 1.0), ("weights / sqrt(D)", 1 / jnp.sqrt(D))]:
        scores, weights = attention_weights(x, wq_raw * scale, wk_raw * scale)
        print(f"--- {label} ---")
        print("score std:", float(jnp.std(scores)))
        print("first row of weights:", jnp.round(weights[0], 3))
        print("max weight per row:  ", jnp.round(weights.max(axis=-1), 3))
        print()


if __name__ == "__main__":
    main()
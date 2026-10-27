from pathlib import Path

import jax

from tinytr.attention import attention, init_attention_weights
from tinytr.reference import attention_loop

S, D, DK = 8, 64, 16


def main():
    kx, kw = jax.random.split(jax.random.key(0))
    x = jax.random.normal(kx, (S, D))
    wq, wk, wv = init_attention_weights(kw, D, DK)

    out_dir = Path("ir/m2")
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, fn in [("vectorized", attention), ("loop", attention_loop)]:
        jaxpr = jax.make_jaxpr(fn)(x, wq, wk, wv)
        n_eqns = len(jaxpr.jaxpr.eqns)          # number of recorded operations
        # (out_dir / f"attention_{name}.jaxpr.txt").write_text(str(jaxpr))
        (out_dir / f"attention_{name}.jaxpr.txt").write_text(jaxpr.pretty_print(use_color=False))
        print(f"{name:>10}: {n_eqns} operations")


if __name__ == "__main__":
    main()
from pathlib import Path

import jax
import jax.numpy as jnp

from tinytr.ops import softmax


def main():
    x = jnp.array([1.0, 2.0, 3.0])

    # trace softmax with x's shape/dtype (float32[3]) and get the recorded IR
    jaxpr = jax.make_jaxpr(softmax)(x)
    print(jaxpr)

    # save it next to the other IR artifacts
    out_dir = Path("ir/m1")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "softmax.jaxpr.txt").write_text(str(jaxpr))


if __name__ == "__main__":
    main()
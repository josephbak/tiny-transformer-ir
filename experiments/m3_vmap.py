from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from tinytr.attention import attention, init_attention_weights

B, S, D, DK = 2, 8, 64, 16


def main():
    kx, kw = jax.random.split(jax.random.key(0))
    xb = jax.random.normal(kx, (B, S, D))               # a batch of B sequences
    wq, wk, wv = init_attention_weights(kw, D, DK)      # shared by every sequence

    # fix the static flag first: vmap would otherwise try to batch the keyword argument
    causal_attention = partial(attention, causal=True)

    # in_axes: x is batched along axis 0; the three weights are shared (None)
    batched = jax.vmap(causal_attention, in_axes=(0, None, None, None))

    # 1. shape check
    out = batched(xb, wq, wk, wv)
    print("input  shape:", xb.shape)                    # (2, 8, 64)
    print("output shape:", out.shape)                   # (2, 8, 16)

    # 2. same result as looping over the batch by hand
    looped = jnp.stack([causal_attention(xb[b], wq, wk, wv) for b in range(B)])
    np.testing.assert_allclose(out, looped, rtol=1e-6)
    print("vmap matches per-sequence loop: OK")

    # 3. compiler checkpoint: compare the jaxpr for one sequence vs the vmapped batch
    out_dir = Path("ir/m3")
    out_dir.mkdir(parents=True, exist_ok=True)

    single = jax.make_jaxpr(causal_attention)(xb[0], wq, wk, wv)
    batch = jax.make_jaxpr(batched)(xb, wq, wk, wv)

    for name, jaxpr in [("single", single), ("vmap", batch)]:
        text = jaxpr.pretty_print(use_color=False)
        (out_dir / f"attention_causal_{name}.jaxpr.txt").write_text(text)
        print(f"{name:>6}: {len(jaxpr.jaxpr.eqns)} operations")

    # the interesting lines: how each matmul's dimension numbers changed
    print("\ndot_general dimension_numbers after vmap:")
    for eqn in batch.jaxpr.eqns:
        if eqn.primitive.name == "dot_general":
            print("  ", eqn.params["dimension_numbers"])


if __name__ == "__main__":
    main()
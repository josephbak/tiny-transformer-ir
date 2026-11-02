import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from tinytr.model import (CONFIG, forward, forward_scan, forward_scan_prestacked,
                          init_model_params, stack_blocks)


def stats(fn, ids, params):
    jaxpr = jax.make_jaxpr(fn)(ids, params)
    t0 = time.perf_counter()
    lowered = jax.jit(fn).lower(ids, params)
    lowered.compile()
    compile_s = time.perf_counter() - t0
    return len(jaxpr.jaxpr.eqns), lowered.as_text().count("\n"), compile_s


def main():
    ids = jnp.arange(8)
    print(f"{'L':>3} | {'loop ops':>8} {'loop lines':>10} {'loop s':>7} | "
          f"{'scan ops':>8} {'scan lines':>10} {'scan s':>7}")
    for L in [2, 8, 16]:
        cfg = {**CONFIG, "n_layers": L}
        params = init_model_params(jax.random.key(0), cfg)

        # same numbers either way
        np.testing.assert_allclose(forward(ids, params, cfg), forward_scan(ids, params, cfg),
                                   rtol=1e-5, atol=1e-5)

        lo = stats(partial(forward, cfg=cfg), ids, params)
        sc = stats(partial(forward_scan, cfg=cfg), ids, params)
        print(f"{L:>3} | {lo[0]:>8} {lo[1]:>10} {lo[2]:>7.2f} | {sc[0]:>8} {sc[1]:>10} {sc[2]:>7.2f}")

        # pre = {**params, "blocks": stack_blocks(params["blocks"])}     # stack ONCE, outside jit
        # ps = stats(partial(forward_scan_prestacked, cfg=cfg), ids, pre)
        # print(f"     pre-stacked scan: {ps[0]} ops, {ps[1]} lines")
        # print("\n")



if __name__ == "__main__":
    main()
"""M10 step 7: time XLA's decode_step on this machine, for comparison with iree-benchmark-module.

Same inputs as experiments.m10_export. Reports the median time per call over many calls,
plus an empty jitted call as the fixed-overhead baseline.

Run from the repo root:  uv run python -m experiments.m10_bench_xla
"""

import statistics
import time

import jax
import jax.numpy as jnp

from experiments.m10_export import make_inputs
from tinytr.kv_cache import _decode_jit
from tinytr.model import CONFIG

N_WARMUP, N_CALLS = 200, 5000


def median_us(step):
    """Median wall time of one call, in µs. `step()` must run one call and block until it's done."""
    for _ in range(N_WARMUP):
        step()
    times = []
    for _ in range(N_CALLS):
        t0 = time.perf_counter_ns()
        step()
        times.append(time.perf_counter_ns() - t0)
    return statistics.median(times) / 1e3


def main():
    params, token, cache, pos = make_inputs(CONFIG)
    state = {"cache": cache}

    def decode():
        # the cache is donated, so feed the returned cache back in, as generate() does
        logits, state["cache"] = _decode_jit(params, token, state["cache"], pos)
        logits.block_until_ready()

    empty_jit = jax.jit(lambda x: x)
    x = jnp.int32(5)

    def empty():
        empty_jit(x).block_until_ready()

    print(f"XLA decode_step: {median_us(decode):7.1f} µs  (median of {N_CALLS} calls)")
    print(f"XLA empty call:  {median_us(empty):7.1f} µs")


if __name__ == "__main__":
    main()
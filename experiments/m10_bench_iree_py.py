"""M10 step 7: time IREE's decode_step from Python, to match how experiments.m10_bench_xla times XLA.

Uses the local-sync driver. Inputs are put on the device once; the returned k/v are fed back in
each call (the cache is donated), as m10_bench_xla does for XLA.
Needs ir/m10/decode_step.vmfb, decode_step_ref.npz, and ir/m10/bench/empty.vmfb (made by
scripts/m10_bench_iree.sh).

Run from the repo root:  uv run python -m experiments.m10_bench_iree_py
"""

import statistics
import time

import iree.runtime as ireert
import numpy as np

N_WARMUP, N_CALLS = 200, 5000
K_IN, V_IN = 18, 19                      # flat input indices of cache k, v (outputs 1, 2)


def load_main(config, path):
    """Load a .vmfb into its own context and return its `main` function."""
    ctx = ireert.SystemContext(config=config)
    module = ireert.VmModule.mmap(config.vm_instance, path)
    ctx.add_vm_module(module)
    return ctx.modules[module.name]["main"]


def median_us(step):
    """Median wall time of one call to step(), in µs."""
    for _ in range(N_WARMUP):
        step()
    times = []
    for _ in range(N_CALLS):
        t0 = time.perf_counter_ns()
        step()
        times.append(time.perf_counter_ns() - t0)
    return statistics.median(times) / 1e3


def main():
    config = ireert.Config("local-sync")
    ref = np.load("ir/m10/decode_step_ref.npz")
    # Put the 21 inputs on the device once, so the timed loop doesn't copy from NumPy each call.
    inputs = [ireert.asdevicearray(config.device, ref[f"in_{i:02d}"]) for i in range(len(ref["in_names"]))]

    decode_main = load_main(config, "ir/m10/decode_step.vmfb")

    def decode():
        _, k, v = decode_main(*inputs)    # local-sync: the call returns when the work is done
        inputs[K_IN], inputs[V_IN] = k, v  # feed the updated (donated) cache back in

    empty_main = load_main(config, "ir/m10/bench/empty.vmfb")
    x = ireert.asdevicearray(config.device, np.int32(5))

    def empty():
        empty_main(x)

    print(f"IREE decode_step (from Python, local-sync): {median_us(decode):7.1f} µs  (median of {N_CALLS} calls)")
    print(f"IREE empty call  (from Python, local-sync): {median_us(empty):7.1f} µs")


if __name__ == "__main__":
    main()
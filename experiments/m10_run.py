"""M10 step 4: run the compiled IREE module on the saved inputs and compare with JAX's outputs.

Needs, in ir/m10/:
  decode_step.vmfb       from step 3 (iree-compile)
  decode_step_ref.npz    from step 2 (experiments.m10_export)

Run from the repo root:  uv run python -m experiments.m10_run
"""

import pathlib
import sys

import iree.runtime as ireert
import numpy as np

IR = pathlib.Path("ir/m10")
RTOL = ATOL = 1e-5
OUT_NAMES = ["logits", "k", "v"]


def main():
    # The problem and the answer key, saved by the export script.
    ref = np.load(IR / "decode_step_ref.npz")
    names = list(ref["in_names"])
    inputs = [ref[f"in_{i:02d}"] for i in range(len(names))]   # same flat order as the export
    expected = [ref[f"out_{j}"] for j in range(len(OUT_NAMES))]

    # Decision 1: Python runtime. Decision 2: local-sync = single-threaded.
    module = ireert.load_vm_flatbuffer_file(str(IR / "decode_step.vmfb"), driver="local-sync")
    results = module.main(*inputs)
    got = [np.asarray(r.to_host()) for r in results]

    # Decision 3: per output, shape check + largest absolute difference + allclose.
    assert len(got) == len(expected), f"IREE returned {len(got)} outputs, expected {len(expected)}"
    ok_all = True
    print(f"{'output':<8}{'shape':<18}{'max |IREE - JAX|':>18}  allclose")
    for name, g, e in zip(OUT_NAMES, got, expected):
        assert g.shape == e.shape and g.dtype == e.dtype, f"{name}: {g.shape} {g.dtype} vs {e.shape} {e.dtype}"
        diff = float(np.max(np.abs(g - e)))
        ok = bool(np.allclose(g, e, rtol=RTOL, atol=ATOL))
        ok_all &= ok
        print(f"{name:<8}{str(g.shape):<18}{diff:>18.3e}  {ok}")

    # The cache write itself: slot pos must have changed from the input, and match JAX.
    pos = int(inputs[names.index("[3]")])
    k_in, k_out = inputs[names.index("[2]['k']")], got[1]
    print(f"\nslot {pos} of k: input max |k| = {np.abs(k_in[:, pos]).max():.3e}, "
          f"IREE output max |k| = {np.abs(k_out[:, pos]).max():.3e}")

    print("\nPASS" if ok_all else "\nFAIL")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
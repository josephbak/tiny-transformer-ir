"""M10 step 8: export prefill from JAX as StableHLO for IREE, and dump XLA's optimized HLO for it.

Same setup as M8: key(0) weights, prompt [17, 3, 42, 8, 91] (P = 5), empty cache, prefill
jitted without donation. Writes, into ir/m10/:
  prefill.mlirbc            versioned StableHLO bytecode (input to iree-compile)
  prefill.stablehlo.mlir    the same module as readable text
  prefill_ref.npz           the 20 flat inputs + JAX's 3 outputs (reference for m10_run)
  prefill.xla_opt.hlo       XLA's optimized HLO (for comparing kernels)

Run from the repo root:  uv run python -m experiments.m10_export_prefill
"""

import collections
import functools
import pathlib
import re

import jax
import jax.numpy as jnp
import numpy as np
from jax import export

from experiments.m10_xla_dump import BOOKKEEPING, entry_instructions
from tinytr.kv_cache import init_cache, prefill
from tinytr.model import CONFIG, init_model_params

OUT = pathlib.Path("ir/m10")
_prefill_jit = jax.jit(functools.partial(prefill, cfg=CONFIG))   # as in M8: no donation


def make_inputs(cfg):
    params = init_model_params(jax.random.key(0), cfg)
    prompt = jnp.array([17, 3, 42, 8, 91])                        # P = 5
    return params, prompt, init_cache(cfg)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    args = make_inputs(CONFIG)

    with_path, _ = jax.tree_util.tree_flatten_with_path(args)
    names = [jax.tree_util.keystr(path) for path, _ in with_path]
    flat_in = [np.array(leaf) for _, leaf in with_path]

    # StableHLO for IREE
    exp = export.export(_prefill_jit)(*args)
    assert exp.in_tree == jax.tree.structure((args, {})), "export flattened the inputs differently"
    (OUT / "prefill.mlirbc").write_bytes(exp.mlir_module_serialized)
    (OUT / "prefill.stablehlo.mlir").write_text(exp.mlir_module())

    # JAX's reference outputs
    logits, cache_out = _prefill_jit(*args)
    flat_out = [np.asarray(x) for x in jax.tree.leaves((logits, cache_out))]
    np.savez(OUT / "prefill_ref.npz",
             **{f"in_{i:02d}": x for i, x in enumerate(flat_in)},
             **{f"out_{j}": x for j, x in enumerate(flat_out)},
             in_names=np.array(names))

    # XLA's optimized HLO, and its kernel count
    hlo = _prefill_jit.lower(*args).compile().as_text()
    (OUT / "prefill.xla_opt.hlo").write_text(hlo)
    instrs = entry_instructions(hlo)
    kernels = [(n, op) for n, op in instrs if op not in BOOKKEEPING]
    by_kind = collections.Counter(re.sub(r"(\.\d+)+$", "", n) if op == "fusion" else op for n, op in kernels)

    print(f"{len(flat_in)} inputs: " + ", ".join(f"{n} {x.shape}" for n, x in zip(names, flat_in) if not n.startswith("[0]")))
    print(f"{len(flat_out)} outputs: " + ", ".join(str(x.shape) for x in flat_out))
    print(f"\nXLA: {len(instrs)} top-level instructions, {len(kernels)} kernel-launching")
    for kind, n in by_kind.most_common():
        print(f"  {n:3d}  {kind}")
    print(f"\nwrote {OUT}/prefill.mlirbc, prefill.stablehlo.mlir, prefill_ref.npz, prefill.xla_opt.hlo")


if __name__ == "__main__":
    main()
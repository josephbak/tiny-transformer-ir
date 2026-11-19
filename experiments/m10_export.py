"""M10 step 2: export decode_step from JAX as StableHLO, for IREE.

Writes, into ir/m10/:
  decode_step.mlirbc          versioned StableHLO bytecode (input to iree-compile)
  decode_step.stablehlo.mlir  the same module as readable text
  decode_step_ref.npz         the 21 flat inputs + JAX's 3 outputs (reference for step 4)

Run from the repo root:  uv run python -m experiments.m10_export
"""

import pathlib
import re

import jax
import jax.numpy as jnp
import numpy as np
from jax import export

from tinytr.kv_cache import _decode_jit, init_cache, prefill
from tinytr.model import CONFIG, init_model_params

OUT = pathlib.Path("ir/m10")


def make_inputs(cfg):
    """Decision 2: M8's decode setup. key(0) weights, prefill P = 5, then decode token [7] at pos 5."""
    params = init_model_params(jax.random.key(0), cfg)
    prompt = jnp.array([17, 3, 42, 8, 91])
    _, cache = prefill(params, prompt, init_cache(cfg), cfg)    # a real cache, not zeros
    return params, jnp.array([7]), cache, jnp.int32(prompt.shape[0])   # pos = next free slot = P


def aliased_inputs(text):
    """Decision 4: {flat input index: output index} from `tf.aliasing_output` in the signature of @main."""
    sig = text[text.index("func.func public @main("):]
    sig = sig[:sig.index("{\n")]                                # stop at the start of the function body
    return {int(a): int(o) for a, o in re.findall(r"%arg(\d+): [^%]*?tf\.aliasing_output = (\d+)", sig)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    args = make_inputs(CONFIG)

    # Decision 3: the flat argument order comes from JAX, never hard-coded.
    with_path, _ = jax.tree_util.tree_flatten_with_path(args)
    names = [jax.tree_util.keystr(path) for path, _ in with_path]
    # Decision 5 trap: copy the inputs to NumPy now; calling _decode_jit deletes the donated cache.
    flat_in = [np.array(leaf) for _, leaf in with_path]    # np.array copies; np.asarray would only view
    # flat_in = [np.asarray(leaf) for _, leaf in with_path]

    # Decision 1: params are passed as arguments (nothing is closed over), so they stay runtime buffers.
    exp = export.export(_decode_jit)(*args)
    assert exp.in_tree == jax.tree.structure((args, {})), "export flattened the inputs differently"
    text = exp.mlir_module()
    blob = exp.mlir_module_serialized

    # Decision 4: donation must survive the export: exactly cache k and v, aliased to outputs 1 and 2.
    alias = aliased_inputs(text)
    assert {names[i] for i in alias} == {"[2]['k']", "[2]['v']"}, f"unexpected aliasing: {alias}"

    # Decision 5: save the compile input, the readable text, and the reference values.
    version = re.search(rb"StableHLO_v[\d.]+", blob[:64]).group().decode()
    (OUT / "decode_step.mlirbc").write_bytes(blob)
    (OUT / "decode_step.stablehlo.mlir").write_text(text)

    logits, cache_out = _decode_jit(*args)                     # JAX reference (donates args' cache)
    flat_out = [np.asarray(x) for x in jax.tree.leaves((logits, cache_out))]
    np.savez(
        OUT / "decode_step_ref.npz",
        **{f"in_{i:02d}": x for i, x in enumerate(flat_in)},
        **{f"out_{j}": x for j, x in enumerate(flat_out)},
        in_names=np.array(names),
    )

    # Report
    print(f"bytecode: {version}, {len(blob):,} B    text: {len(text):,} chars")
    print(f"\n{len(flat_in)} inputs (flat order):")
    for i, (name, x) in enumerate(zip(names, flat_in)):
        note = f"  -> aliases output {alias[i]}" if i in alias else ""
        print(f"  {i:2d}  {name:<30} {str(x.shape):<16} {x.dtype}{note}")
    print(f"\n{len(flat_out)} outputs:")
    for j, x in enumerate(flat_out):
        print(f"  {j}  {str(x.shape):<16} {x.dtype}")
    print(f"\nwrote {OUT}/decode_step.mlirbc, decode_step.stablehlo.mlir, decode_step_ref.npz")


if __name__ == "__main__":
    main()
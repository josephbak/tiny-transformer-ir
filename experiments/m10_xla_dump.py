"""M10 step 6: save XLA's optimized HLO for decode_step and count its top-level instructions.

Same inputs as experiments.m10_export. Writes ir/m10/decode_step.xla_opt.hlo.

Run from the repo root:  uv run python -m experiments.m10_xla_dump
"""

import collections
import pathlib
import re

from experiments.m10_export import make_inputs
from tinytr.kv_cache import _decode_jit
from tinytr.model import CONFIG

OUT = pathlib.Path("ir/m10")

# Top-level instructions that launch no kernel: they name, reinterpret, or package buffers.
BOOKKEEPING = {"parameter", "constant", "bitcast", "tuple", "get-tuple-element"}


def entry_instructions(hlo):
    """(name, opcode) for each instruction in the ENTRY computation (not inside fusions)."""
    body = hlo[hlo.index("\nENTRY "):]
    body = body[body.index("{") + 1: body.index("\n}")]
    # a line looks like:  name.12 = f32[1,64]{1,0} opcode(operands), attrs...   (ROOT optional)
    pat = re.compile(r"^\s*(?:ROOT\s+)?%?([\w.\-]+)\s*=\s*.*?\s([a-z][\w\-]*)\(", re.M)
    return pat.findall(body)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    args = make_inputs(CONFIG)
    hlo = _decode_jit.lower(*args).compile().as_text()        # optimized HLO, after fusion
    (OUT / "decode_step.xla_opt.hlo").write_text(hlo)

    instrs = entry_instructions(hlo)
    by_opcode = collections.Counter(op for _, op in instrs)
    kernels = [(n, op) for n, op in instrs if op not in BOOKKEEPING]
    # group kernels by name with the trailing .N removed (e.g. slice_bitcast_fusion.7 -> slice_bitcast_fusion)
    by_kind = collections.Counter(re.sub(r"(\.\d+)+$", "", n) if op == "fusion" else op for n, op in kernels)

    print(f"top-level instructions: {len(instrs)}")
    print(f"  bookkeeping (no kernel): {sum(by_opcode[o] for o in BOOKKEEPING)}  "
          + ", ".join(f"{o} {by_opcode[o]}" for o in sorted(BOOKKEEPING) if by_opcode[o]))
    print(f"  kernel-launching:        {len(kernels)}")
    print("\nkernel-launching, by kind:")
    for kind, n in by_kind.most_common():
        print(f"  {n:3d}  {kind}")
    print(f"\nwrote {OUT}/decode_step.xla_opt.hlo")


if __name__ == "__main__":
    main()
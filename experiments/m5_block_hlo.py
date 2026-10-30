import re
from functools import partial
from pathlib import Path

import jax

from tinytr.block import init_block_params, transformer_block

S, D, F, H = 8, 64, 256, 4


def main():
    kx, kp = jax.random.split(jax.random.key(0))
    x = jax.random.normal(kx, (S, D))
    params = init_block_params(kp, D, F)     # a pytree: jit traces each leaf as an input

    block = partial(transformer_block, n_heads=H, causal=True)

    lowered = jax.jit(block).lower(x, params)
    compiled = lowered.compile()

    out_dir = Path("ir/m5")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "block.stablehlo.mlir").write_text(lowered.as_text())          # before XLA
    (out_dir / "block.optimized.hlo.txt").write_text(compiled.as_text())      # after XLA

    stablehlo, hlo = lowered.as_text(), compiled.as_text()

    # ops in StableHLO (one line per op, roughly)
    n_stablehlo = len(re.findall(r"= stablehlo\.", stablehlo))
    print(f"StableHLO ops:            {n_stablehlo}")

    # fusions in the optimized HLO, grouped by kind
    kinds = re.findall(r"kind=(k\w+)", hlo)
    print(f"optimized HLO fusions:    {len(kinds)}")
    for kind in sorted(set(kinds)):
        print(f"  {kind:<10} {kinds.count(kind)}")

    # a few things worth checking by eye in the optimized HLO
    for label, pattern in [
        ("dot ops", r"\bdot\("),
        ("custom-calls", r"custom-call\("),
        ("transposes", r"\btranspose\("),
        ("sqrt ops (should be folded away)", r"\bsqrt\("),
    ]:
        print(f"{label:<34} {len(re.findall(pattern, hlo))}")

    print("\nwrote ir/m5/block.stablehlo.mlir and ir/m5/block.optimized.hlo.txt")


if __name__ == "__main__":
    main()
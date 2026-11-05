"""M7 step 1: dump XLA's HLO after every optimization pass for one Transformer block."""

import os
import re
import shutil
from functools import partial
from pathlib import Path

# XLA reads XLA_FLAGS once, when JAX first starts its backend, so set it BEFORE importing jax
DUMP_DIR = Path("ir/xla/m7_block")
shutil.rmtree(DUMP_DIR, ignore_errors=True)             # start clean on every run
os.environ["XLA_FLAGS"] = (
    f"--xla_dump_to={DUMP_DIR} "                        # where to write the dumps
    "--xla_dump_hlo_pass_re=.*"                         # dump after EVERY pass, not just before/after
)

import jax  # noqa: E402  (must come after setting XLA_FLAGS)

from tinytr.block import init_block_params, transformer_block  # noqa: E402

S, D, F, H = 8, 64, 256, 4


def main():
    kx, kp = jax.random.split(jax.random.key(0))
    x = jax.random.normal(kx, (S, D))
    params = init_block_params(kp, D, F)

    block = partial(transformer_block, n_heads=H, causal=True)
    jax.jit(block).lower(x, params).compile()           # compiling triggers the dumps

    # other small programs (e.g. the random-number setup) get dumped too; keep only the block's
    files = sorted(p.name for p in DUMP_DIR.iterdir() if "transformer_block" in p.name)
    print(f"{len(files)} dump files for the block\n")

    # per-pass files are named like  module_NNNN.jit_transformer_block.NNNN.<pipeline>.after_<pass>.before_<next>.txt
    # print each one's sequence number, pipeline, and pass, in order
    pattern = re.compile(r"\.(\d{4})\.([\w\-]+)\.after_([\w\-]+)\.before_")
    for name in files:
        m = pattern.search(name)
        if m:
            print(f"{m.group(1)}  {m.group(2):<32} after {m.group(3)}")
        else:
            print(f"      {name}")                      # before/after-optimizations and other files


if __name__ == "__main__":
    main()
from functools import partial
from pathlib import Path

import jax

from tinytr.attention import init_mha_weights, multi_head_attention

S, D, H = 8, 64, 4


def main():
    kx, kw = jax.random.split(jax.random.key(0))
    x = jax.random.normal(kx, (S, D))
    wq, wk, wv, wo = init_mha_weights(kw, D)

    # n_heads decides shapes (the reshape), and causal decides control flow (the if),
    # so both must be fixed before tracing: they are configuration, not data
    mha = partial(multi_head_attention, n_heads=H, causal=True)

    out_dir = Path("ir/m4")
    out_dir.mkdir(parents=True, exist_ok=True)

    # level 1: jaxpr
    jaxpr = jax.make_jaxpr(mha)(x, wq, wk, wv, wo)
    (out_dir / "mha.jaxpr.txt").write_text(jaxpr.pretty_print(use_color=False))

    # level 2: StableHLO (MLIR), what JAX hands to XLA
    stablehlo = jax.jit(mha).lower(x, wq, wk, wv, wo).as_text()
    (out_dir / "mha.stablehlo.mlir").write_text(stablehlo)

    # quick summary of the ops worth reading
    for op in ["stablehlo.dot_general", "stablehlo.transpose", "stablehlo.reshape",
               "stablehlo.reduce", "stablehlo.select"]:
        print(f"{op:>24}: {stablehlo.count(op + ' ') + stablehlo.count(op + '(')}")

    print("\nwrote ir/m4/mha.jaxpr.txt and ir/m4/mha.stablehlo.mlir")


if __name__ == "__main__":
    main()
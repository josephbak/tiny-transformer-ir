# tiny-transformer-ir

A decoder-only Transformer written from scratch in `jax.numpy` (no Flax or Optax),
followed through JAX's compilation stack — jaxpr → StableHLO → optimized XLA HLO —
and the same StableHLO through IREE,
with hand-derived FLOP/byte accounting and a KV-cache decode path.
Includes a PyTorch port of the same model, for comparing frontends.

## Goals
- Understand the model: the shapes, data flow, and cost of every operation
- Understand compilation: fusion, layouts, compute-bound vs. memory-bound phases
- Compare compilers: what XLA and IREE build from the same StableHLO (kernels, copies, layout, time)
- Compare frontends: the same model through JAX/StableHLO and PyTorch/torch-mlir

## Status
- [x] M0 — Setup
- [x] M1 — JAX basics: stable softmax, `jit` tracing and retracing, jaxpr
- [x] M2 — Single-head attention
- [x] M3 — Causal mask, batching with `vmap`
- [x] M4 — Multi-head attention
- [x] M5 — Transformer block
- [x] M6 — Full model
- [x] M7 — Reading the compiler's output
- [x] M8 — Cost accounting: prefill vs decode, XLA `cost_analysis` check, copy variants, M4 timing (done after M9)
- [x] M9 — KV cache: prefill/decode + equivalence tests + generate
- [x] M10 — Into IREE: decode/prefill through `iree-compile`, IREE dispatches vs XLA fusions, benchmark (torch-mlir comparison not done)
- [x] PyTorch port of the model (`gpt_torch/`)

## Run
    uv sync
    uv run pytest

The M10 experiments also need IREE's compiler and runtime:

    uv sync --group iree

Experiments run as modules from the project root, e.g.:

    uv run python -m experiments.m1_softmax_ir
    uv run python -m experiments.m10_export

Shell scripts run from the project root too:

    bash scripts/m10_bench_iree.sh

The PyTorch port runs cell by cell in VS Code (`# %%` cells), or as a script:

    uv run python gpt_torch/model.py

## Layout
- `tinytr/` — the JAX model: ops, attention, block, model, KV cache
- `gpt_torch/` — the same model in PyTorch
- `tests/` — shape, reference, and causality tests
- `experiments/` — scripts that trace, dump IR, and analyze cost
- `scripts/` — shell scripts (e.g. the IREE benchmark)
- `ir/` — dumped jaxpr / StableHLO / HLO / IREE IR text, organized by milestone (binary outputs such as `.vmfb` and `.npz` are not committed; the scripts rebuild them)
- `docs/NOTATION.md` — names and shapes, JAX vs. PyTorch

Full milestone plan: [docs/PLAN.md](docs/PLAN.md)

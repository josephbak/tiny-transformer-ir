# tiny-transformer-ir

A decoder-only Transformer written from scratch in `jax.numpy` (no Flax or Optax),
followed through JAX's compilation stack — jaxpr → StableHLO → optimized XLA HLO —
with hand-derived FLOP/byte accounting and a KV-cache decode path.
Includes a PyTorch port of the same model, for comparing frontends.

## Goals
- Understand the model: the shapes, data flow, and cost of every operation
- Understand compilation: fusion, layouts, compute-bound vs. memory-bound phases
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
- [ ] M8 — Cost accounting
- [x] M9 — KV cache: prefill/decode + equivalence tests + generate
- [ ] M10 — Into IREE (optional)
- [x] PyTorch port of the model (`gpt_torch/`)

## Run
    uv sync
    uv run pytest

Experiments run as modules from the project root, e.g.:

    uv run python -m experiments.m1_softmax_ir

The PyTorch port runs cell by cell in VS Code (`# %%` cells), or as a script:

    uv run python gpt_torch/model.py

## Layout
- `tinytr/` — the JAX model: ops, attention, block, model, KV cache
- `gpt_torch/` — the same model in PyTorch
- `tests/` — shape, reference, and causality tests
- `experiments/` — scripts that trace, dump IR, and analyze cost
- `ir/` — dumped jaxpr / StableHLO / HLO, organized by milestone
- `docs/NOTATION.md` — names and shapes, JAX vs. PyTorch

Full milestone plan: [docs/PLAN.md](docs/PLAN.md)
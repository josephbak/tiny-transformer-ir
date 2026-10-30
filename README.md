# tiny-transformer-ir

A decoder-only Transformer written from scratch in `jax.numpy` (no Flax or Optax),
followed through JAX's compilation stack — jaxpr → StableHLO → optimized XLA HLO —
with hand-derived FLOP/byte accounting and a KV-cache decode path.

## Goals
- Understand the model: the shapes, data flow, and cost of every operation
- Understand compilation: fusion, layouts, compute-bound vs. memory-bound phases

## Status
- [x] M0 — Setup
- [x] M1 — JAX basics: stable softmax, `jit` tracing and retracing, jaxpr
- [x] M2 — Single-head attention
- [x] M3 — Causal mask, batching with `vmap`
- [x] M4 — Multi-head attention
- [x] M5 — Transformer block
- [ ] M6 — Full model
- [ ] M7 — Reading the compiler's output
- [ ] M8 — Cost accounting
- [ ] M9 — KV cache: prefill vs. decode
- [ ] M10 — Into IREE (optional)

## Run
    uv sync
    uv run pytest

Experiments run as modules from the project root, e.g.:

    uv run python -m experiments.m1_softmax_ir

## Layout
- `tinytr/` — the model: ops, attention, block, model, KV cache
- `tests/` — shape, reference, and causality tests
- `experiments/` — scripts that trace, dump IR, and analyze cost
- `ir/` — dumped jaxpr / StableHLO / HLO, organized by milestone

Full milestone plan: [docs/PLAN.md](docs/PLAN.md)
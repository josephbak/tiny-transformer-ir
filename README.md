# tiny-transformer-ir

A decoder-only Transformer written from scratch in `jax.numpy` (no Flax or Optax),
followed through JAX's compilation stack — jaxpr → StableHLO → optimized XLA HLO —
with hand-derived FLOP/byte accounting and a KV-cache decode path.

## Goals
- Understand the model: the shapes, data flow, and cost of every operation
- Understand compilation: fusion, layouts, compute-bound vs. memory-bound phases

## Status
- [x] M0 — Setup
- [ ] M1 — JAX basics: softmax, `jit`, jaxpr
- [ ] M2 — Single-head attention
- [ ] M3 — Causal mask, batching with `vmap`
- [ ] M4 — Multi-head attention
- [ ] M5 — Transformer block
- [ ] M6 — Full model
- [ ] M7 — Reading the compiler's output
- [ ] M8 — Cost accounting
- [ ] M9 — KV cache: prefill vs. decode
- [ ] M10 — Into IREE (optional)

## Run
    uv sync
    uv run pytest

## Layout
- `model/` — attention, block, model, KV cache
- `tests/` — shape, reference, and causality tests
- `ir/` — dumped jaxpr / StableHLO / HLO
- `notes/` — write-up per milestone

Full milestone plan: [docs/PLAN.md](docs/PLAN.md)
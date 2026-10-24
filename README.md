# tiny-transformer-ir

A decoder-only Transformer written from scratch in `jax.numpy`, followed through the compiler stack. Three goals, in this order of priority:

1. Understand the model: shapes, data flow, and costs of every operation.
2. Understand compilation: what the model becomes in jaxpr, StableHLO, and optimized XLA HLO, and why.
3. Learn JAX along the way, only as much as each milestone needs.

## Rules

- No Flax, no Optax, no `jax.nn` building blocks except as reference checks. Write every layer yourself.
- Random weights, forward pass only. No training.
- Write first, look later. Compare against the Annotated Transformer and nanoGPT only after Milestone 6.
- Every milestone ends with a short note in `notes/` answering its questions in your own words.

## Layout

```
tiny-transformer-ir/
├── README.md
├── model/      # your code: attention, block, model, kv cache
├── tests/      # pytest: shape checks, reference checks, causality
├── ir/         # dumped jaxpr / StableHLO / HLO text
└── notes/      # m0.md, m1.md, … one per milestone
```

## Config (fixed for the whole project)

| Name | Meaning | Value |
|---|---|---|
| `B` | batch size | 2 |
| `S` | sequence length | 8 |
| `V` | vocabulary size | 100 |
| `D` | model width (d_model) | 64 |
| `H` | number of heads | 4 |
| `Dk` | per-head width = D / H | 16 |
| `F` | feed-forward width | 256 |
| `L` | number of layers | 2 |

Small enough that every tensor can be printed and every FLOP counted by hand.

---

## M0 — Setup

- Python virtual environment, then `pip install -U jax pytest`. The CPU build works on both macOS (Apple Silicon or Intel) and Linux; no GPU needed.
- Verify: `python -c "import jax; print(jax.devices())"` prints a CPU device.

**Done when:** an empty `pytest` run passes and JAX imports.

## M1 — JAX basics, through two tiny functions

**Build:** `linear(x, w, b)` and your own `softmax(x)` over the last axis.

**JAX concepts to learn here:**
- `jax.numpy` arrays, shapes, broadcasting, default `float32`
- Arrays are immutable: `x.at[i].set(v)` instead of `x[i] = v`
- Randomness uses explicit keys: `jax.random.key`, `jax.random.split`
- `jax.jit`, and what *tracing* means: why a `print` inside a jitted function fires only once

**Understand:**
- Why a correct softmax subtracts the row max before exponentiating (what overflows without it)
- How to check your softmax against `jax.nn.softmax`

**Compiler checkpoint:** print `jax.make_jaxpr(softmax)(x)`. Find the two reductions (max, sum) and the elementwise operations between them.

**Done when:** tests pass for your softmax against the reference, including large inputs like `x = [1000, 1001]`.

## M2 — Single-head attention, one sequence

**Build:** `attention(x, wq, wk, wv)` for `x` of shape `[S, D]`, no batch yet.

**Understand:**
- The shape of every intermediate: `Q`, `K`, `V`, scores, weights, output
- Why scores are divided by √Dk (what happens to the softmax as Dk grows otherwise)
- The score matrix is `[S, S]`: how its memory grows with sequence length

**Test:** write a slow reference using plain Python loops over positions, and check that the vectorized version matches it.

**Compiler checkpoint:** compare the jaxpr of the loop version (if you `jit` it) with the vectorized one. Loops written in Python get *unrolled* at trace time. Note how many operations each produces.

## M3 — Causal mask and batching with `vmap`

**Build:** add the causal mask. Then batch to `[B, S, D]` using `jax.vmap`, without rewriting the function.

**JAX concept:** `vmap` is a *program transformation*: it rewrites your function to carry an extra batch dimension.

**Test:** causality. Change the token at position `t`; outputs at positions `< t` must not change.

**Compiler checkpoint:** print the jaxpr before and after `vmap`. Where did the batch dimension appear in each operation?

## M4 — Multi-head attention

**Build:** split `D` into `H` heads of width `Dk`, attend per head, merge, and project with `W_O`.

**Understand:**
- The reshape `[B,S,D] → [B,S,H,Dk]` and the transpose to `[B,H,S,Dk]`: which is free (just reinterprets memory) and which may physically move data
- `jnp.einsum` as a way to write the matmuls with named axes

**Compiler checkpoint:** in StableHLO (`jax.jit(f).lower(...).as_text()`), find `stablehlo.dot_general`. For each one, identify its *batching* and *contracting* dimensions and map them back to your einsum strings. Find the `transpose` ops.

## M5 — Transformer block

**Build:** your own LayerNorm, the feed-forward network, and a pre-norm block: residual around attention, residual around FFN.

**JAX concept:** *pytrees*. Store parameters as nested dicts, and write an `init_params(key, config)` that splits keys properly.

**Understand:** why the residual path is left untouched in pre-norm, and which tensors must stay alive until each residual add.

**Compiler checkpoint:** in the optimized HLO (`.lower(...).compile().as_text()`), find where the residual add and LayerNorm ended up. Were they fused with neighbors?

## M6 — Full model

**Build:** embedding lookup (`jnp.take`, a gather), a learned position table, `L` blocks, final LayerNorm, output projection to logits `[B, S, V]`, and a parameter-count function.

**Try two ways to stack layers:** a Python `for` loop, and `jax.lax.scan` over stacked layer weights. Compare the size of the resulting IR.

**Then compare** your code against the Annotated Transformer's attention, multi-head, mask, LayerNorm, and residual sections, and nanoGPT's `model.py`. Write down every difference and decide which version is right.

**Done when:** `logits` has shape `[2, 8, 100]`, the causality test passes end to end, and the parameter count matches your hand count.

## M7 — Reading the compiler's output

**Do:** jit the full model and dump all three levels into `ir/`:
1. jaxpr (`jax.make_jaxpr`)
2. StableHLO (`lower(...).as_text()`)
3. Optimized HLO (`lower(...).compile().as_text()`), plus the full XLA dump via the `XLA_FLAGS=--xla_dump_to=ir/xla` environment variable

**Answer in `notes/m7.md`:**
- Which operations got fused into which kernels?
- How was softmax lowered: how many passes over the data?
- Change `S` from 8 to 16 and call again. What happens (retracing, recompiling), and what does that imply about dynamic shapes?

## M8 — Cost accounting

**Do:** for every operation in one block, compute FLOPs and bytes moved by hand. Compare your totals with `compiled.cost_analysis()`.

**Understand:**
- Arithmetic intensity (FLOPs per byte) for each operation; which ones are compute-bound vs. memory-bound
- Your forward FLOPs vs. the rule of thumb of about 2 × parameters per token
- Why the attention score computation's cost grows with S² while the matmuls with weights grow with S

This milestone is the bridge into the scaling book (Chapters 1 and 4).

## M9 — KV cache: prefill vs. decode

**Build:** a `prefill` function that processes the prompt and fills a key/value cache, and a `decode_step` that processes one new token using the cache.

**Constraint to work around:** jit needs static shapes, so preallocate the cache at maximum length and write into it (`.at[].set` or `jax.lax.dynamic_update_slice`).

**Test:** greedy-generate a few tokens with and without the cache. The logits must match.

**Compiler checkpoint:** compare `decode_step`'s HLO with the prefill's. Decode should be dominated by matrix-vector products. Compute its arithmetic intensity, and explain why decode is memory-bound.

## M10 (optional) — Into IREE

Take your model's StableHLO from M7 and compile it with `iree-compile` for the CPU backend. Inspect the dispatch regions IREE forms, and compare them with XLA's fusions. Optionally, write the same model in plain PyTorch, export it through torch-mlir, and compare the IR from the two frontends.

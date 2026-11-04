# Notation

The JAX model (`tinytr/`) and the PyTorch port (`gpt_torch/`) use different names for the
same sizes. This table maps them.

## Model sizes

| JAX (`tinytr/model.py` CONFIG) | Symbol | JAX value | Meaning | PyTorch (`gpt_torch/`) |
|---|---|---|---|---|
| `vocab` | $V$ | 100 | number of distinct tokens: rows of the embedding table, width of the logits | `vocab_size` (65 for Tiny Shakespeare) |
| `max_len` | $S_{\max}$ | 16 | context length: rows of the position table, longest input allowed | `block_size` |
| `d_model` | $D$ | 64 | width of every token vector in the residual stream | `n_embd` |
| `n_heads` | $H$ | 4 | number of attention heads | `n_head` |
| `d_ff` | $F$ | 256 | FFN hidden width ($4D$) | `4 * n_embd` |
| `n_layers` | $L$ | 2 | number of stacked blocks | `n_layer` |

## Derived and per-call sizes

| JAX | Symbol | Meaning | PyTorch |
|---|---|---|---|
| `d_k` | $d_k = D / H$ | per-head width of queries, keys, and values | `head_size` (`hs`) |
| `S` | $S \le S_{\max}$ | length of the current input, from `ids.shape[0]` | `T` |
| — | $B$ | batch size: sequences processed together (the JAX model is unbatched) | `B` / `batch_size` |

## Shapes

| Tensor | JAX | PyTorch |
|---|---|---|
| token ids | `[S]` | `[B, T]` |
| residual stream | `[S, D]` | `[B, T, n_embd]` |
| per-head Q / K / V | `[S, H, d_k]` | `[B, H, T, head_size]` |
| attention weights | `[H, S, S]` (inside `vmap`) | `[B, H, T, T]` |
| logits | `[S, V]` | `[B, T, vocab_size]` |

## Weight layout

- JAX (`tinytr/`): weights stored `[in, out]`; `linear(x, w, b) = x @ w + b`.
- PyTorch (`nn.Linear(in, out)`): weight stored `[out, in]`; computes `x @ W.T + b`.
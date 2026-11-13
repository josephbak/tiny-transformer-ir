"""M8: which code structure causes XLA's extra copies in decode?"""
import jax
import jax.numpy as jnp
import time


from tinytr.model import CONFIG, init_model_params, embed, output_head
from tinytr.kv_cache import init_cache, prefill, decode_step, cached_block
from tinytr.ops import softmax, layer_norm
from tinytr.block import ffn
from m8_cost import xla_cost, entry_lines, hlo_breakdown, INSTR      # script dir is on sys.path

HAND_DECODE_BYTES = 712_340


def attn_into_stacked(x_new, attn, k_all, v_all, i, pos, n_heads):
    """cached_attention, but writes straight into the stacked [L, S_max, H, d_k] cache."""
    T, D = x_new.shape
    d_k = D // n_heads
    q = (x_new @ attn["wq"]).reshape(T, n_heads, d_k)
    k = (x_new @ attn["wk"]).reshape(1, T, n_heads, d_k)
    v = (x_new @ attn["wv"]).reshape(1, T, n_heads, d_k)
    k_all = jax.lax.dynamic_update_slice(k_all, k, (i, pos, 0, 0))
    v_all = jax.lax.dynamic_update_slice(v_all, v, (i, pos, 0, 0))
    k_c, v_c = k_all[i], v_all[i]                                     # [S_max, H, d_k], read only
    scores = jnp.einsum("thd,shd->hts", q, k_c) / jnp.sqrt(d_k)
    mask = jnp.arange(k_c.shape[0])[None, :] <= (pos + jnp.arange(T))[:, None]
    w = softmax(jnp.where(mask, scores, -jnp.inf), axis=-1)
    out = jnp.einsum("hts,shd->thd", w, v_c).reshape(T, D) @ attn["wo"]
    return out, k_all, v_all


def block_into_stacked(x, p, k_all, v_all, i, pos, n_heads):
    a = layer_norm(x, p["ln1"]["gamma"], p["ln1"]["beta"])
    out, k_all, v_all = attn_into_stacked(a, p["attn"], k_all, v_all, i, pos, n_heads)
    h = x + out
    f = layer_norm(h, p["ln2"]["gamma"], p["ln2"]["beta"])
    return h + ffn(f, p["ffn"]), k_all, v_all


def make_decode(stacked_params, direct_cache, cfg=CONFIG):
    L, H = cfg["n_layers"], cfg["n_heads"]

    def step(top, blocks, token, cache, pos):
        layers = [jax.tree.map(lambda a: a[i], blocks) for i in range(L)] if stacked_params else blocks
        x = embed(top, token, pos + jnp.arange(1))
        k_all, v_all = cache["k"], cache["v"]
        for i, p in enumerate(layers):
            if direct_cache:
                x, k_all, v_all = block_into_stacked(x, p, k_all, v_all, i, pos, H)
            else:
                x, k_i, v_i = cached_block(x, p, k_all[i], v_all[i], pos, H)
                k_all, v_all = k_all.at[i].set(k_i), v_all.at[i].set(v_i)
        return output_head(top, x), {"k": k_all, "v": v_all}

    return jax.jit(step, donate_argnums=(3,))


def count_ops(compiled):
    ops = [INSTR.match(l.split(", metadata=")[0]) for l in entry_lines(compiled.as_text())]
    ops = [m for m in ops if m]
    n_copy = sum(m.group(3) == "copy" for m in ops)
    n_wslice = sum(m.group(3) == "fusion" and m.group(1).startswith("slice") for m in ops)
    return n_copy, n_wslice


def bench(fn, top, blocks, token, cache, pos, n=1000, warmup=100):
    """Median / p10 / p90 wall time per call, µs. Feeds the output cache back in (donation)."""
    c = jax.tree.map(jnp.copy, cache)
    for _ in range(warmup):                         # compile + cold caches + clock ramp-up
        _, c = fn(top, blocks, token, c, pos)
    jax.block_until_ready(c)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        logits, c = fn(top, blocks, token, c, pos)
        jax.block_until_ready((logits, c))          # wait for the work, not just the dispatch
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return [ts[int(q * n)] * 1e6 for q in (0.5, 0.1, 0.9)]


def overhead_floor(n=1000, warmup=100):
    """A jitted call that does almost nothing: the per-call cost floor, µs."""
    f = jax.jit(lambda x: x + 1)
    x = jnp.zeros((1,))
    for _ in range(warmup):
        x = f(x)
    x.block_until_ready()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        x = f(x)
        x.block_until_ready()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[n // 2] * 1e6


if __name__ == "__main__":
    params = init_model_params(jax.random.key(0), CONFIG)
    top = {k: v for k, v in params.items() if k != "blocks"}
    stacked = params["blocks"]
    unstacked = [jax.tree.map(lambda a: a[i], stacked) for i in range(CONFIG["n_layers"])]

    _, cache = prefill(params, jnp.array([17, 3, 42, 8, 91]), init_cache(CONFIG), CONFIG)
    token, pos = jnp.array([7]), jnp.int32(5)
    ref_logits, _ = decode_step(params, token, cache, pos, CONFIG)       # undonated reference

    variants = {
        "A stacked + .at[i].set": (True, False),
        "B unstacked + .at[i].set": (False, False),
        "C stacked + direct": (True, True),
        "D unstacked + direct": (False, True),
    }
    print(f"{'variant':<28}{'FLOPs':>10}{'bytes':>12}{'vs hand':>9}{'copy':>6}{'w-slice':>9}  match")

    fns, costs = {}, {}
    for name, (st, dc) in variants.items():
        fn = make_decode(st, dc)
        blocks = stacked if st else unstacked
        compiled = fn.lower(top, blocks, token, cache, pos).compile()
        f, b, _ = xla_cost(compiled)
        n_copy, n_wslice = count_ops(compiled)
        logits, _ = fn(top, blocks, token, jax.tree.map(jnp.copy, cache), pos)   # copy: donation eats it
        ok = bool(jnp.allclose(logits, ref_logits, rtol=1e-5, atol=1e-5))
        print(f"{name:<28}{int(f):>10,}{int(b):>12,}{b / HAND_DECODE_BYTES:>9.2f}{n_copy:>6}{n_wslice:>9}  {ok}")

        # if name.startswith("D"):
            # hlo_breakdown("D decode HLO", compiled, top=20)
        fns[name], costs[name] = fn, (f, b)

    print(f"\noverhead floor (empty jitted call): {overhead_floor():.1f} µs")
    print(f"{'variant':<28}{'median µs':>10}{'p10':>8}{'p90':>8}{'GFLOP/s':>9}{'GB/s':>7}")
    for name, (st, _) in variants.items():
        med, p10, p90 = bench(fns[name], top, stacked if st else unstacked, token, cache, pos)
        f, b = costs[name]
        print(f"{name:<28}{med:>10.1f}{p10:>8.1f}{p90:>8.1f}{f / med / 1e3:>9.2f}{b / med / 1e3:>7.2f}")
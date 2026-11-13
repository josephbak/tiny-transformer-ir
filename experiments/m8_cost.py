import functools
import jax
import jax.numpy as jnp
import re

from tinytr.model import CONFIG, init_model_params
from tinytr.kv_cache import init_cache, prefill, _decode_jit
from collections import defaultdict

BYTES = 4  # fp32

ELEM_BYTES = {"f32": 4, "s32": 4, "u32": 4, "f16": 2, "bf16": 2, "pred": 1, "s8": 1, "u8": 1, "s64": 8}
SHAPE = re.compile(r"\b(f32|s32|u32|f16|bf16|pred|s8|u8|s64)\[([\d,]*)\]")
INSTR = re.compile(r"^\s*(?:ROOT\s+)?%?([\w.\-]+)\s*=\s*(.+?)\s+([a-z][\w\-]*)\((.*)$")
SKIP = {"parameter", "constant", "tuple", "get-tuple-element", "bitcast"}
OPERAND = re.compile(r"%([\w.\-]+)")


def matmul(m, k, n, batch=1):
    return 2 * batch * m * k * n, BYTES * batch * (m*k + k*n + m*n)


def elementwise(N, c, n_in, n_out, extra_elems=0):
    return c * N, BYTES * (N * (n_in + n_out) + extra_elems)


def block_cost(T, cfg):
    """One cached_block call on T new rows → list of (name, flops, bytes)."""
    D, H, F, S_max = cfg["d_model"], cfg["n_heads"], cfg["d_ff"], cfg["max_len"]
    d_k = D // H

    f, b = matmul(T, D, F)
    w1 = ("W1+b1", f + T*F, b + BYTES*F)          # + bias add, + bias read
    f, b = matmul(T, F, D)
    w2 = ("W2+b2", f + T*D, b + BYTES*D)

    return [
        ("LN1",      *elementwise(T*D, 8, 1, 1, extra_elems=2*D)),
        ("QKV",      *[3 * x for x in matmul(T, D, D)]),
        ("cache_wr", 0, 2 * BYTES * (T*D + T*D)),          # k and v: read new rows, write into slice
        ("scores",   *matmul(T, d_k, S_max, batch=H)),      # per head [T,16] @ [16,32]
        ("softmax",  *elementwise(H*T*S_max, 7, 1, 1)),     # scale, mask, max, sub, exp, sum, div
        ("mix",      *matmul(T, S_max, d_k, batch=H)),      # per head [T,32] @ [32,16]
        ("W_O",      *matmul(T, D, D)),
        ("res1",     *elementwise(T*D, 1, 2, 1)),
        ("LN2",      *elementwise(T*D, 8, 1, 1, extra_elems=2*D)),
        w1,
        ("ReLU",     *elementwise(T*F, 1, 1, 1)),
        w2,
        ("res2",     *elementwise(T*D, 1, 2, 1)),
    ]


def print_table(rows):
    print(f"{'op':<10}{'FLOPs':>10}{'bytes':>10}{'AI':>7}")
    total_f = total_b = 0
    for name, f, b in rows:
        ai = f"{f / b:7.2f}" if f else f"{'—':>7}"
        print(f"{name:<10}{f:>10,}{b:>10,}{ai}")
        total_f += f
        total_b += b
    print("-" * 37)
    print(f"{'total':<10}{total_f:>10,}{total_b:>10,}{total_f / total_b:7.2f}")


def total(rows):
    return sum(r[1] for r in rows), sum(r[2] for r in rows)


def model_cost(T, cfg, head_rows=1):
    """One prefill/decode call on T new rows → list of (name, flops, bytes)."""
    D, V, L = cfg["d_model"], cfg["vocab"], cfg["n_layers"]
    f, b = total(block_cost(T, cfg))
    return [
        ("embed",          T*D, BYTES * (3*T*D + T)),      # read 2 rows/token, write 1, + token ids
        (f"blocks x{L}",   L*f, L*b),
        ("LN_f",           *elementwise(head_rows*D, 8, 1, 1, extra_elems=2*D)),
        ("head",           *matmul(head_rows, D, V)),
    ]


def xla_cost(compiled):
    ca = compiled.cost_analysis()
    ca = ca[0] if isinstance(ca, (list, tuple)) else ca     # older JAX returns a list
    return ca.get("flops", 0), ca.get("bytes accessed", 0), ca.get("transcendentals", 0)


def compile_calls(cfg):
    params = init_model_params(jax.random.key(0), cfg)
    prompt = jnp.array([17, 3, 42, 8, 91])                  # P = 5
    cache0 = init_cache(cfg)
    prefill_c = jax.jit(functools.partial(prefill, cfg=cfg)).lower(params, prompt, cache0).compile()
    _, cache = prefill(params, prompt, cache0, cfg)
    decode_c = _decode_jit.lower(params, jnp.array([7]), cache, jnp.int32(5)).compile()
    return prefill_c, decode_c


def compare(label, T, cfg, compiled):
    hf, hb = total(model_cost(T, cfg))
    h_exp = cfg["n_layers"] * cfg["n_heads"] * T * cfg["max_len"]   # one exp per score
    xf, xb, xt = xla_cost(compiled)
    print(f"\n{label}")
    print(f"{'':<16}{'hand':>12}{'XLA':>12}{'XLA/hand':>10}")
    print(f"{'FLOPs':<16}{hf:>12,}{int(xf):>12,}{xf / hf:>10.2f}")
    print(f"{'bytes':<16}{hb:>12,}{int(xb):>12,}{xb / hb:>10.2f}")
    print(f"{'exp (transc.)':<16}{h_exp:>12,}{int(xt):>12,}{xt / h_exp:>10.2f}")


def shape_bytes(text):
    total = 0
    for dt, dims in SHAPE.findall(text):
        n = 1
        for d in filter(None, dims.split(",")):
            n *= int(d)
        total += n * ELEM_BYTES[dt]
    return total


def entry_lines(hlo):
    lines = hlo.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("ENTRY"))
    for l in lines[start + 1:]:
        if l.startswith("}"):
            break
        yield l


def hlo_breakdown(label, compiled, top=15):
    parsed, size = [], {}
    for line in entry_lines(compiled.as_text()):
        m = INSTR.match(line.split(", metadata=")[0])
        if m:
            name, out, op, rest = m.groups()
            size[name] = shape_bytes(out)
            parsed.append((name, out, op, rest))

    rows = []
    for name, out, op, rest in parsed:
        if op in SKIP:
            continue
        args = rest.split("), ")[0]                               # operand list, before attributes
        in_b = sum(size.get(a, 0) for a in OPERAND.findall(args)) # calls=%fused_computation → not in size → 0
        rows.append((size[name] + in_b, op, name, out[:36], args[:90]))

    by_op = defaultdict(int)
    for b, op, *_ in rows:
        by_op[op] += b
    print(f"\n{label}: {sum(r[0] for r in rows):,} B over {len(rows)} instructions")
    for op, b in sorted(by_op.items(), key=lambda kv: -kv[1]):
        print(f"  {op:<22}{b:>12,}")
    print(f"  top {top}:")
    for b, op, name, out, args in sorted(rows, reverse=True)[:top]:
        print(f"  {b:>10,}  {op:<20}{name:<40}{out}  <- {args}")
        

if __name__ == "__main__":
    from tinytr.model import CONFIG
    # for label, T in (("prefill (T = 5)", 5), ("decode (T = 1)", 1)):
    #     print(f"\n{label}")
    #     print_table(block_cost(T, CONFIG))
    #     print_table(model_cost(T, CONFIG))

    prefill_c, decode_c = compile_calls(CONFIG)
    compare("prefill vs XLA (T = 5)", 5, CONFIG, prefill_c)
    compare("decode vs XLA (T = 1)", 1, CONFIG, decode_c)

    hlo_breakdown("prefill HLO", prefill_c)
    hlo_breakdown("decode HLO", decode_c)
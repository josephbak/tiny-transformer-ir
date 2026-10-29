import jax
import numpy as np

from tinytr.attention import attention, init_mha_weights, multi_head_attention

S, D, H = 8, 64, 4


def main():
    kx, kw = jax.random.split(jax.random.key(0))
    x = jax.random.normal(kx, (S, D))
    wq, wk, wv, wo = init_mha_weights(kw, D)

    # 1. shape check: [S, D] in, [S, D] out
    out = multi_head_attention(x, wq, wk, wv, wo, n_heads=H, causal=True)
    print("output shape:", out.shape)                      # expect (8, 64)

    # 2. with one head, multi-head attention is single-head attention followed by W_O
    mha_1 = multi_head_attention(x, wq, wk, wv, wo, n_heads=1, causal=True)
    single = attention(x, wq, wk, wv, causal=True) @ wo
    np.testing.assert_allclose(mha_1, single, rtol=1e-5, atol=1e-6)
    print("n_heads=1 matches single-head @ W_O: OK")


if __name__ == "__main__":
    main()
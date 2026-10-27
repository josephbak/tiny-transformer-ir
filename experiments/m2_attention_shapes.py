import jax
import jax.numpy as jnp
from tinytr.attention import attention


def main():
    key = jax.random.key(0)

    kx, kq, kk, kv = jax.random.split(key, 4)   # split returns 4 new keys; unpack them
    x = jax.random.normal(kx, (8, 64))          # one key per random array

    D = 64
    wq = jax.random.normal(kq, (64, 16)) / jnp.sqrt(D)  # scale weights by 1/sqrt(D)
    wk = jax.random.normal(kk, (64, 16)) / jnp.sqrt(D)
    wv = jax.random.normal(kv, (64, 16)) / jnp.sqrt(D)

    attn_out = attention(x, wq, wk, wv)

    print("x shape:", x.shape)
    print("wq shape:", wq.shape)
    print("wk shape:", wk.shape)
    print("wv shape:", wv.shape)
    print("attn_out shape:", attn_out.shape)

if __name__ == "__main__":
    main()
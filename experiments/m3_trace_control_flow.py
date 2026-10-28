import jax
import jax.numpy as jnp


def f(x):
    if x.sum() > 0:           # Python if on a VALUE
        return x * 2
    else:
        return x * -1


def f_cond(x):                # both branches put INTO the IR; chosen at run time
    return jax.lax.cond(x.sum() > 0, lambda x: x * 2, lambda x: x * -1, x)


def f_where(x):               # both branches computed; selected elementwise
    return jnp.where(x.sum() > 0, x * 2, x * -1)


def main():
    pos, neg = jnp.array([1., 2.]), jnp.array([-1., -2.])

    print("eager (no jit):", f(pos), f(neg))            # works: real values

    try:
        jax.jit(f)(pos)
    except Exception as e:
        print("jit(f):", type(e).__name__)               # loud error, nothing compiled

    print("jit(f_cond): ", jax.jit(f_cond)(pos), jax.jit(f_cond)(neg))
    print("jit(f_where):", jax.jit(f_where)(pos), jax.jit(f_where)(neg))

    print(jax.make_jaxpr(f_cond)(pos))                   # look for `cond`, with two branches inside


if __name__ == "__main__":
    main()
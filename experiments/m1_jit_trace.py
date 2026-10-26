import jax
import jax.numpy as jnp
from tinytr.ops import linear, softmax


def wrapper(x):
    print("input:  ", x)
    out = softmax(x)
    print("softmax:", out)
    # if x[0] > 0:
    #     print("hello")
    return out

jitted = jax.jit(wrapper)   # nothing runs yet; this just creates the new function

def main():
    print("returned:", jitted(jnp.array([1., 2.])))
    print("returned:", jitted(jnp.array([5., 7.])))
    print("returned:", jitted(jnp.array([1., 2., 3.])))

if __name__ == "__main__":
    main()
import jax
import jax.numpy as jnp
from tinytr.ops import softmax, softmax_naive

def main():
    # for each input: print softmax, softmax_naive, jax.nn.softmax
    input1 = jnp.array([1000., 1001.])
    input2 = jnp.array([-1000., -999.])
    input3 = jnp.array([1., 2.])
    
    for x in (input1, input2, input3):
        print("input:        ", x)
        print("softmax:      ", softmax(x))
        print("softmax_naive:", softmax_naive(x))
        print("jax.nn:       ", jax.nn.softmax(x))
        print()

if __name__ == "__main__":
    main()
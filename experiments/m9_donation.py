import jax, jax.numpy as jnp
from tinytr.model import CONFIG, init_model_params
from tinytr.kv_cache import init_cache, prefill, _decode_jit

params = init_model_params(jax.random.key(0), CONFIG)
prompt = jnp.array([17, 3, 42, 8, 91])
_, cache = prefill(params, prompt, init_cache(CONFIG), CONFIG)
token = jnp.array([7])

lowered = _decode_jit.lower(params, token, cache, jnp.int32(5))   # trace → StableHLO; nothing runs
compiled = lowered.compile()                                      # XLA compile; nothing runs

stablehlo = lowered.as_text()
hlo = compiled.as_text()
print("StableHLO asks for donation:", "tf.aliasing_output" in stablehlo or "jax.buffer_donor" in stablehlo)
print("XLA granted it:             ", "input_output_alias" in hlo)
print(hlo.splitlines()[0][:600])   # the HloModule header line, where input_output_alias appears
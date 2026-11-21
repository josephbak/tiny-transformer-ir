"""M10: does IREE overwrite a non-donated input? A minimal repro with dynamic_update_slice.

r = dynamic_update_slice(x, y, 0) with x = 32 zeros, y = [1, 1, 1, 1], and NO donation.
Expected: r = [1, 1, 1, 1, 0, ...] and x unchanged (all zeros).

Run from the repo root:  uv run python -m experiments.m10_repro_input_mutation
"""

import iree.compiler as ireec
import iree.runtime as ireert
import numpy as np

PROGRAM = """
func.func @main(%x: tensor<32xf32>, %y: tensor<4xf32>) -> tensor<32xf32> {
  %c0 = stablehlo.constant dense<0> : tensor<i32>
  %r = stablehlo.dynamic_update_slice %x, %y, %c0 : (tensor<32xf32>, tensor<4xf32>, tensor<i32>) -> tensor<32xf32>
  return %r : tensor<32xf32>
}
"""


def main():
    vmfb = ireec.compile_str(PROGRAM, input_type="stablehlo", extra_args=[
        "--iree-hal-target-device=local",
        "--iree-hal-local-target-device-backends=llvm-cpu",
        "--iree-llvmcpu-target-cpu=host",
    ])
    config = ireert.Config("local-sync")
    ctx = ireert.SystemContext(config=config)
    module = ireert.VmModule.copy_buffer(config.vm_instance, vmfb)
    ctx.add_vm_module(module)
    f = ctx.modules[module.name]["main"]

    # Device-resident input that the caller keeps (like a real application would).
    x = ireert.asdevicearray(config.device, np.zeros(32, np.float32))
    y = ireert.asdevicearray(config.device, np.ones(4, np.float32))

    print("x before:", x.to_host()[:8], "...")
    r = f(x, y)
    print("r       :", r.to_host()[:8], "...")
    print("x after :", x.to_host()[:8], "...")
    changed = bool(np.any(x.to_host() != 0))
    print("\ninput x was modified:", changed, "(expected False: x is not donated)")


if __name__ == "__main__":
    main()
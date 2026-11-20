#!/usr/bin/env bash
# M10 step 7: time IREE with iree-benchmark-module, on both CPU drivers.
#   A. decode_step                     (needs ir/m10/decode_step.vmfb and decode_step_ref.npz)
#   B. baselines: an empty call, and a call with one tiny dispatch (compiled here)
# Prints one line per measurement: median and stddev of the wall time per call, in µs.
# Run from the repo root:  bash scripts/m10_bench_iree.sh
set -euo pipefail

FLAGS=(--iree-hal-target-device=local --iree-hal-local-target-device-backends=llvm-cpu
       --iree-llvmcpu-target-cpu=host)
BENCH=(--benchmark_repetitions=5 --benchmark_min_time=0.5s
       --benchmark_report_aggregates_only=true --benchmark_format=csv)

# The 21 decode inputs as .npy files (an .npz is a zip of .npy files), in flat order.
mkdir -p ir/m10/inputs
unzip -oq ir/m10/decode_step_ref.npz 'in_[0-9]*.npy' -d ir/m10/inputs
DECODE_INPUTS=()
for f in ir/m10/inputs/in_[0-9]*.npy; do DECODE_INPUTS+=("--input=@$f"); done

# B. Two baseline programs: no dispatch at all, and exactly one tiny dispatch.
mkdir -p ir/m10/bench
cat > ir/m10/bench/empty.mlir <<'MLIR'
func.func @main(%x: tensor<i32>) -> tensor<i32> {
  return %x : tensor<i32>
}
MLIR
cat > ir/m10/bench/one_dispatch.mlir <<'MLIR'
func.func @main(%x: tensor<64xf32>) -> tensor<64xf32> {
  %y = stablehlo.add %x, %x : tensor<64xf32>
  return %y : tensor<64xf32>
}
MLIR
for m in empty one_dispatch; do
  uv run iree-compile ir/m10/bench/$m.mlir "${FLAGS[@]}" -o ir/m10/bench/$m.vmfb
done

# bench <label> <device> <module> <inputs...>: run, keep the median and stddev rows, print in µs.
bench() {
  local label=$1 dev=$2 module=$3; shift 3
  uv run iree-benchmark-module --module="$module" --function=main --device="$dev" "$@" \
      "${BENCH[@]}" 2>/dev/null |
    awk -F, -v label="$label" -v dev="$dev" '
      /_median"/ { med = $3 }
      /_stddev"/ { sd = $3 }
      END { printf "%-14s %-11s median %8.1f µs   stddev %6.1f µs\n", label, dev, med * 1000, sd * 1000 }'
}

for dev in local-sync local-task; do
  bench decode_step  "$dev" ir/m10/decode_step.vmfb       "${DECODE_INPUTS[@]}"
  bench empty        "$dev" ir/m10/bench/empty.vmfb        --input=i32=5
  bench one_dispatch "$dev" ir/m10/bench/one_dispatch.vmfb --input=64xf32=1
done
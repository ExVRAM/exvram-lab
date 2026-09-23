# Benchmark harness

The first harness is intentionally a planner plus a CPU-only synthetic workload. It does not
pretend to be model inference. Real measurements must record the external runtime, model hash,
GPU telemetry, context, batch size, warm-up policy, and decoder settings in the result schema.

Use `exvram run-synthetic-microbenchmark` to validate the result-writing path without CUDA.


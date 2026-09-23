# P1 real GPU results

Captured on 2026-09-23 on the RTX 5060 described in [GPU_ENVIRONMENT.md](GPU_ENVIRONMENT.md).
These are real CUDA layer measurements, not full-model inference and not synthetic timing.

## Scope and status

The primary raw JSONL contains 51 records:

- 12 Torch FP16 records;
- 12 Torch BF16 records;
- 12 GemLite 2-bit records;
- 12 GemLite 4-bit records;
- 3 ExLlamaV3 EXL3 K=4 records for the attention Q projection at M=1/8/32.

The GemLite/Torch matrix is in
[`experiments/results/p1_real_gpu_layer_benchmark.jsonl`](../experiments/results/p1_real_gpu_layer_benchmark.jsonl).
The corrected ExLlamaV3 run is in
[`experiments/results/exl3_real_gpu_attention_q_m1_v3.jsonl`](../experiments/results/exl3_real_gpu_attention_q_m1_v3.jsonl).
Every record has `synthetic: false` and `measurement_status: measured`.

## Representative attention-Q comparison

Shape is 4096×4096, batch 1, warmup 10, repeats 50 for the primary matrix. Effective bandwidth is
computed from the stored weight bytes divided by CUDA-event median time; it is not a profiler DRAM
counter. Torch rows are dense baselines, not low-bit representations.

| Backend | Median ms | P95 ms | Effective GiB/s | Physical bpw | Correctness reference |
|---|---:|---:|---:|---:|---|
| Torch FP16 | 0.116192 | 0.128960 | 268.951 | 16.0 nominal | FP32 CUDA matmul |
| Torch BF16 | 0.115856 | 0.121920 | 269.731 | 16.0 nominal | FP32 CUDA matmul |
| GemLite W2 | 0.261744 | 0.282016 | 14.924 | 2.500029 | original FP32 and dequantized fixture |
| GemLite W4 | 0.263712 | 0.286912 | 29.625 | 4.500029 | original FP32 and dequantized fixture |
| ExLlamaV3 EXL3 K4 | 0.065456* | 0.075008* | 119.355* | 4.007813 | original FP32 fixture |

`*` ExLlama values are from the corrected 20-repeat run in
`exl3_real_gpu_attention_q_m1_v3.jsonl`; the earlier 30-repeat run is retained only as an
independent cross-check. Use v3 for current ExLlama workspace and M-scaling evidence.

For GemLite W2 on this fixture, the 2-bit packed tensor is 4,194,304 bytes, scales are 524,288
bytes, zero points are 524,288 bytes, and metadata is 56 bytes: 5,242,940 persistent bytes in
total, or 2.500029 physical bpw. W4 has the same group metadata and 4.500029 physical bpw. The
group size is 64 in this fixture, so these overheads must not be transplanted into the full-model
planner without a checkpoint manifest.

## M scaling

Attention-Q results from the backend-specific raw records:

| Backend | M=1 median ms | M=8 median ms | M=32 median ms |
|---|---:|---:|---:|
| GemLite W2 | 0.261744 | 0.229824 | 0.335808 |
| GemLite W4 | 0.263712 | 0.229824 | 0.240640 |
| ExLlamaV3 EXL3 K4 | 0.065456 | 0.071088 | 0.100816 |

The values demonstrate layer-level behavior only. They do not imply decode tok/s because a token
step includes attention, MLP, norms, sampling, cache handling, and runtime scheduling.

## Quality and memory boundary

The random matrices are backend fixtures, not a model checkpoint. GemLite's error against the
original FP32 matrix is quantization error; the additional kernel-vs-dequantized error is the
kernel correctness check. ExLlama's calibration-free K=4 fixture reports an EXL3 proxy error but
does not establish model perplexity or task quality. No KV-cache allocation, 8k-context run, or
full-model peak has been measured.

The current conclusions are therefore:

- Real existing low-bit paths are executable on this GPU in isolated environments.
- GemLite W2 has the lowest measured physical bpw in the exercised fixture.
- ExLlamaV3 has the lowest attention-Q layer latency among the measured rows, but its K=4 result is
  not an apples-to-apples quality comparison with GemLite W2.
- The `>=25` decode tok/s target is **unmeasured**.
- Custom-kernel work is **NO-GO**: two existing paths run, but no same-format, checkpoint-backed
  bottleneck has been proven against two approaches.

## Next gate

Select and hash one concrete dense ~27B checkpoint, run its native ExLlama/GGUF/BitNet format path,
capture KV and residency at 8k, and apply the quality gate before making any full-model fit or
throughput claim.

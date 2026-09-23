# Memory map: dense ~27B / 8k / batch 1

The planner's current legacy estimate is **8.319329 GiB**. It is a model-based estimate, not a
driver allocation measurement. The default placeholder deliberately has no checkpoint manifest
for embeddings, lm_head, small tensors, or codebooks, so those rows are explicit zeroes rather
than invented sizes.

| Component | Theoretical / modeled GiB | Precision | Residency | Source | Movable/compressible |
|---|---:|---|---|---|---|
| Packed transformer weights | 6.286427 | 2-bit | VRAM | 27B × 2-bit packing | yes |
| Input embeddings | 0.000000 | 2-bit placeholder | VRAM | checkpoint manifest missing | conditional |
| lm_head | 0.000000 | 6-bit placeholder | VRAM | checkpoint manifest missing | conditional |
| Norms, biases, small FP tensors | 0.000000 | FP16/FP32 placeholder | VRAM | checkpoint manifest missing | conditional |
| Quantization scales | 0.392902 | FP16 | VRAM | one scale per group of 128 | yes |
| Codebooks / metadata | 0.000000 | runtime-specific | VRAM | checkpoint/runtime manifest missing | conditional |
| KV cache — full attention | 1.000000 | FP16 | VRAM | analytic KV formula | yes |
| CUDA context | 0.120000 | runtime allocation | VRAM | 8% reserve split | no |
| Allocator reserve | 0.200000 | runtime reserve | VRAM | 8% reserve split | conditional |
| Runtime scratch/workspace | 0.100000 | runtime reserve | VRAM | 8% reserve split | conditional |
| Attention workspace | 0.080000 | runtime reserve | VRAM | 8% reserve split | conditional |
| Kernel-specific workspace | 0.080000 | runtime reserve | VRAM | 8% reserve split | conditional |
| Graph capture overhead | 0.000000 | optional | VRAM | graph capture disabled in placeholder | conditional |
| Safety reserve | 0.060000 | reserve | optional | 8% reserve split | do not remove |
| **Total** | **8.319329** |  |  |  |  |

The physical device telemetry is 7.959961 GiB, while the research target is deliberately stricter:
`target_peak_vram_gib = 7.5`. The planner therefore reports both physical-budget fit and target-peak
fit. The 8.319329 GiB total is reproducible by running:

```text
exvram plan-experiment --config experiments/weights/baseline_27b_8gb.json --index 0
```

## Modeled existing-upstream options

`exvram compare-memory-configurations` ranks scenarios. The most conservative modeled combination
that crosses 7.5 GiB is:

| Scenario | Modeled total GiB | Status |
|---|---:|---|
| KV-Q4 only | 7.569329 | above target |
| KV-Q4 + scratch reuse | 7.469329 | reaches target, unmeasured |
| KV-Q3 + scratch reuse | 7.406829 | reaches target, higher quality risk |
| KV-Q2 + scratch reuse | 7.344329 | reaches target, highest quality risk |

ExLlamaV3 documents quantized KV-cache support; exact cache formats, scratch reuse, and quality
must be verified on the concrete checkpoint. These rows are planning scenarios, not claims that
the current machine has achieved the totals.

## Confidence and source boundary after P1 GPU runs

| Item | Status | Evidence | How it may be used |
|---|---|---|---|
| RTX 5060 / 7.959 GiB / CC 12.0 | measured | Torch and `nvidia-smi` in `docs/GPU_ENVIRONMENT.md` | hardware budget only |
| Torch FP16/BF16 layer timings | measured | 24 records in the primary P1 JSONL | layer baseline only |
| GemLite W2/W4 component bytes and physical bpw | measured | real GemLite pack/kernel fixture, group size 64 | that fixture and compatible packing only |
| ExLlamaV3 EXL3 K4 component bytes and timings | measured | official `LinearEXL3` fixture JSONL | EXL3 layer comparison only |
| EXL3/GemLite APIs and license status | upstream documented | pinned environment/package records and `THIRD_PARTY.md` | integration boundary, not performance guarantee |
| 27B packed weight arithmetic | derived | parameter count × nominal bpw | planning estimate until checkpoint manifest |
| KV-cache rows, reserve and workspace rows | modeled/assumed | analytic planner inputs | scenario comparison only |
| 27B full-model peak, quality and tok/s | unavailable | no checkpoint-backed run | must not be inferred from layer records |

The measured GemLite W2 fixture stores 4,194,304 packed-weight bytes, 524,288 scale bytes,
524,288 zero-point bytes, and 56 metadata bytes for a 4096×4096 matrix. Its 5,242,940 persistent
bytes equal 2.500029 physical bpw. EXL3 K4 stores 8,388,608 trellis bytes plus 16,384 bytes of
sign/scale metadata, or 4.007813 physical bpw. These observations correct the corresponding
layer representation only; they do not replace the full-model 8.319329 GiB planner total without
real tensor shapes, group sizes, codebooks, KV policy and runtime residency data.

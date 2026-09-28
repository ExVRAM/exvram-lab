# Benchmark harness

The first harness is intentionally a planner plus a CPU-only synthetic workload. It does not
pretend to be model inference. Real measurements must record the external runtime, model hash,
GPU telemetry, context, batch size, warm-up policy, and decoder settings in the result schema.

Use `exvram run-synthetic-microbenchmark` to validate the result-writing path without CUDA.

`exvram score-refusal` scores a local JSONL of responses. It does not load a checkpoint.
`exvram plan-heretic` prints an external Heretic command and does not run it. Neither command
is a performance benchmark.

## P5 full-model context scaling

`run_p5_context_scaling.py` is an optional llama.cpp server integration. It starts one external
`llama-server` process at a time, streams a fixed-seed completion, samples NVIDIA telemetry and
Windows CPU/RAM counters, then appends one JSONL record. It does not bundle llama.cpp or model
weights.

Example on the RTX 5060 research machine:

```powershell
.\.venv\Scripts\python.exe benchmark\run_p5_context_scaling.py `
  --server <llama-server.exe> `
  --model <Qwen3.8-27B-UD-IQ2_XXS.gguf> `
  --contexts 256,512,1024,2048,4096,8192 `
  --cache-type-k q4_0 --cache-type-v q4_0 --decode-tokens 256 `
  --output experiments\results\p5_context_scaling_q4.jsonl `
  --work-dir work\p5_context_scaling_q4 `
  --allow-removable-storage
```

The runner marks context-only rows with `quality_gate.status=NOT_RUN`. A result is not evidence
of model quality, and an 8k row must not be interpreted as a stable speed guarantee without
repeatability and a separate quality evaluation.

## P6 reproducibility

`run_p6_reproducibility.py` repeats short or tokenizer-filled prompts after warm-up and records
configured context separately from occupied context. It refuses to start a model when the
pre-start GPU memory exceeds `--max-background-vram-mib` unless `--allow-contended-gpu` is
explicitly supplied. An output lock prevents two P6 runs from sharing one JSONL file and GPU.

The default workflow is CPU-safe until the runner is invoked with external `llama-server.exe`
and a model path. Filled-context runs are intentionally separate from short-prompt controls;
their throughput must not be compared as if they had the same occupancy.

## P8 binary-representation planner

P8 is an isolated layer-level research track for frequent-group 1-bit and
partially binary representations. It uses optional external HQQ/GemLite
boundaries and records PB-LLM/BiLLM as references. No quantizer or CUDA/Triton
kernel is copied into this repository.

The CPU-safe command writes a machine-readable plan and physical-footprint
estimate for the real Qwen3.8-27B projection shapes:

```powershell
.venv\\Scripts\\exvram.exe plan-p8-binary `
  --manifest experiments\\manifests\\p8_binary_qwen38.json `
  --output experiments\\results\\p8_binary_plan.json
```

The plan is explicitly `synthetic: true` and `measurement_status: PLANNED`.
It contains no tok/s, MSE, VRAM, or quality result. GPU work is deferred until
the optional dependencies are pinned in a GPU environment and the card has a
safe idle VRAM slot. See [BINARY_1BIT_RESEARCH.md](../docs/BINARY_1BIT_RESEARCH.md)
and [HQQ_GEMLITE_RESULTS.md](../docs/HQQ_GEMLITE_RESULTS.md).

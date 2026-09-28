# ExVRAM Lab

**Trading GPU compute for memory to run larger LLMs on smaller GPUs.**

ExVRAM Lab is an open-source research project focused on running larger dense LLMs on memory-constrained consumer GPUs by combining ultra-low-bit representations, full-GPU residency, memory-aware runtime configuration and existing open-source inference technologies.

Current verified milestone:

- NVIDIA RTX 5060 8 GB
- Qwen3.8-27B
- 65/65 layers on CUDA
- CPU decoder weight offload: 0
- CUDA model buffer: 6521 MiB
- peak VRAM: 7611 MiB
- measured short-context decode: 31.39 tok/s
- occupied 8192-token prompt, q4_0 KV: median decode 30.10 tok/s over 5 runs, peak VRAM 7767 MiB
- quality gate and the uncensored checkpoint are not done

The 31.39 tok/s figure is the short-context FULL_GPU decode. The 30.10 tok/s figure is the repeated occupied-8192 median. Neither is a quality result.

## Research goals

- Dense ~27B on 8 GB consumer GPU
- Full GPU residency
- 8k context
- >=25 tok/s target
- minimal measurable quality loss
- minimal-refusal / steerable model variant
- reuse mature OSS before custom kernels

## Core principle

Reuse mature open-source implementations first.

Current upstream ecosystem includes:

- ExLlamaV3
- llama.cpp
- GemLite
- CUTLASS
- BitNet
- Marlin
- FlashInfer
- Ollama

ExVRAM should only introduce custom low-level code after a measurable gap is demonstrated.

Current research target:

- NVIDIA RTX 5060 8 GB
- SM120 / consumer Blackwell
- Dense ~27B class models
- 8k context
- Batch-1 interactive inference
- Target: ≥25 tok/s where achievable
- Quality loss must be measured, not assumed
- Steerable, with substantially reduced built-in refusal behaviour (minimal-refusal)

These are research targets. A number in this list is not a measured result unless [Current status](#current-status) says it was measured. "Minimal-refusal" means a measured drop in refusal rate at a small measured capability cost. It does not mean the checkpoint is completely uncensored.

ExVRAM aims for a local model that is large, fast, memory-efficient, steerable, and minimal-refusal. Model behaviour is a property of the weights. Any application restriction stays outside the weights, in a separate configurable policy. Runtime performance and that policy layer are not the same track.

## Why ExVRAM

The working thesis is to trade additional GPU compute for reduced VRAM traffic and residency.

Low-bit storage, on-the-fly reconstruction, and a tighter memory hierarchy can keep a larger model resident, or keep more of it on the GPU, when the card cannot hold a dense high-precision checkpoint. That trade is only useful when the extra compute still leaves interactive latency and when quality loss is measured on the same checkpoint.

Several of the techniques already exist. ExLlamaV3, GemLite, BitNet, llama.cpp, CUTLASS, and related projects already implement quantizers, kernels, and runtimes. ExVRAM Lab first integrates, compares, measures, and combines those open-source solutions. New first-party low-level code is written only after a measured gap shows that reuse is not enough.

## Principles

1. Reuse mature open source first.
2. Measure before optimizing.
3. Never confuse nominal bpw with physical bpw.
4. Never report modeled results as measured.
5. Custom kernels require a demonstrated gap.
6. Quality, VRAM, speed, and refusal rate are optimized together.
7. Do not write a first-party uncensoring method while a ready abliterated checkpoint or an existing tool such as Heretic still applies.

## Current status

Confirmed in this repository as of 2026-09-26:

- RTX 5060 8 GB detected: 8151 MiB reported by the driver, compute capability 12.0 / SM120.
- Real CUDA layer microbenchmarks are recorded. They use generated matrices, not a full checkpoint.
- GemLite W2 physical bpw measured around 2.50 (2.500029) for the tested layer representation.
- GemLite W4 physical bpw measured around 4.50 (4.500029) for the tested layer representation.
- ExLlamaV3 EXL3 K4 physical bpw measured around 4.01 (4.007813) for the tested representation.
- One llama.cpp full-model smoke at context 128, batch 1, text-only, bartowski IQ2_XXS: prefill 31.3 tok/s and decode 3.8 tok/s. Quality was not run. This is not an 8k result and does not meet the 25 tok/s decode target.
- Ollama installed-model smoke for `qwen2.5-coder:7b`: prefill 1399.5 tok/s and decode 45.8 tok/s on the detected RTX 5060. This is a 7B runtime baseline, not a 27B result; quality was not run.
- ExLlamaV3 full-model attempts on the tested EXL3 artifact failed to fit; there is no successful ExLlamaV3 full-model tok/s.
- P4 now contains one real IQ2_XXS Qwen3.8-27B `FULL_GPU_CONFIRMED` run: llama.cpp reported 65/65 layers on CUDA0, 31.39 tok/s at the runtime-clamped c256 profile, and 18.11 tok/s in a ctx512 control with 256 generated tokens. See [docs/FULL_GPU_27B.md](docs/FULL_GPU_27B.md), [docs/FULL_GPU_PROFILE.md](docs/FULL_GPU_PROFILE.md), and [experiments/results/p4_full_gpu_27b.jsonl](experiments/results/p4_full_gpu_27b.jsonl).
- P5 contains a standardized context curve. Its single c8192 point of 29.04 tok/s used a 5240-token prompt, not a full 8192-token occupancy. See [docs/P6_REPRODUCIBILITY.md](docs/P6_REPRODUCIBILITY.md).
- The uncensored 8k track has not been run. A local Coletti BF16 GGUF exists, but no matching IQ2_XXS artifact, refusal rate, or quality result is claimed. See [docs/UNCENSORED_QUANTIZATION.md](docs/UNCENSORED_QUANTIZATION.md).
- The requested c128 setting is currently clamped to actual n_ctx=256 by this llama.cpp/model combination. No 8k quality result is claimed; the quality gate remains NOT_RUN.
- Occupied 8192-token prompt on UD-IQ2_XXS, 65/65 CUDA, q4_0 KV: median decode 30.10 tok/s, five runs, CV 2.0%. Quality is still `NOT_RUN`. See [docs/P6_FINAL_RESULT.md](docs/P6_FINAL_RESULT.md).
- Minimal-refusal sources were inventoried on 2026-09-24. No uncensored checkpoint was loaded on this GPU. Refusal rate, quality delta, decode, and 8k fit for those files are not ExVRAM measurements. The decision gate is `REQUANTIZATION_REQUIRED`. See [docs/UNCENSORED_PARETO.md](docs/UNCENSORED_PARETO.md).
- P8 binary research scaffolding is present: real Qwen3.8-27B layer shapes, physical-bpw planner, HQQ adapter boundary, and PB-LLM/BiLLM reference records. P8 quality, CUDA timing, and full-model results are `NOT_RUN`; no production recipe changed. See [docs/BINARY_1BIT_RESEARCH.md](docs/BINARY_1BIT_RESEARCH.md).

Layer benchmarks are not full-model results. Nominal bits per weight are not physical bits per weight. Details are in [docs/P1_REAL_GPU_RESULTS.md](docs/P1_REAL_GPU_RESULTS.md), [docs/P2_RESULTS.md](docs/P2_RESULTS.md), and [docs/GPU_ENVIRONMENT.md](docs/GPU_ENVIRONMENT.md).

## Architecture

ExVRAM Lab sits in front of existing runtimes. It does not replace them.

```text
ExVRAM Lab
  -> adapters
  -> benchmark
  -> experiment database
  -> planner / search
  -> existing runtimes and kernels
  -> optional custom components, only after a measured gap
```

Experiment configs and a memory planner produce a plan. Adapters describe optional external runtimes without vendoring their kernels. Benchmarks write versioned JSON or JSONL. A small research database records queue state. Search policies stay scaffolding until measurements justify a choice. `custom/` stays minimal on purpose.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Upstream projects

ExVRAM does not try to rewrite these projects from scratch:

- [ExLlamaV3](https://github.com/turboderp-org/exllamav3) — MIT
- [GemLite](https://github.com/dropbox/gemlite) — Apache-2.0
- [CUTLASS](https://github.com/NVIDIA/cutlass) — BSD-3-Clause for applicable core components; some files have separate NVIDIA terms
- [BitNet](https://github.com/microsoft/BitNet) — MIT
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — MIT
- [Ollama](https://github.com/ollama/ollama) — MIT CLI/runtime boundary; already-installed models only
- [HQQ](https://github.com/dropbox/hqq) — Apache-2.0, optional P8 quantization boundary
- [PB-LLM](https://github.com/hahnyuan/PB-LLM) — MIT, P8 algorithmic reference only
- [BiLLM](https://github.com/Aaronhuang-778/BiLLM) — MIT, P8 layer-feasibility reference only

QTIP is a GPLv3 research reference only. No GPL code is copied into this Apache-2.0 repository. Upstream license checks for this publication are recorded in [THIRD_PARTY.md](THIRD_PARTY.md).

## Repository structure

- `src/exvram/` — contracts, hardware detection, memory calculator, adapter registry, CLI, results.
- `src/exvram/optimizer/` — bounded adaptive search and hardware-aware evidence-based recommendations.
- `schemas/` — versioned experiment, result, and layer-benchmark JSON schemas.
- `adapters/{exllamav3,gemlite,cutlass,bitnet,llamacpp,ollama,hqq,pbllm,billm}` — integration notes and boundaries. Kernels are not vendored.
- `benchmark/` — protocol notes and optional full-model runner entry points.
- `experiments/weights`, `experiments/kv_cache`, `experiments/residency`, `experiments/kernels` — experiment inputs and notes.
- `experiments/manifests/` — provenance for external checkpoints. Weights themselves are not in Git.
- `experiments/results/` — small raw JSONL records and the research SQLite database.
- `search/` — future allocation and runtime search policies.
- `experiments/search/` — bounded target configuration search spaces; planned curve inputs live under `experiments/residency/` and `experiments/kv_cache/`.
- `custom/` — first-party low-level boundary, intentionally minimal.
- `tests/` — CPU unit tests. GPU-only behavior skips or reports unavailable when CUDA is absent.
- `docs/` — architecture, GPU environment, measured P1/P2 results, and the research plan.

## Quick start

Python 3.10 or newer is required. The base package has no runtime dependency on Torch or a GPU. On Windows, use a virtual environment so GPU packages stay off the global interpreter.

Windows:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[dev]"
.\.venv\Scripts\exvram list-adapters
.\.venv\Scripts\exvram validate-environment
.\.venv\Scripts\exvram plan-experiment --config experiments/weights/baseline_27b_8gb.json --index 0
.\.venv\Scripts\exvram compare-memory-configurations --config experiments/weights/baseline_27b_8gb.json --index 0
.\.venv\Scripts\exvram plan-search --config experiments/search/rtx5060_27b_search.json --stage adaptive
.\.venv\Scripts\exvram recommend-config --config experiments/search/rtx5060_27b_search.json
.\.venv\Scripts\exvram plan-p8-binary --manifest experiments/manifests/p8_binary_qwen38.json --output experiments/results/p8_binary_plan.json
.\.venv\Scripts\exvram run-synthetic-microbenchmark --output results/synthetic.json
.\.venv\Scripts\python -m unittest discover -s tests -v
.\.venv\Scripts\python -m compileall -q src tests
.\.venv\Scripts\ruff check .
```

macOS and Linux:

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
exvram list-adapters
exvram validate-environment
exvram plan-experiment --config experiments/weights/baseline_27b_8gb.json --index 0
exvram compare-memory-configurations --config experiments/weights/baseline_27b_8gb.json --index 0
exvram run-synthetic-microbenchmark --output results/synthetic.json
python -m unittest discover -s tests -v
python -m compileall -q src tests
ruff check .
```

`run-synthetic-microbenchmark` writes `synthetic: true` and does not report tokens/second. It checks the workflow, not model inference.

`run-layer-benchmark` times real CUDA kernels when a CUDA build of Torch is installed. Without CUDA it must not invent tok/s. Isolated GPU environments used for the recorded layer results are `.venv-gpu` (Torch and GemLite) and `.venv-exllama` (ExLlamaV3). Both are gitignored. Recreate them only when you intend to measure; CPU tests do not need them. See [docs/GPU_ENVIRONMENT.md](docs/GPU_ENVIRONMENT.md).

Optional P2 commands, once a manifest is present:

```text
exvram validate-research-manifest --manifest experiments/manifests/qwen38_p2.json
exvram plan-p2 --manifest experiments/manifests/qwen38_p2.json --queue-output experiments/results/p2_queue.jsonl --database experiments/results/research.sqlite
```

Full-model runners under `benchmark/` call an installed ExLlamaV3 environment or an external llama.cpp binary. Model weights and those binaries stay outside Git. Follow [docs/SAFE_WORKFLOW.md](docs/SAFE_WORKFLOW.md) before any run that touches external storage.

For a local Ollama baseline, use the installed-model-only command:

```text
exvram run-ollama --model qwen2.5-coder:7b --prompt "Reply briefly: ExVRAM smoke test ready." --output experiments/results/ollama_smoke.jsonl
```

This command checks `ollama list` first and never downloads a missing model.

## Research roadmap

- P0 — infrastructure and memory planner.
- P1 — real layer-level GPU measurements. Recorded for the fixtures in this repo; not a full-model claim.
- P2 — full-model OSS shootout. A context-128 llama.cpp smoke exists; quality and a matched runtime comparison remain open.
- P3 — bottleneck-specific optimization, only after a measured full-model gap.
- P4 — optional ExVRAM runtime or orchestrator, only if existing runtimes still leave a measured gap.
- P5 — context scaling and KV precision. P6 then repeated an occupied 8192-token prompt at 30.10 tok/s median. Quality gating is still open.

The minimal-refusal track does not replace that order. It runs beside the FULL_GPU work: pick a published abliterated BF16 checkpoint, quantize it with an existing tool, then compare VRAM, tok/s, quality, and refusal rate at a matched quant. Details are in [docs/UNCENSORED_OSS_MATRIX.md](docs/UNCENSORED_OSS_MATRIX.md).

## License

Original ExVRAM Lab code is Apache License 2.0. See [LICENSE](LICENSE).

Third-party projects keep their own licenses. See [THIRD_PARTY.md](THIRD_PARTY.md).

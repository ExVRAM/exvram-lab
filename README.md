# ExVRAM Lab

**Trading GPU compute for memory to run larger LLMs on smaller GPUs.**

ExVRAM Lab is an open-source research environment for testing how low-bit representations, GPU-native reconstruction, memory hierarchy management, selective offload and existing inference runtimes can reduce VRAM requirements for local LLM inference.

Current research target:

- NVIDIA RTX 5060 8 GB
- SM120 / consumer Blackwell
- Dense ~27B class models
- 8k context
- Batch-1 interactive inference
- Target: ≥25 tok/s where achievable
- Quality loss must be measured, not assumed

These are research targets. A number in this list is not a measured result unless [Current status](#current-status) says it was measured.

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
6. Quality, VRAM and speed are optimized together.

## Current status

Confirmed in this repository as of 2026-09-23:

- RTX 5060 8 GB detected: 8151 MiB reported by the driver, compute capability 12.0 / SM120.
- Real CUDA layer microbenchmarks are recorded. They use generated matrices, not a full checkpoint.
- GemLite W2 physical bpw measured around 2.50 (2.500029) for the tested layer representation.
- GemLite W4 physical bpw measured around 4.50 (4.500029) for the tested layer representation.
- ExLlamaV3 EXL3 K4 physical bpw measured around 4.01 (4.007813) for the tested representation.
- One llama.cpp full-model smoke at context 128, batch 1, text-only, bartowski IQ2_XXS: prefill 31.3 tok/s and decode 3.8 tok/s. Quality was not run. This is not an 8k result and does not meet the 25 tok/s decode target.
- ExLlamaV3 full-model attempts on the tested EXL3 artifact failed to fit; there is no successful ExLlamaV3 full-model tok/s.
- The full-model 27B / 8k / ≥25 tok/s target is still under investigation.

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

QTIP is a GPLv3 research reference only. No GPL code is copied into this Apache-2.0 repository. Upstream license checks for this publication are recorded in [THIRD_PARTY.md](THIRD_PARTY.md).

## Repository structure

- `src/exvram/` — contracts, hardware detection, memory calculator, adapter registry, CLI, results.
- `schemas/` — versioned experiment, result, and layer-benchmark JSON schemas.
- `adapters/{exllamav3,gemlite,cutlass,bitnet,llamacpp}` — integration notes and boundaries. Kernels are not vendored.
- `benchmark/` — protocol notes and optional full-model runner entry points.
- `experiments/weights`, `experiments/kv_cache`, `experiments/residency`, `experiments/kernels` — experiment inputs and notes.
- `experiments/manifests/` — provenance for external checkpoints. Weights themselves are not in Git.
- `experiments/results/` — small raw JSONL records and the research SQLite database.
- `search/` — future allocation and runtime search policies.
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

## Research roadmap

- P0 — infrastructure and memory planner.
- P1 — real layer-level GPU measurements. Recorded for the fixtures in this repo; not a full-model claim.
- P2 — full-model OSS shootout. A context-128 llama.cpp smoke exists; 8k context, quality, and the comparison matrix are still open.
- P3 — bottleneck-specific optimization, only after a measured full-model gap.
- P4 — optional ExVRAM runtime or orchestrator, only if P3 shows that composing existing runtimes is not enough.

## License

Original ExVRAM Lab code is Apache License 2.0. See [LICENSE](LICENSE).

Third-party projects keep their own licenses. See [THIRD_PARTY.md](THIRD_PARTY.md).

# Language migration plan

Status: **audit complete on 2026-10-12**. The user-facing runtime is already
the Rust core in `control/core` plus the Tauri shell. This pass did not
start a second runtime and did not rewrite the Python lab.

Decision labels:

- `MIGRATE` — a move is justified and, if marked done, landed in this branch.
- `KEEP` — the current language is the right one.
- `OPTIONAL` — only after a measurement shows a real gap.
- `DO_NOT_TOUCH` — a port would add code without a product gain.

## Audit

| Component | Current language | Recommended language | Reason | Expected benefit | Migration risk | Decision |
|---|---|---|---|---|---|---|
| Control window, tray, static page | HTML / JS | HTML / JS | The panel is one page. A framework port does not change inference | None measured | UI regression | KEEP |
| Tauri shell: process spawn, stop, health poll | Rust | Rust | Already supervises only the llama-server it started, at below-normal priority, bound to 127.0.0.1 | Already the product path | Rewriting it duplicates a working supervisor | KEEP |
| Core: recipe, KV choice, VRAM gate, launch arguments, GPU text parse | Rust | Rust | Decisions are pure and tested. They read the published 8k recipe instead of searching | Already no Python dependency for the panel | A second optimizer would drift from the recipe | KEEP |
| Host RAM and commit gate | Was an inline RAM check in the shell | Rust | Commit headroom was not checked, so a launch could still push Windows into the pagefile | Reliability, not speed. The check refuses under 1.5 GiB free RAM or 2 GiB free commit when Protect My PC is on | A machine with a small pagefile can now be refused even when RAM looks fine | MIGRATE, done |
| `src/exvram/hardware.py` | Python | Python | One `nvidia-smi` call dominates. Measured at 121 ms, of which `nvidia-smi` itself was 80 ms | A Rust port would not move that wait | Two detectors | KEEP |
| `src/exvram/memory.py` | Python | Python | Research estimates for the lab planner, not the live panel | None for the desktop app | Divergent budgets | KEEP |
| `src/exvram/optimizer/` | Python | Python | Search over unmeasured configs belongs to the lab. The panel already calls `recommend` and stays on measured q4_0 | None until a search result is published as a recipe | Silent quality changes | KEEP |
| Adapters (llama.cpp, ExLlamaV3, HQQ, GemLite, BitNet, Ollama) | Python | Python | They describe external tools. They do not run the desktop session | None | Breaking the lab CLI | KEEP |
| `benchmark/`, P9, quantization, imatrix, calibration | Python | Python | Evaluation and quantization stay in the lab | None | Losing the active P9 path | DO_NOT_TOUCH |
| llama.cpp and ExLlamaV3 | C++ / CUDA, external | unchanged | No measured gap that those projects do not already cover | A private engine would be a larger, slower codebase | Full inference rewrite | DO_NOT_TOUCH |
| First-party CUDA | none | none until a named gap | The roadmap already refuses a kernel without a measured miss | None today | Unmaintained kernels on SM120 | DO_NOT_TOUCH |
| Recipes and JSONL results | YAML / JSONL | same files | Both sides already share `recipes/rtx5060_8gb_qwen38_27b_8k.yaml` | One recipe, no translated copy | A second schema | KEEP |

C++ was not added. No new crate was added for the host check. The shell
calls `GlobalMemoryStatusEx` and passes the numbers into `host_pressure`.

## What this branch changed

`host_pressure` lives in `control/core`. The Start command uses it instead
of the inline 1.5 GiB comparison. Protect My PC off still skips that floor.
The file-larger-than-RAM check is unchanged. The function never kills a
process and never changes the pagefile.

A release build of the core ran 100,000 decisions in **3.08 ms** (about
31 ns each). The uncommitted lab sampler's decision loop, which also builds
a list of reasons, took **161 ms** for the same count. Neither number is
the launch time. `nvidia-smi` on this machine was **80 ms**, and
`detect_hardware()` was **121 ms**. Decode and prefill were not re-run.
The panel's working set was not re-measured; the earlier debug figure in
[CONTROL.md](CONTROL.md) still stands at about 170 MiB with the window
open and no model loaded.

## What stays on Python

The lab CLI, adapters, benchmarks, P9, quantization, calibration, and the
research optimizer. NumPy and PyTorch stay where the work is already inside
those libraries. Opening the panel does not start Python.

## What not to move next

Do not port the benchmark runners, the quality scorer, or the quantization
scripts. Do not wrap llama.cpp. Do not add a CUDA kernel on the hope that
Rust or C++ is faster. The next useful check, if one is needed, is a
release-build measurement of the panel's working set and of llama-server
startup with the new commit refusal in the path. That is a measurement,
not another port.

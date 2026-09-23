# P2 full-model results

Status: c128 throughput smoke measured as of 2026-09-23; the 8k target and quality gate are
still open. This document intentionally separates provenance, preflight, and measured results.
No unmeasured full-model tok/s, VRAM fit, or quality result is claimed.

## Fixed target and protocol

- Checkpoint identity: `Qwen/Qwen3.8-27B`, HF revision
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, Apache-2.0.
- Hardware: NVIDIA RTX 5060, 8151 MiB reported by the driver, Torch-visible compute capability
  12.0 / SM120.
- Workload: text-only input, batch 1, contexts 128, 2048, and 8192, with prefill and decode
  recorded separately where the external runtime exposes both.
- Quality: not run by the throughput smoke commands; quality is a separate gate and cannot be
  replaced by a quantizer's self-reported perplexity table.
- Quality suite scaffold: [`experiments/quality/qwen38_mini_suite.json`](../experiments/quality/qwen38_mini_suite.json).
- Native architecture caveat: Qwen3.8-27B is a vision-language checkpoint with a hybrid
  linear-attention/GDN and full-attention trunk plus MTP. “Text-only” means no image input.

## Candidate provenance

The machine-readable source of truth is
[`experiments/manifests/qwen38_p2.json`](../experiments/manifests/qwen38_p2.json). It records
immutable HF revisions, file sizes, artifact terms, and compatibility state. Large files stay
outside Git in an external data directory.

## Preflight observations

- Installed ExLlamaV3 `1.5.1` contains the `Qwen3_5ForConditionalGeneration` architecture and
  constructs a separate `text` component. The downloaded EXL3 metadata has 64 layers, 5120
  hidden size, 24 attention heads, 4 KV heads, 16 GDN key heads, 48 GDN value heads, one MTP
  layer, and a vision config. This is compatibility evidence, not an inference result.
- Official llama.cpp Windows x64 CUDA 13.3 nightly `b10964` was staged. Its first device probe
  returned `(none)` until the separate official CUDA DLL bundle was installed; the probe is
  repeated after that environment repair.
- The selected standard GGUF low-memory candidate is bartowski IQ2_XXS. Its model card reports
  a 2.56 bpw artifact and self-reported quality statistics; those figures are provenance only,
  not ExVRAM quality measurements.

## Measured c128 smoke

The following is a real text-only run, not synthetic data:

| Backend / artifact | Status | Prefill | Decode | Context | KV cache | Quality |
|---|---:|---:|---:|---:|---|---|
| llama.cpp b10964 / bartowski IQ2_XXS | PASS | 31.3 tok/s | 3.8 tok/s | 128 | q4_0 / q4_0 | NOT_RUN |

The run took 284.83 s wall time and returned code 0. The final llama.cpp fit breakdown reported
744 MiB free on the 8150 MiB CUDA device, with 5673 MiB model, 112 MiB context, and 71 MiB
compute allocation in that final snapshot. This is a c128 observation, not an 8k fit claim.

The ExLlamaV3 runner has recorded failed full-model fit attempts for this artifact/runtime;
there is no successful ExLlamaV3 full-model inference result yet. The measured llama.cpp decode
rate is below the 25 tok/s research hypothesis, so P3 is not selected.

## Raw result files

- ExLlamaV3: `experiments/results/p2_exllama_full_model.jsonl`
- llama.cpp public summary: `experiments/results/p2_llamacpp_full_model_public.jsonl`
- The raw llama.cpp capture remains local-only because it contains host-specific absolute paths.
- Queue/database: `experiments/results/p2_queue.jsonl` and `experiments/results/research.sqlite`

The c128 llama.cpp row is measured, but P2 remains `INCONCLUSIVE` for the stated target until
the 8k context, quality gate, and comparison matrix are completed. P3 must not be selected from
synthetic or layer-only numbers.

## Storage safety boundary

Large model and runtime artifacts used by earlier GPU measurements stay outside the repository.
CPU, synthetic, lint, and unit-test workflows do not need that external volume. Do not start
another long GPU run while removable storage is unstable. A future full-model run should first
place the required artifacts on a stable data volume, use a bounded timeout, and run in the
foreground so it can be stopped safely.

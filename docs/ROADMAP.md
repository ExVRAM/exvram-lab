# Roadmap after the occupied-8k speed result

Status: **planned**. The occupied-8192 UD-IQ2_XXS speed result is measured.
Nothing in the phases below is a new measurement.

The hardware claim is already met: dense ~27B, RTX 5060 8 GB, 65/65 CUDA
layers, CPU decoder offload 0, occupied 8192-token prompt, q4_0 KV, median
decode 30.10 tok/s over 5 runs, CV 2.0%, peak 7767 MiB. The open risk is
whether that checkpoint stays capable, whether a minimal-refusal twin can be
reproduced, and whether another user can launch it.

## P9 — Quality and refusal validation

This is the active phase.

1. Base quality gate against a higher-quality reference of the same model
   family. Same tokenizer, prompts, seed, and decoding settings. Suites:
   reasoning, math, coding, factual QA, instruction following, long-context
   retrieval. Status remains `NOT_RUN`. Throughput is not a quality score.
2. Coletti `JonathanColetti/Qwen3.8-27B-Uncensored` revision
   `5bb7aa90f0efef548e87005b1fb7658e522b6b7f`: local text-only BF16 GGUF
   (53,808,282,560 bytes) → a finished imatrix → IQ2_XXS on the same
   pipeline as the base UD quant → full GPU → occupied 8k → speed →
   quality. The 13,582,676-byte imatrix file is a partial calibration and
   is not an input. No Heretic run and no fine-tune.
3. Refusal, measured here on the final quant, not copied from upstream.
   Two separate rates: refusal on requests the model is expected to refuse,
   and over-refusal on ordinary benign requests.

No Hugging Face upload in P9.

## P10 — Runtime performance

Starts after P9 reports exist.

- Report prefill, time to first token, and decode separately. One combined
  tok/s figure is not the UX result. The occupied-8192 prefill on the base
  quant was already about 243.8 s median; that is a measured input to this
  phase, not a new optimized result.
- Close the occupied-8192 KV matrix. q4_0 is the current leader at median
  30.10 tok/s and about 148.5 MiB. q8_0 fit and its median of all five runs
  was 20.31 tok/s, with one much slower run. f16 loaded and was not
  measured. Name a final winner only after f16 is measured or explicitly
  abandoned.
- Speculative decoding only after that split and the KV matrix: N-gram,
  then MTP, then DFlash2. The hope is 35–45+ effective tok/s without more
  VRAM and without a quality regression. That range is a target, not a
  result.

## P11 — Productization

Starts after P9 and P10.

ExVRAM Runtime is a thin orchestrator over an existing backend, not a new
inference engine. It detects GPU and VRAM, selects a published recipe,
checks the model artifact, launches the backend, picks KV and context,
refuses CPU decoder offload when the goal is performance, and exposes an
OpenAI-compatible endpoint. Launch is one command.

The Hugging Face release is the winning minimal-refusal GGUF, not a git
weight commit. The card carries SHA256, source revision, imatrix
provenance, quantizer revision, the RTX 5060 measurements, quality and
refusal results, and the recipe link.

## Held

P8 1-bit, HQQ, GemLite, PB-LLM, and BiLLM stay on `experiment/p8-low-bit`.
The tree has a protocol, adapter boundaries, and a physical-bpw planner.
No 1-bit GPU or quality result is measured. That track does not block the
27B / 8 GB release. It resumes only if a layer-level test shows a real
Pareto gain.

No first-party CUDA or Triton kernel. The speed target was met with
existing llama.cpp. A kernel waits on a named measured gap, for example
700 MiB saved at under 5% speed loss, or a 20% `lm_head` speedup. No such
gap is recorded.

## Planning estimate

These percentages are an operator judgment on 2026-10-02. They are not
measurements.

| Area | Estimate |
|---|---|
| Research infrastructure | ~85% |
| Hardware-concept proof | ~90% |
| Quality and steerability | ~30% |
| Reproducible model artifact | ~40% |
| End-user product | ~20% |

v0.1 means this 27B checkpoint is still capable enough, the two refusal
rates are measured, and another user can reproduce the launch. The speed
result alone is not v0.1.

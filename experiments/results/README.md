# Experiment results

This directory stores small, reviewable JSONL records produced by ExVRAM workflows.

- Public summaries must contain provenance, configuration, measured metrics, and quality-gate
  state without host-specific absolute paths.
- Raw captures from external runtimes may remain local-only when they contain machine paths,
  verbose logs, or other host-specific data. The local llama.cpp capture is ignored by Git;
  `p2_llamacpp_full_model_public.jsonl` is the sanitized publication record. The Ollama smoke
  has the same split: `ollama_smoke_public.jsonl` is a small 7B baseline summary, while local
  verbose captures remain optional and are not required for the repository.
- Model weights, runtime binaries, virtual environments, caches, and multi-gigabyte artifacts do
  not belong here.
- `synthetic: true` is required for synthetic measurements; synthetic records must not claim
  tok/s for model inference.

The schemas in `schemas/` define the stable record contracts. P2 provenance and queue state live
in `experiments/manifests/` and the research plan, not in model files.

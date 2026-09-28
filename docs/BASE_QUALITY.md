# Base IQ2_XXS quality gate

Status: **NOT RUN**.

No quality score is claimed for Qwen3.8-27B IQ2_XXS. The repository contains
quality-gate scaffolding that fails closed when no reference metrics or fixed
corpus are supplied. A valid future run must use the same tokenizer, prompts,
seed, decoding settings and reference path for:

- reasoning;
- math;
- coding;
- instruction following;
- factual QA;
- long-context retrieval.

Raw model outputs and scores must be stored before declaring a base 8k recipe
reproducible. P6 intentionally did not download a reference model or invent a
capability score from throughput measurements.

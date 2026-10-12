# P9 quality comparison

Question: can UD-IQ2_XXS do the same tasks as official Qwen3.8-27B BF16,
within the locked margins in [P9_EVAL_PROTOCOL.md](P9_EVAL_PROTOCOL.md)?

Gate for that question: **`REFERENCE_UNAVAILABLE`**.

No BF16 arm was run. No higher-bit proxy was run. A ratio against a missing
reference is not a pass, a fail, or a claim of no loss. The program verdict
stays `P9_INCONCLUSIVE` because Coletti IQ2, refusal, coding, and long
context are also missing.

## Arms

| Arm | What it is | This host | Status |
|---|---|---|---|
| A | Official Qwen3.8-27B BF16 | Does not fit 16 GB RAM. Not loaded | NOT_RUN |
| B | Unsloth `Qwen3.8-27B-UD-IQ2_XXS.gguf` on RTX 5060 | 305 samples in `p9_base_quality.jsonl` | MEASURED, partial |
| C | `JonathanColetti/Qwen3.8-27B-Uncensored` revision `5bb7aa90f0efef548e87005b1fb7658e522b6b7f` | Text BF16 GGUF exists. Not loaded | NOT_RUN |
| D | IQ2_XXS derived from C | Not produced. Local BF16 imatrix is not allowed on this host | NOT_RUN |
| E | Ternary-Bonsai | Not downloaded. Different weights. See [P9_BONSAI_COMPARISON.md](P9_BONSAI_COMPARISON.md) | NOT_RUN |

Comparisons that this file does not contain, because one or both arms are
missing: A vs B, C vs D, A vs C, B vs D, B vs E.

## Absolute scores for B only

Wilson intervals are on one arm. They are not intervals on a difference.

| Category | IQ2 | BF16 | Delta | Status |
|---|---:|---|---|---|
| Reasoning | 0.550 (33/60) | — | — | MEASURED absolute, delta NOT_RUN |
| Math | 0.890 (89/100) | — | — | MEASURED absolute, delta NOT_RUN |
| Coding | — | — | — | NOT_RUN |
| Knowledge (TruthfulQA binary, not MMLU-Pro) | 0.800 (80/100) | — | — | MEASURED absolute, delta NOT_RUN |
| Instructions | 0.867 (39/45) | — | — | MEASURED absolute, short of n=50, delta NOT_RUN |
| Long context 2k / 4k / 8k | — | — | — | NOT_RUN |

MMLU-Pro was not selected. The locked factual task is TruthfulQA binary.
HumanEval was selected and not run. GSM8K and the three BBH tasks were run
under the locked subset seed.

## Pareto

Decode and VRAM for B are the existing P6 occupied-8192 series, not a new
measurement: median decode 30.10 tok/s, peak 7767 MiB of 8151. Quality cells
below are the partial P9 scores. Empty cells were not measured.

| Model | VRAM | Decode | Reasoning | Math | Coding | Knowledge | Instructions | Long context |
|---|---|---|---|---|---|---|---|---|
| A BF16 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| B IQ2 | MEASURED 7767 MiB peak | MEASURED 30.10 tok/s | 0.550 | 0.890 | NOT_RUN | 0.800 | 0.867 (n=45) | NOT_RUN |
| C Coletti BF16 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| D Coletti IQ2 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| E Bonsai | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |

Capability retention versus BF16: NOT_RUN. Quality per GB: not computed,
because the numerator would be an average of an unfinished suite and the
denominator would have no BF16 pair. The 8 GB constraint is already the
machine B ran on. That does not grade B against A.

## What would change the gate

A same-protocol reference on another host, or a remote endpoint pinned to
the same revision, scored on the same item ids. A higher-bit proxy is
allowed only if the report names it as a proxy. Until one of those exists,
the gate stays `REFERENCE_UNAVAILABLE`.

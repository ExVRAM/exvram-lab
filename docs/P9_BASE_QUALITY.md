# P9 base quality

Status: **partial**. Protocol `p9-2026-10-02-1`. The file
`experiments/results/p9_base_quality.jsonl` holds 305 measured samples for
`base_iq2_xxs`, plus one status row. Every sample is
`full_gpu: true`, SHA-256
`e792d8fb3142fe6d9171876d6da0f71f05a71028718debc72dbec93ff645e67d`,
7,266,070,528 bytes.

Reference kind: `UNAVAILABLE`. There is no `REFERENCE_BF16` and no
`REFERENCE_PROXY`. These numbers are absolute accuracies of the IQ2 file.
They are not a loss against BF16.

Verdict: `BASE_QUALITY_INCONCLUSIVE`. `quality_verdict` returns
`INCONCLUSIVE` when the reference is missing. Coding and long context have
no samples. Instruction has 45 of the locked 50. Two reasoning rows and
seven math rows hit the old token cap and were not rerun at 512 / 768.

| Category | Correct / scored | Wilson 95% | Protocol n | Status |
|---|---:|---|---:|---|
| Reasoning | 33/60 (0.550) | 0.425–0.669 | 60 | MEASURED, not closed: 2 rows hit `n_predict` 64 |
| Math | 89/100 (0.890) | 0.814–0.937 | 100 | MEASURED, not closed: 7 rows hit `n_predict` 256 |
| Factual | 80/100 (0.800) | 0.711–0.867 | 100 | MEASURED. None hit the cap of 16 |
| Instruction | 39/45 (0.867) | 0.738–0.937 | 50 | MEASURED, short by 5 items |
| Coding | — | — | 40 | NOT_RUN |
| Long context | — | — | 15 × 3 | NOT_RUN |

Reasoning by task, same file: logical deduction 14/20 (0.481–0.855),
tracking shuffled objects 9/20 (0.258–0.658), date understanding 10/20
(0.299–0.701).

Stored decoding is temperature 0.0, seed 42, no added system prompt, chat
mode. The token cap is mixed: reasoning 47 rows at 64 and 13 at 512; math
40 at 256 and 60 at 768; instruction 9 at 256 and 36 at 768; factual 100
at 16. The locked caps are 512 / 768 / 512 / 16 / 768 / 64. Rows that
stopped before the old cap were kept, which matches the changelog. Rows
that hit the old cap were not rerun.

The continuation was not started on 2026-10-12. GPU was idle (4%, 38 C,
19.9 W, 1367 MiB used of 8151). Free physical RAM was about 4.7 GB of
16.3 GB. That is enough for the runner's VRAM gate (2048 MiB) and not
enough headroom to load this 7.3 GB GGUF without risking the desktop.
`experiments/results/p9_long_context.jsonl` is still the `not_run` stub.

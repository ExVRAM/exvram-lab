# P9 final

Comparison gate: **`REFERENCE_UNAVAILABLE`**.

Program verdict: **`P9_INCONCLUSIVE`**. P10 does not start.

Official Qwen3.8-27B BF16 was not run. UD-IQ2_XXS has a partial absolute
suite only. That cannot support `QUALITY_PRESERVED_WITHIN_MARGIN`,
`QUALITY_DEGRADED`, `QUALITY_MIXED`, or any zero-loss label. Coding, long
context, Coletti IQ2, and both refusal gates are missing. Nine math and
reasoning rows still sit on the old token cap.

`P9_SUCCESS` is not available from the P6 speed result.

| Report field | Value |
|---|---|
| Base reference | `UNAVAILABLE`. BF16 was not loaded |
| Models actually scored | B only: `Qwen3.8-27B-UD-IQ2_XXS.gguf` |
| Items scored | 305. Locked suite is 60+100+40+100+50+45 = 395, plus refusal |
| Benchmarks | BBH three tasks, GSM8K, TruthfulQA binary, IFEval subset. HumanEval and the needle set were not run |
| Reasoning delta | NOT_RUN. Absolute 33/60 |
| Math delta | NOT_RUN. Absolute 89/100, of which 7 misses are truncated |
| Coding delta | NOT_RUN |
| Knowledge delta | NOT_RUN. Absolute 80/100 |
| Instruction delta | NOT_RUN. Absolute 39/45 |
| Long-context delta | NOT_RUN |
| Uncertainty | Wilson 95% on the IQ2 arm only. No paired interval |
| IQ2 vs BF16 | `REFERENCE_UNAVAILABLE` |
| IQ2 vs Bonsai | NOT_RUN. Different weights |
| Coletti IQ2 vs Coletti BF16 | NOT_RUN. D was not produced |
| Strongest absolute miss | Tracking shuffled objects 9/20 and date understanding 10/20. Cause versus BF16 unknown |
| Practical use | Factual 80/100 and the finished math rows are usable evidence that this file answers those items. They are not a retention rate |
| Reliable so far | TruthfulQA binary at 0.800, n=100, no cap hits |
| Not established | Coding, long context, quantization loss, layer blame |
| Next experiment | Same item ids on official BF16, on a host that can hold it. On this PC, only if free RAM recovers: rerun the 9 cap-hit rows, the 5 missing IFEval items, HumanEval 40, then the needle set |
| Coletti IQ2 bytes | not produced |
| Refusal rates | not measured |
| P10 | no |

Layer blame (attention, MLP, embeddings, `lm_head`, calibration) is not
available from unpaired scores. EXL3, a denser imatrix, and Bonsai stay
candidates to test after a reference exists. No new CUDA kernel follows
from this file.

Machine-readable copy of the table above:
`experiments/results/p9_quality_summary.json`.

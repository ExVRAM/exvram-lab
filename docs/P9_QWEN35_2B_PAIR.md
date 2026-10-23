# P9 Qwen3.5-2B pair

Study label: **`METHOD_COMPARISON_NOT_27B_RETENTION`**.

This is the BF16-versus-UD-IQ2_XXS pair that fits this PC. It uses
Qwen3.5-2B, a smaller network with the same 3:1 Gated DeltaNet hybrid.
It is not official Qwen3.8-27B, and it does not move the 27B gate off
`REFERENCE_UNAVAILABLE`.

Protocol `p9-2026-10-02-1`. Same sample ids, prompt SHA-256 values, gold
answers, temperature 0, seed 42, and caps on both arms. Prompt and gold
mismatches: 0. Machine-readable copy:
`experiments/results/p9_qwen35_2b_paired.json`.

## Weights

| Arm | File | Bytes | SHA-256 | What it is |
|---|---|---:|---|---|
| BF16 | `models/qwen35-2b/Qwen3.5-2B-BF16.gguf` | 3,775,709,216 | `dc11be2ca4519954a3e4b013ef59497b5522209987d67891851113eda89619f1` | Unsloth GGUF, revision `f6d5376be1edb4d416d56da11e5397a961aca8ae`. Reference kind `UNSLOTH_BF16_GGUF` |
| IQ2 | `models/qwen35-2b/Qwen3.5-2B-UD-IQ2_XXS.gguf` | 768,270,592 | `43aedbd2b03a3c2cc39f49ccf74fcd3c394ed0b2a1ede8a30ee652ee9cfc27ef` | Same Unsloth revision, UD-IQ2_XXS |

Official `Qwen/Qwen3.5-2B` safetensors are 4,548,221,488 bytes and include
vision. They were not executed. The BF16 arm is the text GGUF above.

llama.cpp b10982, same server flags as the 27B suite: context 8448, batch
32, ubatch 1, q4_0 KV, flash-attn, 4 threads, `--reasoning off`,
`--load-mode dio`. The runner's `full_gpu` field is false on both files
because that flag is set only when the log says `65/65`. These models have
24 repeating layers. The logs say `offloaded 25/25 layers to GPU`.

| Arm | CUDA0 weights | CUDA_Host weights | KV | Graph splits | Tensor types |
|---|---:|---:|---:|---:|---|
| BF16 | 3590.35 MiB | 970.00 MiB | 27.84 MiB | 2 | f32 133, bf16 187 |
| IQ2 | 722.23 MiB | 333.44 MiB | 27.84 MiB | 2 | f32 133, q8_0 36, q4_K 18, q5_K 1, q2_K 3, iq3_xxs 24, iq3_s 5, iq2_s 5, iq2_xxs 95 |

The host buffer is the embedding table on both arms. Repeating layers and
the output layer were on the GPU. The two arms share that split, so the
score gap is the quantizer under the same launch, not a GPU-versus-CPU
decoder difference.

BF16 wall time 2,768,691 ms, exit 0, 2026-10-13T17:45:05Z. During that run
a snapshot showed about 5.2 GB RAM free, GPU memory around 4458–4732 MiB,
58–60 C, about 100–121 W. IQ2 wall time 1,564,260 ms, exit 0,
2026-10-13T18:11:29Z. A snapshot then showed about 7.4 GB RAM free, GPU
1653 MiB, 59 C, 103 W. Those are snapshots, not a traced peak.

Median `predicted_per_second` stored on the sample rows, not a separate
speed protocol: all samples BF16 83.76 tok/s and IQ2 216.25 tok/s; the
fifteen 8192-token needles BF16 79.28 tok/s and IQ2 182.00 tok/s.

## Scores

`quality_verdict` on the point estimates is **`FAIL`**. The study call is
**`QUALITY_DEGRADED`**: the bootstrap 95% interval on the aggregate
difference lies entirely below 0.90 of the BF16 aggregate. A category can
still sit inside its own ratio. That happened for long context only.

The paired interval is the Wald 95% interval on (IQ2 accuracy − BF16
accuracy). The aggregate interval is a 10,000-draw paired bootstrap, seed
42, because the aggregate is the unweighted mean of six categories.
Bootstrap category intervals agree with the Wald call on every row.
`INSIDE_MARGIN` means the whole interval is at least as good as the locked
ratio. `OUTSIDE_MARGIN` means the whole interval is worse. A crossing
interval is `INCONCLUSIVE` for that row.

| Category | BF16 | IQ2 | Delta | 95% interval | Margin on the delta | Call |
|---|---:|---:|---:|---|---:|---|
| Reasoning | 29/60 (0.483) | 8/60 (0.133) | −0.350 | −0.502 to −0.198 | −0.073 (ratio 0.85) | OUTSIDE_MARGIN |
| Math | 81/100 (0.810) | 6/100 (0.060) | −0.750 | −0.839 to −0.661 | −0.203 (ratio 0.75) | OUTSIDE_MARGIN |
| Coding | 13/40 (0.325) | 1/40 (0.025) | −0.300 | −0.442 to −0.158 | −0.049 (ratio 0.85) | OUTSIDE_MARGIN |
| Factual | 72/100 (0.720) | 42/100 (0.420) | −0.300 | −0.422 to −0.178 | −0.180 (ratio 0.75) | STRADDLES_MARGIN |
| Instruction | 43/50 (0.860) | 10/50 (0.200) | −0.660 | −0.803 to −0.517 | −0.215 (ratio 0.75) | OUTSIDE_MARGIN |
| Long context | 45/45 (1.000) | 43/45 (0.956) | −0.044 | −0.105 to +0.016 | −0.250 (ratio 0.75) | INSIDE_MARGIN |
| Aggregate | 0.700 | 0.299 | −0.401 | −0.451 to −0.352 | −0.070 (ratio 0.90) | OUTSIDE_MARGIN |

Point ratios, IQ2 divided by BF16: reasoning 0.276, math 0.074, coding
0.077, factual 0.583, instruction 0.233, long context 0.956, aggregate
0.427. The locked floors are 0.85, 0.75, 0.85, 0.75, 0.75, 0.75, and 0.90.

Items where BF16 was right and IQ2 was wrong, then the reverse: reasoning
25 and 4, math 76 and 1, coding 12 and 0, factual 39 and 9, instruction
34 and 1, long context 2 and 0.

Reasoning by task: BF16 logical deduction 12/20, tracking 10/20, date
understanding 7/20. IQ2 5/20, 2/20, and 1/20.

Long context by length: BF16 15/15 at 2048, 4096, and 8192. IQ2 15/15,
14/15, and 14/15. The two misses stopped by themselves (`predicted_n` 6,
cap 64) and dropped the last digit of the 6-digit code: `needle:4096:3`
early, gold `460936`, prediction `46093`; `needle:8192:7` middle, gold
`858687`, prediction `85868`. That interval still sits inside the 0.75
ratio. It includes differences other than zero, so it is not `ZERO_LOSS`.

## What the misses look like

IQ2 math: 94 wrong. 84 of those filled the locked cap of 768, and 10
finished with a wrong last number. The gold number string does not appear
in any of the 94 saved wrong answers. Five further math rows filled 768
and were still scored right; one right answer finished early. BF16 math
filled the cap 5 times, all wrong. The IQ2 openings that were inspected
are repetitive step text that never reaches the gold value.

IQ2 factual: 98 of 100 rows filled the 16-token cap. Of the 58 wrong rows,
34 contain a different leading A–D letter and 24 contain no leading letter.
BF16 factual filled that cap 0 times. The point estimate is below 0.75 of
BF16. The paired interval's upper end is −0.178 against a margin of
−0.180, so this one category stays `INCONCLUSIVE`.

IQ2 instruction: 45 of 50 rows filled 768 tokens, and 38 of those were
wrong. BF16 filled that cap twice. IQ2 coding: 20 of 40 rows filled 512
tokens, all 20 wrong; the single pass stopped early. BF16 coding filled
512 twice, both wrong, with 13 passes.

The caps were not raised. A higher cap would be a new protocol and would
have to be run on both arms.

## What this does not say

The 27B IQ2 absolute suite is a different network. Its aggregate on the
same protocol is 0.774, which is higher than this 2B BF16 aggregate of
0.700 and much higher than this 2B IQ2 aggregate of 0.299. Those three
numbers are not a retention ratio. Official 27B BF16 was not loaded.
Coletti IQ2, Bonsai, EXL3, and both refusal gates were not run.

## Q8_0, one GPU graph

Measured 2026-10-14 against the same BF16 arm and the same protocol.
File `Qwen3.5-2B-Q8_0.gguf`, 2,012,012,800 bytes, SHA-256
`1b04acba824817554f4ce23639bc8495ff70453b8fcb047900c731521021f2c1`.
The loader reports f32 133 and q8_0 187. This is a uniform 8-bit quant,
not the IQ2 mix.

The suite log with `--override-tensor token_embd.weight=CUDA0` says the
CUDA0 weight buffer is 1908.35 MiB, the KV buffer is 27.84 MiB, the
recurrent state is 19.27 MiB, and the CUDA_Host compute scratch is 0.29
MiB. `graph splits = 1`. There is no CPU weight buffer. Without that
override the same file kept a 515.31 MiB embedding copy on the host and
split the graph in two. A snapshot during the GPU run showed 3099 MiB of
GPU memory, 54 C, about 94 W, and about 5.8 GB of RAM free. That is a
snapshot, not a traced peak.

Point-estimate `quality_verdict`: **`PASS`**. Aggregate 0.693 against the
BF16 aggregate 0.700, which is 0.991 of BF16. The locked aggregate floor
is 0.90, and the bootstrap interval on the difference stays above it.

| Category | BF16 | Q8 on GPU | Share of BF16 | Call |
|---|---:|---:|---:|---|
| Reasoning | 29/60 | 30/60 | 1.034 | INSIDE_MARGIN |
| Math | 81/100 | 78/100 | 0.963 | INSIDE_MARGIN |
| Coding | 13/40 | 14/40 | 1.077 | STRADDLES_MARGIN |
| Factual | 72/100 | 69/100 | 0.958 | INSIDE_MARGIN |
| Instruction | 43/50 | 42/50 | 0.977 | INSIDE_MARGIN |
| Long context | 45/45 | 45/45 | 1.000 | INSIDE_MARGIN |
| Aggregate | 0.700 | 0.693 | 0.991 | INSIDE_MARGIN |

Coding is one item above BF16. Three coding items disagree, so the Wald
interval still reaches past the 0.85 coding ratio. That category stays
`INCONCLUSIVE` under the paired-interval rule. The point estimate clears
every locked floor, including 0.90 of the aggregate.

An earlier Q8 run without the embedding override scored the same 395 items.
Zero items changed when the embedding op moved onto the GPU.

Rows: `experiments/results/p9_qwen35_2b_q8_gpu.jsonl`,
`experiments/results/p9_qwen35_2b_q8_gpu_long.jsonl`, and
`experiments/results/p9_qwen35_2b_q8_gpu_paired.json`.
A Q4, Q6, or Q8 file of the 27B does not fit this card. This section does
not supply the missing 27B BF16 reference.

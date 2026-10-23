# P9 quality breakpoint

Status: **partial**. No model was started for this note. The numbers below
are counts of rows already stored under protocol `p9-2026-10-02-1`.
Empty cells were not measured. The locked pass ratios were not changed.

Study label for the 2B rows: `METHOD_COMPARISON_NOT_27B_RETENTION`.
The 27B IQ2 file is a different network. Its scores are not a ratio
against 2B BF16 and not a ratio against 27B Q4.

## What was already done

| Item | State |
|---|---|
| 27B UD-IQ2_XXS absolute suite, 395 rows | Measured. Aggregate 0.774. Reference `UNAVAILABLE` |
| 27B error table | Measured classes in [P9_ERROR_ANALYSIS.md](P9_ERROR_ANALYSIS.md). No BF16 arm |
| 2B BF16 vs UD-IQ2_XXS, same ids | Measured. Call `QUALITY_DEGRADED`. See [P9_QWEN35_2B_PAIR.md](P9_QWEN35_2B_PAIR.md) |
| 2B Q8_0, embedding on CUDA0 | Measured. Point estimate `PASS`, aggregate 0.693 vs BF16 0.700 |
| 2B Q4_K_M short suite, 350 rows | Measured. Same prompt SHA-256 as BF16 on every short id. No summary row |
| 2B Q4_K_M long context | Partial: 20/45 stored, all 20 correct. 25 needles are absent |
| Q5, Q3, IQ3 | Not run |
| LONGTASK-V1 | Not run. Spec in [LONGTASK_V1.md](LONGTASK_V1.md), scorer in `src/exvram/longtask_v1.py` |
| 27B Q4 | Not run. The published file is larger than this card |

Q4 weights: `Qwen3.5-2B-Q4_K_M.gguf`, 1,280,835,840 bytes, SHA-256
`aaf42c8b7c3cab2bf3d69c355048d4a0ee9973d48f16c731c0520ee914699223`,
role `qwen35_2b_q4_k_m`. Same protocol, temperature 0, seed 42, no system
prompt. Caps match the lock: reasoning 512, math 768, coding 512,
factual 16, instruction 768.

## Measured 2B scores

Six-category aggregate uses the unweighted mean. Q4 has no six-category
aggregate because 25 long-context rows are missing.

| Format | File bytes | Reasoning | Math | Coding | Factual | Instruction | Long context | Aggregate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BF16 GGUF | 3,775,709,216 | 29/60 | 81/100 | 13/40 | 72/100 | 43/50 | 45/45 | 0.700 |
| Q8_0 GPU | 2,012,012,800 | 30/60 | 78/100 | 14/40 | 69/100 | 42/50 | 45/45 | 0.693 |
| Q4_K_M | 1,280,835,840 | 29/60 | 71/100 | 12/40 | 67/100 | 41/50 | 20/20 stored, 25 absent | |
| IQ2_XXS | 768,270,592 | 8/60 | 6/100 | 1/40 | 42/100 | 10/50 | 43/45 | 0.299 |
| Q5_K_M | | | | | | | | |
| Q3_K_M | | | | | | | | |

Short-sample medians of `predicted_per_second` and `prompt_ms` over the
350 short rows. These are not the occupied-8192 speed protocol.

| Format | Decode tok/s | Prefill tok/s |
|---|---:|---:|
| BF16 | 88.11 | 119.22 |
| Q8_0 GPU | 141.19 | 184.28 |
| Q4_K_M | 186.74 | 252.46 |
| IQ2_XXS | 217.22 | 287.83 |

Decode is the median of `predicted_per_second` on rows with at least 8
predicted tokens. Prefill is the median of `prompt_per_second` on the
short rows. `prompt_ms` is prompt time. A separate TTFT was not stored.
Cap-hit IQ2 rows make its decode sample longer than the other arms.

Recorded llama-server weight buffers at context 8448, not a traced
nvidia-smi peak of the quality run: Q4_K_M CUDA0 1211.05 MiB with the
embedding on GPU; Q8_0 CUDA0 1908.35 MiB, one graph; IQ2 CUDA0 722.23 MiB
plus a 333.44 MiB host embedding; BF16 CUDA0 3590.35 MiB plus a 970 MiB
host embedding. Q4 cap hits on the short suite: math 7/100, coding 3/40,
instruction 2/50, reasoning 0/60, factual 0/100.

Q8 stays inside the locked aggregate ratio of 0.90 (share 0.991). Coding
on Q8 still straddles its own 0.85 ratio. IQ2 falls outside the aggregate
interval. Q4's finished categories sit near BF16. Paired Wald 95% intervals on
Q4 minus BF16: reasoning −0.092 to +0.092, math −0.176 to −0.024,
coding −0.110 to +0.060, factual −0.108 to +0.008, instruction −0.150
to +0.070. Math is the only closed cell whose interval lies entirely
below zero. The others cross zero. That is not a six-category call.

```mermaid
xychart-beta
    title "2B six-category aggregate, measured arms only"
    x-axis ["BF16", "Q8", "IQ2"]
    y-axis "Aggregate" 0 --> 1
    bar [0.700, 0.693, 0.299]
```

## Why 2B IQ2 scores 6/100 on GSM8K

Same prompts, same caps, same `--reasoning off`, same chat mode as BF16,
Q8, and Q4. The launch is not the difference. Classes are from
`exvram.breakpoint.math_failure_class` on the stored IQ2 math rows:

| Class | IQ2 rows |
|---|---:|
| ok | 6 |
| repetition (a line longer than 40 characters repeated at least 3 times) | 48 |
| cap hit and no `####` marker | 26 |
| cap hit and a wrong `####` marker | 10 |
| finished with a wrong marker | 10 |
| first `####` matched gold while the stored score was wrong | 0 |

94 predictions are wrong. 84 of those hit the 768 cap. 48 of the wrong
rows repeat one line, and 26 hit the cap with no `####` marker.
On BF16, Q8, and Q4, zero wrong rows had a repeated line, and zero wrong
rows had a first `####` that matched gold while the score said wrong.
The scorer is not throwing away a correct first marker on this file.
The IQ2 answers loop and stop at the cap. Raising the cap was not done.
More tokens of the same loop are not evidence the answer would appear.

27B IQ2 math on the same protocol is 89/100 absolute, with 4 finished
wrong numbers and 7 rows cut at 768. That is not the 2B loop, and it is
not a comparison with 27B BF16.

## Long tasks and context

LONGTASK-V1 is specified in [LONGTASK_V1.md](LONGTASK_V1.md) and
`src/exvram/longtask_v1.py`. It has not been executed. The checker scores
saved text and, for the ledger only, runs candidate modules under
`python -I` for at most 5 seconds. `MODEL_RUN` is `NOT_RUN`.

Stored needles, actually filled:

| Arm | 2048 | 4096 | 8192 |
|---|---:|---:|---:|
| BF16 | 15/15 | 15/15 | 15/15 |
| Q8_0 GPU | 15/15 | 15/15 | 15/15 |
| Q4_K_M | 15/15 | 5/15 | |
| IQ2 | 15/15 | 14/15 | 14/15 |

The two IQ2 misses dropped the last digit of a 6-digit code and stopped
on their own (`predicted_n` 6, cap 64). Filling a longer prompt is not,
by itself, a long task.

## 27B, without a 2B ratio

The working file remains UD-IQ2_XXS, 7,266,070,528 bytes, peak 7767 MiB
at occupied 8192. Absolute weak cells are coding 21/40 and tracking
shuffled objects 9/20. Long context is 45/45. Math does not show the 2B
repetition collapse. No 27B Q4 or BF16 arm exists, so none of those cells
are a quantization loss against a 27B reference.

A head-only lift of `output.weight` from `q3_k` to `q4_k` fits the derived
16k spare. Embeddings and the `iq1_s` body do not. That recipe is not a
measured quality gain. See [QUANT_PROVENANCE.md](QUANT_PROVENANCE.md).

## Aggregate gate for the missing needles

Checked against `aggregate_score` and `quality_verdict` in
`src/exvram/p9_eval.py`, using the stored counts. BF16 aggregate is
0.699722. The locked 0.90 ratio is 0.62975. Q4's five finished accuracies
are fixed. Long context is the unweighted mean of the three lengths, which
equals correct/45 because each length has 15 items.

| Long-context correct | Q4 aggregate | Share of BF16 | Point verdict |
|---:|---:|---:|---|
| 35/45 | 0.626852 | 0.896 | FAIL |
| 36/45 | 0.630556 | 0.901 | PASS |

36 is the smallest total that passes. The category floor for long context
is 0.75 of BF16's 45/45, which 34/45 already clears. The aggregate is the
tighter rule. The 20 stored needles are all correct, so 36/45 means at
least 16 of the remaining 25 must be correct. That is a point estimate.
It is not equivalence, and it does not replace the paired interval.

The five finished categories already clear their own point floors:
reasoning 29/29 of BF16, math 71/81, coding 12/13, factual 67/72,
instruction 41/43. Math's paired Wald interval on the finished 100 items
stays entirely below zero (−0.176 to −0.024). The point floor can pass
while the arm is still statistically below BF16. Q3 and Q5 stay unstarted
until the 25 rows exist.

## Resume command, not started

The JSONL ids were checked on 2026-10-23. All 370 stored Q4 rows have
`measurement_status` `measured`. The missing ids are exactly
`needle:4096:5` through `needle:4096:14` and `needle:8192:0` through
`needle:8192:14`. The same command that wrote the 370 rows skips those
ids and appends each new row before the next one. Generation caps, chat
mode, temperature, seed, and the embedding override stay as they were.
No second model was started while this note was written.

Host check the same evening, before any launch: RTX 5060, 8151 MiB total,
1343 MiB used, 6554 MiB free, GPU utilization 1%, 39 C. Physical RAM
16310 MiB with 4497 MiB free. Commit limit 31670 MiB with 13501 MiB free.
Drive E: 280 GB free. `llama-server.exe` was not running. The runner
refuses a desktop launch when background VRAM is above 2048 MiB; 1343 is
under that line. The recorded Q4 CUDA weight buffer is 1211.05 MiB at
context 8448, not a traced peak of the quality run. Free VRAM of 6554 MiB
covers that buffer with margin. This check is not permission to start.

```powershell
.\.venv\Scripts\python.exe benchmark\run_p9_quality.py --server E:\ExVRAM\.cache\llama\runtime-b10982-cuda13.4\llama-server.exe --model E:\ExVRAM\models\qwen35-2b\Qwen3.5-2B-Q4_K_M.gguf --model-sha256 aaf42c8b7c3cab2bf3d69c355048d4a0ee9973d48f16c731c0520ee914699223 --model-role qwen35_2b_q4_k_m --output experiments/results/p9_qwen35_2b_q4.jsonl --long-output experiments/results/p9_qwen35_2b_q4_long.jsonl --work-dir work/p9-2b-q4 --server-extra=--override-tensor --server-extra=token_embd.weight=CUDA0
```

Do not delete the partial JSONL. Do not add `--allow-contended-gpu`.
The 27B Q4 file stays unloaded.

## Next run, not started

Finish those 25 needles, then stop. If the finished point verdict is
`PASS`, the next quant to prepare is Q3_K_M. If it is `FAIL`, prepare
Q5_K_M. Neither file is fetched or launched here. A point `PASS` would
still leave math below BF16 on the paired interval, so it would not mean
the formats are equivalent. The one IQ2 follow-up, after Q4 and not
instead of it, is a second pass of the stored repetition-class GSM8K ids
at the same temperature, seed, and 768 cap, to see whether the same line
repeats. That does not change sampling and does not name a tensor.

## Answers

1. The 395 Q4 rows are not complete. 370 are measured. 25 needles are absent.
2. Q4 has no six-category aggregate. 35/45 long-context correct would be
   0.627 and would fail the 0.90 floor. 36/45 would be 0.631 and would pass
   that floor on the point estimate.
3. The point gate is not decided. The five finished categories clear their
   own floors. Long context does not have a score yet.
4. On the finished math items, the paired interval lies entirely below zero.
   The other four finished intervals cross zero. That is not a six-category
   call.
5. Recorded CUDA weight buffers, not suite peaks: Q8 1908.35 MiB, Q4
   1211.05 MiB, a difference of 697.3 MiB. File sizes differ by the same
   697.3 MiB. A traced peak for the Q4 quality run was not stored.
6. The lowest completed arm that clears the aggregate point floor is Q8_0
   (0.693, share 0.991). Its coding interval still straddles the 0.85 ratio.
   IQ2 does not clear the floor. Q4 is not a candidate until the tail exists.
7. Do not choose Q3 or Q5 yet. Q3_K_M is the proposal only if the finished
   Q4 point verdict is `PASS`. Q5_K_M is the proposal if it is `FAIL`.
8. 2B IQ2 math loops: 84 of 94 wrong rows hit 768, and the repetition class
   is absent from the wrong BF16, Q8, and Q4 math rows. No tensor is identified.
9. Still required: the 25 Q4 needles, then the paired interval on the full
   arm. A repeat of the IQ2 repetition ids is the loop check after that.
   LONGTASK-V1 stays unscored. Suite peak VRAM and TTFT were not stored.
10. For 27B, the useful next fact is whether 2B Q4 still clears the locked
    floors. A 27B Q4 file does not fit this card, and the 2B loop is not
    what the 27B math rows show. No 27B weight edit is justified yet.

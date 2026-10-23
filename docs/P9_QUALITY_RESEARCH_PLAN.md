# Quality research plan, 2026-10-23

The question is which abilities the quant loses, why, and how far the bit
width can fall before the locked P9 floors fail. No new quantizer and no
CUDA kernel. No GPU job was started while writing this plan.

Study label for every 2B number: `METHOD_COMPARISON_NOT_27B_RETENTION`.

## What the IQ2 gap is

Same protocol, same 395 ids, same prompt SHA-256, same gold, temperature 0,
seed 42, and the locked caps. Tokenizer token-list hash and chat-template
hash match the BF16, Q8, and Q4 headers. Rescoring reasoning, math, factual,
and needle rows with `src/exvram/p9_eval.py` disagreed with the stored bit
on 0 IQ2 rows. Coding and instruction were not re-executed.

IQ2 aggregate 0.298981 against BF16 0.699722. The pair note's bootstrap
interval on that difference lies entirely under the 0.90 floor. Point
verdict `FAIL`, study call `QUALITY_DEGRADED`. That call is about this 2B
pair. It is not a 27B retention rate.

Primary label on the 285 wrong IQ2 rows, one label each, from the saved
text only:

| Label | Rows | Rule used |
|---|---:|---|
| `GENERATION_TRUNCATION` | 230 | `predicted_n` reached the locked cap |
| `REASONING_FAILURE` | 26 | Generation finished, the scorer found a well-formed answer, and it was wrong |
| `UNRESOLVED` | 21 | 19 finished coding misses were not re-run; 2 finished instruction misses have no saved checker kwargs |
| `ANSWER_FORMAT_FAILURE` | 8 | Finished reasoning text with no choice letter the scorer can read |
| `TEMPLATE_OR_CONFIGURATION` | 0 | Prompt hash, gold, or decoding differed from BF16 |
| `EVALUATOR_FAILURE` | 0 | Stored bit disagreed with a rescore |
| `QUANTIZATION_SUSPECTED` | 0 | Not used. No row has tensor-level evidence |

Math is the sharp cell: 6/100 against BF16 81/100, 94 wrong. 84 of those
94 have `predicted_n` 768. 80 of the 94 contain a 12-word sequence at least
four times, and 71 have at least 30% of their non-empty lines repeated
three or more times. `gsm8k:807` restarts the same "two bags of glass chips"
step and stops at exactly 768 tokens. The gold number string is present
somewhere in 8 of the 94 wrong answers. The scorer uses the last number, or
the number after `####`, so those 8 are not scorer misses. The pair note
said the gold number appears in none of the 94. That sentence is too
strong. Ten wrong math answers finished early and ended on a wrong number.
Those ten are `REASONING_FAILURE`. BF16 was right on 76 of the 94.

So the measured IQ2 behavior on math is a loop into the cap, plus a small
set of finished wrong arithmetic. The loop is not, by itself, proof that a
particular tensor is damaged. `--reasoning off` is what the runner passes
and what the pair note records. The JSONL row does not store that flag.

Factual IQ2 is 42/100, and 56 of the 58 misses hit the 16-token cap. BF16
hit that cap zero times. Raising the cap for IQ2 only would be a new
protocol and would have to be run on every arm.

Long context does not show this failure. IQ2 is 43/45. Both misses stopped
on their own (`predicted_n` 6, cap 64) and dropped the last digit of a
6-digit code.

## What still has to be checked before a cause is named

- A repeat of a handful of cap-hit math ids, same flags, to see whether the
  loop is stable. Not started. Needs an explicit go-ahead.
- The original Q4 server command. The partial rows do not record
  `--override-tensor` or graph splits. No suite log was in `work/`.
- Tensor health. The header match does not inspect weight outliers.
- Coding and instruction rescore. Not done here, because that means running
  saved programs and because instruction kwargs are not in the row.

Until those exist, the honest cause is: under this protocol the 2B IQ2
arm fails by filling the cap with repeated steps, not by a detected
template, tokenizer, or scorer mismatch. `QUANTIZATION_SUSPECTED` stays
unused.

## Q4

File `models/qwen35-2b/Qwen3.5-2B-Q4_K_M.gguf`, 1,280,835,840 bytes,
SHA-256 `aaf42c8b7c3cab2bf3d69c355048d4a0ee9973d48f16c731c0520ee914699223`.
Header `file_type` 15, `quantized_by` Unsloth, same imatrix name and the
same 186/80 imatrix counts as Q8 and IQ2. Do not rerun the 370 stored ids.

Closed categories, Q4 correct / BF16 correct on the same ids. The interval
is the Wald 95% interval on (Q4 accuracy − BF16 accuracy). This is not a
protocol verdict. Long context is short 25 ids, and there is no summary row.

| Category | n | Q4 | BF16 | Delta | 95% interval |
|---|---:|---:|---:|---:|---|
| Reasoning | 60 | 29 | 29 | 0.000 | −0.092 to +0.092 |
| Math | 100 | 71 | 81 | −0.100 | −0.176 to −0.024 |
| Coding | 40 | 12 | 13 | −0.025 | −0.110 to +0.060 |
| Factual | 100 | 67 | 72 | −0.050 | −0.108 to +0.008 |
| Instruction | 50 | 41 | 43 | −0.040 | −0.150 to +0.070 |
| Long context | 20 of 45 | 20 | 20 | not a call | 25 ids absent |

Math is the only closed cell whose interval lies entirely below zero.
That is a loss on this category. It is not `QUALITY_DEGRADED` for the
suite, and it is not equivalence on the other rows: those intervals
contain both zero and values past a locked ratio. The long-context file
has 15/15 at 2048 and 5/15 at 4096, all correct, and none of the 8192
needles. Aggregate, suite peak VRAM, and TTFT are `NOT_STORED`.

Q3_K_M and IQ3 are not next. The missing 25 needles come first, with the
same server flags as the 370 rows. The runner already skips ids marked
measured. Do not delete the partial JSONL. Do not add an embedding
override only for the tail. Do not start that resume without an explicit
go-ahead and a resource check: one model, RTX 5060 8 GB, 16 GB RAM, no
second server, no pagefile change.

File sizes, not VRAM peaks: BF16 3,775,709,216 bytes, Q8 2,012,012,800,
Q4 1,280,835,840, IQ2 768,270,592.

Median of stored sample timings, predictions of at least 8 tokens, not a
speed protocol: decode BF16 88.11, Q8 141.19, Q4 186.74, IQ2 217.07 tok/s.
Prefill medians from `prompt_per_second`: 119.20, 185.80, 253.47, 290.48.
`prompt_ms` is prompt time. TTFT was not stored. Cap-hit IQ2 rows inflate
its decode sample.

## Long tasks

[LONGTASK_V1.md](LONGTASK_V1.md). Three tasks, 18, 18, and 16 automatic
checks. The checker does not start a model. `MODEL_RUN` is `NOT_RUN`.
Three tasks can show a failure mode. They cannot accept or reject a quant.

## When a quant method is acceptable

Use protocol `p9-2026-10-02-1` on both arms. Do not change a cap or a
sample size after seeing scores. A category call needs the Wald 95%
interval on the paired difference:

- The whole interval on the passing side of the locked ratio:
  `INSIDE_MARGIN`.
- The whole interval on the failing side: `OUTSIDE_MARGIN`.
- An interval that crosses the ratio: `INCONCLUSIVE`. Not equivalence.
- A point estimate on the ratio is not `ZERO_LOSS` and not
  `QUALITY_PRESERVED_WITHIN_MARGIN`.

Suite `PASS` still requires aggregate at least 0.90 of the reference,
every category at least 0.75 of that category, reasoning at least 0.85,
and coding at least 0.85. LONGTASK-V1 does not replace any of those floors.
A method that needs a new quantizer or a new kernel is not started from
this 2B gap alone.

## What this suggests for 27B, without transferring the score

The 27B IQ2 absolute aggregate is 0.774167, higher than this 2B BF16
aggregate. That is not evidence the 27B quant "kept" ability. Official 27B
BF16 was not loaded. The mechanisms worth a later look, if a higher 27B
pack is ever small enough for this card, are:

- Cap-filling repetition, because that is the 2B IQ2 math pattern.
- A short factual cap interacting with a verbose answer, because 16 tokens
  separated the 2B arms.
- Mixed precision that already exists in the published IQ2 file, including
  the Unsloth imatrix named in the 2B headers. llama.cpp imatrix is the
  existing tool. This machine cannot run it on the 53.8 GB Coletti GGUF.

Q4 and Q8 of the 27B do not fit this 8 GB card. Improving 27B quality
inside that budget is still an open fit question, not a reason to quantize
the large BF16 locally.

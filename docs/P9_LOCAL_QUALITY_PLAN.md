# P9 local quality plan

No new model was loaded and nothing was downloaded for this plan. Scores
below that are already measured point at
[P9_BASE_QUALITY.md](P9_BASE_QUALITY.md) and
[P9_ERROR_ANALYSIS.md](P9_ERROR_ANALYSIS.md). Everything else is a fit
estimate or an upstream file size.

## IQ2 suite still open

The runner skips sample ids already marked measured. Do not regenerate the
305 rows. The 12 rows that filled the locked cap stay, labeled
`OUTPUT_TRUNCATION`. The cap is not raised.

| Next block | Items | Blocker |
|---|---:|---|
| IFEval `1893`, `2010`, `2828`, `2534`, `1580` | 5 | Needs a free window. Free physical RAM at the audit was 5.3 GB; the GGUF is 7.27 GB |
| HumanEval pass@1 | 40 | `python_program_passes` runs the candidate with the user account and a 5-second timeout. That is not a privilege sandbox. Do not start these items until the checker is confined |
| Needles at 2048, 4096, 8192 | 15 each | Long GPU job. The occupied-8192 prefill on this file was 243.8 seconds in P6 |

All three wait for an explicit go-ahead. The needle set is the long one.

## Higher 27B packs, file size versus the card

The measured IQ2_XXS peak was 7767 MiB of 8151. The file is 7,266,070,528
bytes. Sibling sizes are the Unsloth repo listing at revision
`4ca720788d1e01f1bff70c033e0d0028fd02e502`. None of these files were loaded.

| File | Bytes | Fit |
|---|---:|---|
| UD-IQ2_XXS | 7,266,070,528 | MEASURED FULL_GPU |
| UD-IQ2_S | 8,371,970,048 | About 1.03 GiB larger than the file that already peaked at 7767 MiB. Estimated not to fit. Not launched |
| UD-Q2_K_XL | 9,828,981,664 | Larger than the card |
| UD-IQ3_XXS | 10,934,860,704 | Larger than RAM headroom and the card |
| UD-IQ3_S | 12,040,883,104 | No |
| Q4_0 | 16,056,478,688 | Larger than physical RAM |
| UD-Q4_K_M | 16,464,440,224 | No |

Q3, Q4, and Q8 of this 27B are not local references. A launch would page or
fail the GPU allocation.

ExLlamaV3 `turboderp/Qwen3.8-27B-exl3` branch `SC_1.80bpw_H3` has a decoder
plus `lm_head` of 6,016,269,892 bytes in the header inventory. Embeddings on
that branch are still BF16, 2,542,796,800 bytes, and must stay off the GPU
or the card overflows. That pack is not downloaded. It would be a second
low-bit runtime, not a BF16 reference, and only after a resource check.

## Smaller model, same architecture family

`Qwen/Qwen3.5-2B` revision `15852e8c16360a2fea060d615a32b45270f8a8fc`,
Apache-2.0, one safetensors file of 4,548,221,488 bytes (4.24 GiB). It uses
the same 3:1 Gated DeltaNet hybrid as the 27B. It is a different network.
A paired BF16-versus-IQ2 run on it can say something about the quantizer.
It cannot be quoted as the 27B retention rate.

`Qwen/Qwen3.5-4B` revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` is
9,319,828,096 bytes. That is larger than the 8151 MiB card, so its BF16
does not fit.

The 2B BF16 file is smaller than the card and smaller than physical RAM.
At the audit only 5.3 GB of RAM was free, so it is not safe to load it in
the current session either. An IQ2 GGUF of that same revision is not on
disk. Both downloads wait for confirmation.

## Raising 27B quality inside 8 GB

The current IQ2 file is already mixed precision. The local header parse
shows `iq2_xxs` together with a large `iq1_s` payload, `q2_k` embeddings,
and a `q3_k` output projection. Unsloth publishes
`imatrix_unsloth.gguf` at 13,642,656 bytes for that repo. Re-running
`llama-quantize` or `llama-imatrix` on the 53.8 GB BF16 is the step this
host cannot do.

What can still be compared without a new quantizer:

- The existing IQ2 file, already measured in part.
- An already-built EXL3 pack whose decoder fits, if the resource check
  passes. That is the practical test of "give the sensitive tensors more
  bits inside the same card."
- The 2B BF16/IQ2 pair, as a method check with a different model.

No CUDA kernel. No local requantize of the 27B BF16.

## What needs a yes before it starts

1. Load the existing IQ2 server for the five IFEval items, then HumanEval
   only after the checker is confined, then the needles.
2. Download and run `Qwen3.5-2B` BF16 and a matching IQ2, as a separate
   study.
3. Download the EXL3 1.80 bpw pack and run the same items, if the fit check
   still clears.

Official 27B BF16 stays off. Cloud stays off.

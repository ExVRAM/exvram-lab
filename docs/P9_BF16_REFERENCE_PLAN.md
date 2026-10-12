# P9 BF16 reference plan

Status: **NOT_RUN**. Cloud rental in this file is withdrawn. On 2026-10-12
the operator limited every ExVRAM run to this PC. See
[P9_BF16_LOCAL_FEASIBILITY.md](P9_BF16_LOCAL_FEASIBILITY.md). Do not rent a
GPU and do not call a remote endpoint.

This file is a launch plan. It does not add scores.
The comparison gate stays `REFERENCE_UNAVAILABLE`. The pass ratios in
[P9_EVAL_PROTOCOL.md](P9_EVAL_PROTOCOL.md) are unchanged. No cloud machine
was rented and the official weights were not downloaded again.

## Identity

| Field | Value | Status |
|---|---|---|
| Repository | `Qwen/Qwen3.8-27B` | The Hugging Face API on 2026-10-12 returned this as the model id |
| Revision | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | API `sha` and `lastModified` `2026-08-14T15:00:01Z`. This is the revision already pinned in `experiments/manifests/qwen38_p2.json` |
| License | Apache-2.0 | Card metadata |
| Weights | 18 safetensors shards, BF16 tensor bytes 55,562,855,904 | Local inventory from the published index, not a new download |
| Tokenizer | `tokenizer.json`, `tokenizer_config.json`, `vocab.json` at that revision | Not re-hashed this round |
| Chat template | `chat_template.jinja` at that revision | Official file. The IQ2 run used the template inside the GGUF via llama-server |
| Official sampling defaults | `generation_config.json`: `do_sample` true, temperature 1.0, top_p 0.95, top_k 20 | These are not the eval settings |
| Eval settings | Protocol `p9-2026-10-02-1`: temperature 0.0, seed 42, no added system prompt, `--reasoning off`, caps 512 / 768 / 512 / 16 / 768 / 64 | Locked |

The IQ2 file is `unsloth/Qwen3.8-27B-GGUF` revision
`4ca720788d1e01f1bff70c033e0d0028fd02e502`, file
`Qwen3.8-27B-UD-IQ2_XXS.gguf`. The Unsloth card names `Qwen/Qwen3.8-27B` as
the base model and does not record the base commit. Unsloth also publishes
`BF16/Qwen3.8-27B-BF16-00001-of-00002.gguf` (49,986,159,616 bytes, LFS
`b9966e82b7a4d87028b5eae061d578ee826305ebf8baea5bfc6e09bad0ba191f`) and
`BF16/Qwen3.8-27B-BF16-00002-of-00002.gguf` (4,671,576,000 bytes, LFS
`92e3943c4f9bd6292a7bef82369f65fed9bfed088b9df0fb2fa2ce17c9edfa02`). That
pair is a third-party GGUF, 54,657,735,616 bytes together. It is not the
official safetensors and is not the reference.

## Where it can run

| Option | Fit | Cost | Reproducible | Decision |
|---|---|---|---|---|
| A. This PC, 16 GB RAM, RTX 5060 8 GB | 55.6 GB weights do not fit | Free | Would page the desktop | Do not launch |
| B. Another machine already available | None is recorded in this repo | — | — | NOT_RUN |
| C. One rented 80 GB GPU | A100 80 GB can hold the 51.8 GiB text weights with headroom for the 8k KV of this hybrid model. Confirm the offer is 80 GB, not 40 GB | ESTIMATED, below | Same llama-server flags after a conversion, or the official loader with reasoning disabled | Wait for confirmation. Do not rent from this session |
| D. CPU inference | Needs the weights resident. This host has 16 GB | Free and unusable here | A swap run is not a reference | Do not launch |
| E. Hosted endpoint | Qwen's card points at a hosted service that was marked coming soon, with extra production features | Unknown | The endpoint does not expose this revision and these llama.cpp flags | Not a reference |

RunPod's public pricing page, fetched 2026-10-12, lists on-demand A100 PCIe
at $1.19/hr Community and $1.39/hr Secure, A100 SXM at $1.39/hr and $1.59/hr,
H100 80 GB from $1.99/hr, and H200 at $3.59/hr Community and $4.59/hr Secure.
A third-party snapshot from 2026-09-28 put Vast.ai A100 80 GB offers near
$0.47–$0.74/hr. Recheck the live offer before paying. Storage for a few
hours is small next to the GPU hour.

Time estimate, not a measurement: the 305 IQ2 calls took 36.8 minutes of
server prompt plus decode time. The missing short IQ2 items are smaller than
that. The 45 needles are the long part on this 5060; the occupied-8192 P6
prefill was 243.8 seconds, so 15 prompts at 8192 are on the order of an
hour before the 4k and 2k sets. A BF16 pass of the same 395 items on an
80 GB GPU is estimated at 3–6 hours including the 55.6 GB download and one
retry. At the RunPod A100 PCIe Community rate that is about $4–$8, and a
safe budget is $15. If the 80 GB card runs out of memory once the runtime
allocates activations, the next card is an H200 and the budget should be
$30. Nothing above this has been spent.

## Launch steps, after an explicit yes

1. Rent one GPU with at least 80 GB VRAM and at least 128 GB system RAM.
   Cap the spend at the budget above. Do not keep it for anything else.
2. Download only `Qwen/Qwen3.8-27B` revision
   `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. Record shard SHA-256 values.
3. Run llama.cpp `b10982` commit `fc82583e6` with the locked server flags,
   temperature 0, seed 42, and `--reasoning off`. A GGUF conversion is
   acceptable only if the log names that converter commit and the source
   revision. Do not label the Unsloth BF16 GGUF as this reference.
4. Score the same sample ids the IQ2 file already has, plus the missing
   IQ2 ids once those exist. Keep raw answers in a new JSONL. Do not
   append them to `p9_base_quality.jsonl`.
5. Stop the instance when the file is copied out.

## IQ2 work left on this PC

Do not start it until there is a free window and free RAM is comfortably
above the 7.3 GB file. On 2026-10-12 at the audit, the GPU was idle
(about 1246 MiB used, 38 C) and free RAM was about 5.4 GB of 16.3 GB.
That is the same reason the previous continuation stayed stopped.

The runner already skips completed sample ids.

| Work | Count | Why |
|---|---:|---|
| Missing IFEval | 5 | `ifeval:1893`, `ifeval:2010`, `ifeval:2828`, `ifeval:2534`, `ifeval:1580` |
| HumanEval | 40 | Not started |
| Needles at 2048, 4096, 8192 | 15 each | Not started |
| Rows that stopped under the old cap | 96 | Keep. Protocol says a sample that stopped early stays |
| Rows that filled the locked cap | 12 | Keep as `OUTPUT_TRUNCATION`. The same cap must be used for BF16. A higher cap would be a new protocol for both arms |

Paired intervals are NOT_RUN until the BF16 file exists. Absolute IQ2
scores are not a retention percentage.

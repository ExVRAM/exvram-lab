# Uncensored scorecard

Checked 2026-09-24. Evidence labels:

- `MEASURED` — an ExVRAM run on the RTX 5060 recorded in this repository.
- `UPSTREAM` — a figure copied from the publisher's card or README. Not re-run here.
- `DERIVED` — arithmetic from file bytes and the 8151 MiB driver report. Not a runtime fit.
- `NOT_RUN` — no number.

No row is a measured Pareto point for the uncensored axis. The original llama.cpp IQ2_XXS
decode in [PERFORMANCE_PARETO.md](PERFORMANCE_PARETO.md) is a different checkpoint. It must
not be copied onto an abliterated file.

`REFUSAL_RATE` here, when it comes from Coletti or hotdogs, is refusals on harmful prompts.
It is not `OVERREFUSAL_RATE`. Over-refusal was not published on the inspected cards and has
not been scored in ExVRAM. The local scorer is `exvram score-refusal`. It reads a JSONL of
responses. It does not download a dataset and it does not run a model.

| Model | Method | Quant | Size | VRAM | Decode | 8k | Refusal rate | Quality delta | Evidence |
|---|---|---|---:|---|---|---|---|---|---|
| `Qwen/Qwen3.8-27B` | none | BF16 index | 55,562,855,904 | not resident | NOT_RUN | NOT_RUN | 98/100 UPSTREAM, harmful set | baseline | UPSTREAM refusal; size from the safetensors index |
| bartowski IQ2_XXS | none | IQ2_XXS | 8,881,268,448 | partial GPU, see performance Pareto | 3.8 and 5.5 tok/s on short contexts | NOT_RUN | NOT_RUN | NOT_RUN | MEASURED speed and residency only |
| Coletti Uncensored | Heretic, BF16 | BF16 | 55,562,857,544 index | not resident | NOT_RUN | NOT_RUN | 12/100 UPSTREAM | mean −0.5 on four 0-shot tasks; wikitext-2 PPL +0.0434 vs base BF16 | UPSTREAM |
| hotdogs abliterated | LLM-abliterate, λ=1.2 | BF16 | 55,562,855,904 index | not resident | NOT_RUN | NOT_RUN | 39/100 UPSTREAM | MMLU −0.005, GSM8K strict −0.03, ARC +0.01, limited subsets; KL 0.0001 | UPSTREAM, protocol differs from Coletti |
| Huihui abliterated | remove-refusals, layers 18–51 | BF16 | 55,562,855,904 index | not resident | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | Tensor names match base. Behaviour not scored |
| Coletti IQ2_M fused | Heretic, then imatrix GGUF | IQ2_M | 10,624,771,968 | DERIVED: file alone exceeds 8151 MiB | NOT_RUN | NOT_RUN | inherited from BF16 card, not re-measured at this quant | wikitext-2 PPL 7.8581 vs this model's own f16 7.1557 UPSTREAM | UPSTREAM PPL; DERIVED fit |
| Huihui UD-IQ2_S | abliteration on an Unsloth UD quant | UD-IQ2_S | 8,424,398,848 | DERIVED: 116 MiB under 8151 MiB before KV | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | DERIVED unfit for FULL_GPU once runtime overhead is included |
| Huihui Ternary PTQ1_0 | abliteration on a ternary GGUF | PTQ1_0 | 6,608,603,616 | DERIVED: 1.80 GiB under 8151 MiB before KV | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | DERIVED size only. Runtime is the PrismML fork, not measured |
| HauhauCS Aggressive IQ2_M | method not stated on the card | IQ2_M | 10,319,906,944 | DERIVED: file exceeds 8151 MiB | NOT_RUN | NOT_RUN | 0/465 UPSTREAM, protocol not stated, not comparable | NOT_RUN | UPSTREAM claim only |

`refusal reduction / capability loss` is not computed across these rows. The capability
losses use different tasks, different sample limits, and no ExVRAM run. Coletti's own
reading of the −0.5 mean is that it sits inside evaluation noise. That remains their
statement. `exvram` can compute the ratio later with
`refusal_capability_tradeoff` once both inputs are measured on one protocol. A zero
capability loss leaves the ratio undefined instead of infinite.

## What matching quants would be allowed to mean

A comparison counts only at the same quant family and the same runtime settings:

- Original bartowski or Unsloth IQ2 versus a Coletti IQ2 built from the Heretic BF16.
- Original EXL3 1.8–2.2 bpw text stack versus an EXL3 conversion of the same Heretic BF16.
- BF16 versus BF16 for the refusal and KL numbers.

Coletti IQ2_M versus official BF16 is a quantization gap and an abliteration gap at once.
Huihui ternary versus bartowski IQ2_XXS is also not an abliteration comparison.

## Decision gate

**REQUANTIZATION_REQUIRED**

Ready minimal-refusal BF16 checkpoints exist. The Coletti Heretic checkpoint is the one to
quantize. None of the inspected compact files is a measured RTX 5060 FULL_GPU, 8192-context
artifact on a runtime this repository already runs.

- `OSS_UNCENSORED_SUCCESS` is not met. No uncensored file has a measured 8k FULL_GPU fit or
  a measured decode.
- `ABLITERATION_REQUIRED` is not the next step. Heretic has already been applied, and the
  published front includes lower-KL points if 12/100 at KL 0.1191 is too costly. Call
  `plan-heretic`; do not reimplement the search.
- `FINE_TUNE_REQUIRED` is not met. Nothing measured here shows that abliteration both keeps
  refusal high and damages capability. Upstream hotdogs still refuses 39/100 with near-zero
  KL, which is a different point on the same kind of front, not a fine-tune gap.

The requantization path, when the FULL_GPU performance track is ready for a matching file:

1. Start from `JonathanColetti/Qwen3.8-27B-Uncensored` BF16, not from IQ2_M and not from a
   Huihui ternary or UD file.
2. Keep the tensor inventory check: 1,199 names, 15 `mtp.*`, vision tensors, `lm_head`.
3. Quantize with stock llama.cpp or Unsloth imatrix tools, or with ExLlamaV3's converter.
   The Coletti repo already publishes an imatrix beside the GGUF files.
4. Score refusal and over-refusal with `score-refusal` on a fixed external corpus, at the
   same quant as the original checkpoint.
5. Only if that matched comparison shows too much capability loss, or refusal that is still
   too high, re-export another Heretic front point. Fine-tuning stays after that.

Application policy is still outside this gate. A lower refusal rate in the weights is not a
decision to remove a separate, configurable application policy.

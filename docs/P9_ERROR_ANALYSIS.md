# P9 error analysis

Source: the 305 `base_iq2_xxs` samples in
`experiments/results/p9_base_quality.jsonl`. Counts are of rows the scorer
marked wrong. There is no BF16 arm, so "more often than the reference" is
NOT_RUN for every class.

| Class | Count | Notes |
|---|---:|---|
| Reasoning, wrong letter | 15 | A `(A)`–`(D)` answer is present and it is not the gold letter |
| Reasoning, no letter | 9 | Prose answer. Example gold `(A)`, prediction `The blue jay is the rightmost.` |
| Reasoning, gold letter present but not scored | 2 | The letter appears, and the locked letter rule still marked the row wrong |
| Reasoning, empty | 1 | Empty prediction |
| Reasoning, cut off at the locked cap of 512 | 2 | `OUTPUT_TRUNCATION`. Both wrong. The cap is already the locked value |
| Math, wrong final number, generation finished | 4 | `MODEL_FAILURE` on this arm. `gsm8k:700` stopped at 244 of 256. `gsm8k:590`, `gsm8k:990`, and `gsm8k:125` stopped before 768 |
| Math, cut off at the locked cap of 768 | 7 | `OUTPUT_TRUNCATION`. All 7 are wrong. The trailing number is not a finished answer |
| Factual, wrong A/B | 20 | Choice present, opposite of gold. None hit the 16-token cap |
| Instruction check failed | 6 | 39 of 45 passed. Five of the locked 50 were not generated |
| Coding | 0 | NOT_RUN |
| Long-context fact loss | 0 | NOT_RUN |
| Repetition scored as its own class | 0 | Not labeled in this file |

The 15 wrong-letter rows are the largest closed error class. They are not
format failures. The 9 no-letter rows are instruction-format failures on a
task whose score requires a letter. One of those is the blue-jay sentence
above (`logical_deduction_three_objects:117`). A wrong-letter example is
`logical_deduction_three_objects:13`: gold `(A)`, prediction `The answer is (C).`

Finished math misses are arithmetic or setup errors. `gsm8k:590` gold is
319 and the last number in the prediction is 318. `gsm8k:700` gold is 135
and the last number is 35. The seven rows that filled 768 tokens are
`OUTPUT_TRUNCATION`, including `gsm8k:1070`. Their last number is taken
from a cut answer.

Factual errors are swapped choices, for example `truthfulqa:e854cb7fcd9338e6`
gold `B`, prediction `A`. That is a knowledge or selection error under this
A/B format, not a refusal.

Instruction failures include a checker the output did not meet. One rap
answer (`ifeval:2811`, `keywords:forbidden_words`) was marked wrong. The
file does not say which forbidden word fired. One poem (`ifeval:2398`,
`startend:end_checker`) did not end as required.

No category here shows a rate versus BF16. The weak absolute cells are
tracking shuffled objects (9/20) and date understanding (10/20). Those
intervals are wide. They are not evidence that quantization caused the miss.

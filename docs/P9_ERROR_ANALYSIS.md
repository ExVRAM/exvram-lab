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
| Reasoning, cut off at `n_predict` 64 | 2 | Both wrong. Protocol says these are rerun at 512. They were not |
| Math, wrong final number, generation finished | 4 | Last number differs from the `####` gold. Not a cap hit |
| Math, cut off at `n_predict` 256 | 7 | All 7 are wrong. Protocol says these are rerun at 768. They were not |
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

Math misses that finished are arithmetic or setup errors, not empty answers.
`gsm8k:590` gold is 319 and the last number in the prediction is 318.
`gsm8k:700` gold is 135 and the last number is 35. The seven truncated math
rows cannot be called arithmetic failures until they are regenerated at 768
tokens. `gsm8k:1070` is one of the finished misses: gold 18, last number 1.5.

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

# LONGTASK-V1

Protocol id: `longtask-v1`.

Status: **checker only**. `MODEL_RUN` is `NOT_RUN`. No llama-server job
belongs to this protocol yet. Three tasks are a diagnosis. They are not a
sample size for a quality verdict, and they do not replace
`p9-2026-10-02-1`.

The checker is `src/exvram/longtask_v1.py`. It scores saved text and, for
the ledger task only, runs the candidate modules under `python -I` in a
temporary directory for at most 5 seconds. The same denylist as
`python_program_passes` refuses `os`, `subprocess`, `socket`, `open`,
`eval`, and `exec` before the process starts. A timeout or a refused
source fails every ledger check. Generated code is not given a broader
account than that.

Context lengths, when a run is later authorized: 2048, 4096, and 8192
tokens, filled with a neutral pad that contains none of the answer keys.
The pad is counted by the same llama-server tokenizer as P9. A row that
hits the generation cap, or whose prompt does not fit the 8448 context,
is `CONTEXT_OR_GENERATION_LIMIT`. It is not scored as a wrong requirement.
Arms, if run, are the local Qwen3.5-2B BF16, Q8_0, Q4_K_M, and UD-IQ2_XXS,
one model at a time, same temperature 0 and seed 42. Results on 2B are not
27B results.

## Answer format

Document and route answers are lines of `KEY: value`. Extra prose is
ignored. A missing key fails that check only. The ledger answer is three
Python files: `ledger/money.py`, `ledger/book.py`, `ledger/report.py`.

## 1. Ledger

`parse_cents` reads a money string as cents. `Book.post` adds integer cents
to an account, `balance` returns 0 for an unknown account, and `total` sums
the accounts. Two instances do not share state. Account names are
case-sensitive. `format_line` renders `account` plus a signed decimal with
two digits.

Eighteen checks: four parses (`12.30`, `$1,000.00`, `-2.50`, `0.05`),
accumulate, unknown account, two-account total, negative post, four
formats including zero, separate instances, empty total, case-sensitive
names, a post that returns to zero, and the presence of all three files.

## 2. North Hall permit

The source text is `DOCUMENT` in the checker. The original capacity 120 is
amended to 96 because the east exit is closed. Fee is 40 per person at the
amended capacity (3840), due 2026-11-02. Payment on or before 2026-10-28
discounts 10 percent (3456). Curfew is 22:00, with a rehearsal on
2026-11-01 until 23:30. Vendor is Lumen Audio, catering is not allowed,
deposit 500 stays refundable, contact is `ada@north.example`, load-in is
16:00, the plot has 4 speakers and 2 returns, and cancellation before
2026-10-30 is the cutoff.

Eighteen keys: `PERMIT`, `CAPACITY`, `FEE`, `EARLY_FEE`, `DUE`,
`EARLY_PAY`, `CURFEW`, `REHEARSAL_UNTIL`, `REHEARSAL_DATE`, `VENDOR`,
`CATERING`, `DEPOSIT`, `DEPOSIT_REFUNDABLE`, `CONTACT`, `LOAD_IN`,
`SPEAKERS`, `RETURNS`, `CANCEL_BEFORE`. Using 120 as the capacity fails
`CAPACITY` and is the check that the amendment was kept.

## 3. Route amendment

Start from orders A=10, B=7, C=4 and a truck of 12 under largest-first.
The amendment cancels C, adds 5 to B, sets the truck to 15, and replaces
the rule with smallest-first. A is 10, B is 12, C is 0. A fits in 15 and
B does not fit in the remaining 5, so trip 1 is A and trip 2 is B.
Leftover is 0.

Sixteen keys: `CANCELLED`, `A_QTY`, `B_QTY`, `C_QTY`, `TRUCK`, `RULE`,
`TRIP1`, `TRIP2`, `TRIP_COUNT`, `LEFTOVER`, `C_SHIPPED`, `OLD_TRUCK_USED`,
`LARGEST_FIRST_USED`, `UNLOADED_A`, `UNLOADED_B`, `FINAL`. The last three
negative keys are the values `no`. Shipping C, keeping truck 12, or keeping
largest-first fails those checks.

## What a later run must record

For each arm, task, and context length: the check map, passed/n, whether
the row is `CONTEXT_OR_GENERATION_LIMIT`, the model SHA-256, and the server
flags. Do not collapse the three tasks into one accuracy and compare it
with a P9 ratio. A quant that passes P9 and fails the amendment keys has a
diagnosed failure. A quant that fails P9 is not rescued by these three
tasks.

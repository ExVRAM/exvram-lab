# P9 final

Verdict: **`P9_INCONCLUSIVE`**.

P10 does not start. The missing evidence is the whole comparison, not one
optional cell:

- base quality has not been generated; the reference slot is `UNAVAILABLE`
- Coletti IQ2_XXS does not exist
- refusal and over-refusal were not measured
- occupied-8192 on a derived quant was not run

`P9_SUCCESS` is not available from the P6 speed result. That result remains
a speed measurement only.

| Report field | Value |
|---|---|
| Base reference | `UNAVAILABLE` |
| Base IQ2 deltas | not measured |
| Coletti IQ2 bytes | not produced |
| Coletti physical bpw | not produced |
| FULL_GPU on Coletti IQ2 | not run |
| Occupied 8k on Coletti IQ2 | not run |
| Refusal rates | not measured |
| Another quant or source required | yes, after a finished imatrix on a host that can hold the BF16 |
| P10 | no |

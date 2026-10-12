# P9 Ternary-Bonsai comparison

Quality versus UD-IQ2_XXS: **NOT_RUN**. Speed, VRAM, and long context on
this RTX 5060: **NOT_RUN**. Bonsai was not downloaded and was not loaded.

Bonsai is not a quantization of official Qwen3.8-27B BF16. The card lineage
already recorded in [UNCENSORED_OSS_MATRIX.md](UNCENSORED_OSS_MATRIX.md) is
a ternary model that needs the PrismML llama.cpp fork. Stock llama.cpp
b10982 does not load those files. Treating Bonsai as arm B with different
bits would mix a weight change into a quantization comparison.

Upstream file sizes already inventoried, not re-downloaded:

| File | Bytes | Runtime | Local quality |
|---|---:|---|---|
| Huihui Ternary-Bonsai PTQ1_0 | 6,608,603,616 | PrismML-Eng/llama.cpp `842b1880415d6f508f03b789e5ce70194def7bfd` | NOT_RUN |
| Huihui Ternary-Bonsai PQ2_0 | 7,704,693,216 | same fork | NOT_RUN |

The PrismML card license recorded for that fork is MIT. The ternary method
is upstream work. Nothing from it was copied into this tree. A 4-bit KV
setting on Bonsai is the same class of cache as `--cache-type-k q4_0` on
the IQ2 run. It is not a quality result.

What can be reused later, if a run is justified: the published GGUF and the
fork the card names, under those licenses, as a separate arm. There is no
measured reason yet to reimplement the ternary kernels. There is also no
measured reason to say Bonsai is better or worse than IQ2 on reasoning,
math, coding, knowledge, instructions, or long context.

SM120 support, resident VRAM, decode rate, and needle accuracy for Bonsai
stay NOT_RUN until that fork is actually launched here, one model at a time,
with the same items as arm B.

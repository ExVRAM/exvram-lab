# GGUF Ultra-Low Results

## Artifact provenance

The first artifact was downloaded from the official Hugging Face repository
`unsloth/Qwen3.8-27B-GGUF`, revision
`4ca720788d1e01f1bff70c033e0d0028fd02e502`.

| File | Size | SHA256 | Metadata license | Status |
| --- | ---: | --- | --- | --- |
| `Qwen3.8-27B-UD-IQ2_XXS.gguf` | 7,266,070,528 bytes | `e792d8fb3142fe6d9171876d6da0f71f05a71028718debc72dbec93ff645e67d` | Apache-2.0 | downloaded and verified |
| `Qwen3.8-27B-UD-IQ1_M.gguf` | 6,729,166,848 bytes | `1b5165...` | Apache-2.0 | not downloaded |
| `Qwen3.8-27B-UD-IQ1_S.gguf` | 6,192,222,208 bytes | `3895b6...` | Apache-2.0 | not downloaded |

The abbreviated IQ1 hashes are intentionally not used as verification values here;
the complete metadata is in the acquisition notes and must be rechecked before any
future download. IQ1 artifacts were not needed for Result A.

## Measured IQ2_XXS result

The IQ2_XXS artifact ran with llama.cpp b10982 on an RTX 5060 8 GB. All 65/65 layers
were offloaded to CUDA0. The c128 request was clamped by the runtime to an actual
256-token context. That primary run measured 30.20 prompt tok/s and 31.39 decode
tok/s for 234 generated tokens. A separate ctx512 control measured 18.11 decode
tok/s for 256 generated tokens.

The c128 memory breakdown was:

```text
CUDA0 total 8150 MiB = free 58 + self 6681 (model 6521 + context 154 + compute 6) + unaccounted 1410
Host             398 MiB = model 397 + context 0 + compute 0
```

The host number is the memory-mapped GGUF view, not a CPU decoder offload. Quality
evaluation and 8k context have not been run, so no quality-loss conclusion is made.

## Next GGUF work

P1 should compare the same layer path against the second mature backend before any
custom kernel is considered. P2 then runs a quality suite and a context ladder. IQ1_M
and IQ1_S remain fallback artifacts only if the verified IQ2 path fails at the next
context or quality gate.

# P8 HQQ + GemLite layer results

## Status

The P8 integration boundary and experiment matrix are present, but this report
contains no measured CUDA result yet. The current machine/session did not have a
safe idle VRAM slot for another GPU run, and the default CPU environment does
not contain HQQ. Starting a heavy 27B or kernel run under those conditions would
produce unsafe or non-reproducible evidence.

## Pinned upstreams

| Component | Revision | License | Role |
|---|---|---|---|
| HQQ | `d88a488ec8aa2d58362ef2038a52bca862db2e74` | Apache-2.0 | external 1/2-bit quantization boundary |
| GemLite | `89d9bc705c5dfca9115d3a5620f97a17ba0111a7` | Apache-2.0 | external low-bit kernel candidate |

The source is not copied into ExVRAM. The thin HQQ adapter only imports the
upstream API when the optional dependency is installed. GemLite remains an
optional adapter/integration note.

## Required measurements

For every q/k/v/o, MLP, and supported `lm_head` shape, record:

- HQQ 1-bit and 2-bit at groups 8/16/32/64/128;
- GemLite 1-bit and 2-bit at groups 32/64/128;
- reconstructed tensor quality metrics against the same reference tensor;
- M=1, 8, 32, 128 median/p95 latency, effective GB/s, workspace, VRAM, and
  GPU utilization;
- exact external package/commit, CUDA/Triton/PyTorch versions, warm-up count,
  repeat count, and GPU memory before the run.

The result schema intentionally leaves quality and performance as `NOT_RUN`
until those observations exist. The planner can be checked with:

```powershell
.venv\\Scripts\\exvram.exe plan-p8-binary
```

No P8 result should be promoted into the production recipe from this document.

# Decode profile

No Nsight Systems or Nsight Compute capture exists. Neither `nsys` nor `ncu` is installed, and
`nvcc` is absent, so a kernel timeline was not collected.

What the driver counters did show, during the 2026-09-24 llama.cpp IQ2_XXS rerun, is in
[PERFORMANCE_BASELINE.md](PERFORMANCE_BASELINE.md): framebuffer peaked at 7191 MiB while SM
utilization averaged under 7% in the high-framebuffer samples. That places the stall outside
a single hot GPU kernel. A top-10 kernel table would be invented if it were filled in here.

Revisit this file only after:

1. a full-GPU quant fit is measured, and
2. Nsight Systems records one decode interval of that run.

# P9 local-only strategy

Operator rule, 2026-10-12: every ExVRAM experiment runs on this PC. RTX 5060
8 GB, 16 GB RAM, Windows. Cloud GPUs, rented servers, remote inference APIs,
and paid benchmark services are out. The comparison gate stays
`REFERENCE_UNAVAILABLE`. Protocol `p9-2026-10-02-1` and the 305 IQ2 rows stay
as they are.

This page is the order of work. Measurements live in the other P9 docs.

## What is already measured

`Qwen3.8-27B-UD-IQ2_XXS.gguf` on llama.cpp b10982, FULL_GPU, 305 scored
items. Absolute scores are in [P9_BASE_QUALITY.md](P9_BASE_QUALITY.md). They
are not a retention rate against BF16.

Host snapshot at the audit, not a model launch: GPU idle, 1242 MiB used of
8151, 35 C, 11.8 W. Physical RAM free about 5.3 GB of 16.3 GB. Windows
commit free about 14.7 GB of a 31.7 GB limit. That commit figure includes
the pagefile. A fit decision uses free physical RAM, not the commit limit.

## Order

1. Local BF16 feasibility, written in
   [P9_BF16_LOCAL_FEASIBILITY.md](P9_BF16_LOCAL_FEASIBILITY.md). No BF16
   process was started.
2. Finish the IQ2 suite only after an explicit go-ahead and a fresh resource
   check. Five IFEval ids, then HumanEval, then the needles. The runner
   already skips completed sample ids.
3. A paired BF16-versus-IQ2 run is possible only on a smaller model that
   actually fits. That result is not a 27B result. The candidate is in
   [P9_LOCAL_QUALITY_PLAN.md](P9_LOCAL_QUALITY_PLAN.md).
4. A higher-quality 27B pack inside 8 GB is a separate, still unrun
   experiment. It needs the same confirmation. No new quantizer and no CUDA
   kernel.

## Resource gate

Before any llama-server start: free physical RAM, free VRAM, GPU and CPU
utilization, and an estimate of the file plus the known runtime overhead.
Refuse when the estimate would force Windows to page. Do not change the
pagefile, do not close other programs, and do not load two heavy models.
Long jobs wait for an explicit window. The existing runner resumes from the
last stored sample id.

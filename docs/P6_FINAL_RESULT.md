# P6 final result

## Technical conclusion

Qwen3.8-27B UD-IQ2_XXS on an RTX 5060 8 GB stays FULL_GPU at an actual 8192-token prompt: 65/65 CUDA layers, CPU decoder weight offload 0, q4_0 KV. Five measured runs on 27 Sep 2026 give a median decode of **30.10 tok/s** (CV 2.0%). That meets the speed bar of >=25 tok/s. The quality gate was not run, so the recipe is not marked `MEASURED_REPRODUCIBLE`. The minimal-refusal target was not measured.

## Report

1. The old 29.04 tok/s figure was a real single decode, not a filled-8k result. P6's repeated occupied-8192 median is 30.10 tok/s.
2. Actual occupancy of that old run: prompt 5240, generated 256, final occupancy about 5496 inside a configured window of 8192.
3. Stable short-prompt c256 median: 30.77 tok/s. CV 8.6% because one of the five runs was 25.07.
4. Short-prompt c512 median: 30.66 tok/s (CV 0.25%).
5. Short-prompt c1024 median: 30.76 tok/s (CV 0.28%).
6. Short-prompt c2048 median: 30.76 tok/s (CV 0.20%). Occupied 2048 median: 30.11 tok/s (CV 1.5%, five runs, prompt 2048).
7. Short-prompt c4096 median: 30.61 tok/s (CV 0.53%). Occupied 4096 median: 31.77 tok/s (CV 4.6%, five runs, prompt 4096).
8. Occupied c8192: prompt 8192, generated 256, configured window 8448, median decode **30.10 tok/s**, five runs, CV 2.0%.
9. Best KV at occupied 8192: q4_0/q4_0. Median 30.10 tok/s, KV 148.5 MiB. q8_0/q8_0 also fits at 65/65, median 20.31 tok/s, KV 280.5 MiB, CV above 5% because one run dropped to 4.72. f16/f16 was still running when this note was written. FASTEST_KV, LOWEST_MEMORY_KV, and BEST_BALANCED_KV are q4_0.
10. Peak VRAM on the 27 Sep occupied-8192 series: 7767 MiB (run peaks 7565–7767).
11. Stable base 8k median: 30.10 tok/s on q4_0 KV, UD-IQ2_XXS, b10982.
12. Base quality: NOT_RUN.
13. Uncensored IQ2_XXS artifact: not produced. A local text BF16 GGUF of the Coletti revision exists (53,808,282,560 bytes). Imatrix output is a partial `.dat` only.
14. Uncensored FULL_GPU: not run.
15. Uncensored occupied-8k median: not run.
16. Quality delta: not measured.
17. Harmful refusal: not measured.
18. Benign over-refusal: not measured.
19. Speed target 27B / 8 GB / actual occupied ~8192 / median >=25 tok/s: met for UD-IQ2_XXS q4_0 KV. The full base target also requires the quality gate, which is not done.
20. The same target for a minimal-refusal checkpoint: not met.
21. Next bottleneck: finish q8_0 and f16 KV at the same occupied 8192 prompt, then the base quality suite, then requantize the Coletti BF16 with the existing llama.cpp imatrix path. No custom kernel.
22. New git commit: none for this measurement. Published HEAD remains `d85065d6d2c82f0f6629d93b1c2e9bdc90e5d8ec`.
23. GitHub Actions: not re-run. The last known green run is the one on that commit.

No custom CUDA/Triton kernel and no speculative decoding were used.

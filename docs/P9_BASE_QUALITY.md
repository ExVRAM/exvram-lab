# P9 base quality

Status: **NOT RUN**. Protocol `p9-2026-10-02-1` is locked. No base IQ2
generation has been scored under it.

Reference kind: `UNAVAILABLE`. See [P9_EVAL_PROTOCOL.md](P9_EVAL_PROTOCOL.md).
There is no `REFERENCE_BF16` and no `REFERENCE_PROXY`. The published model is
`Qwen3.8-27B-UD-IQ2_XXS.gguf`, SHA-256
`e792d8fb3142fe6d9171876d6da0f71f05a71028718debc72dbec93ff645e67d`,
7,266,070,528 bytes.

Verdict: `BASE_QUALITY_INCONCLUSIVE`. Absolute category scores do not exist
yet, and a pass is impossible until a same-protocol reference is measured.
The runner is `benchmark/run_p9_quality.py`. A dry run loaded 60 reasoning,
100 math, 100 factual, 50 instruction, and 40 coding items. Long-context
items are built only while the server is up.

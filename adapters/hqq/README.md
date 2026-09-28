# HQQ adapter boundary

HQQ is an optional external Apache-2.0 quantizer. The local boundary only calls the upstream
`BaseQuantizeConfig` and `HQQLinear` APIs; no HQQ source is copied into ExVRAM.

P8 uses HQQ for 1/2-bit group-wise reconstruction when the operator installs the pinned
revision from `experiments/manifests/p8_binary_qwen38.json`. Group sizes 8/16 are quality and
memory probes only; 32/64/128 are the fast-path candidates to compare with GemLite.

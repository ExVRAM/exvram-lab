# GemLite adapter boundary

This directory is reserved for an adapter to the separately installed GemLite kernels. No Triton
source is copied here. Validate shapes, correctness, and target-GPU performance before relying on
any kernel.

The thin local boundary follows the upstream `gemlite.GemLiteLinear` constructor and leaves
packing to the upstream `pack(W_q, scales, zeros, bias)` API. It is not exercised until CUDA
Torch, Triton, and a packed fixture are installed.

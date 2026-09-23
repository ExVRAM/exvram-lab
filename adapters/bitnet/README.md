# BitNet adapter boundary

This directory is reserved for an optional external BitNet integration/reference. BitNet-family
kernels and model formats are not generic replacements for every dense LLM path.

The local probe records BitNet as format-specific. A W2A8 result is admissible only after the
matrix shape and model format are shown to match the official GPU kernel.

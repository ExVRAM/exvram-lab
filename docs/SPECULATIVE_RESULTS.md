# Speculative decoding

No speculative run was started.

llama.cpp n-gram options and Qwen3.8 MTP/DFlash drafts are not comparable until the target
model is GPU-resident. The current IQ2_XXS file is larger than the 8151 MiB device, and the
fit log already keeps 2557 MiB of weights on the host. Adding a draft model on top of that
offload is `NOT_APPLICABLE_MEMORY` for a full-GPU speculative stack.

No acceptance rate, draft count, or effective tok/s is claimed.

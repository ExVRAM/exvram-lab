# P9 local BF16 feasibility

Verdict: **official Qwen3.8-27B BF16 cannot be run safely on this PC.**
No BF16 process was started for this note. `REFERENCE_UNAVAILABLE` stands.
The weight size below is the published tensor total, not a new download.

## What would have to be resident

| Piece | Bytes | Status |
|---|---:|---|
| Official BF16 tensors, `Qwen/Qwen3.8-27B` revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | 55,562,855,904 | Inventory of the published index |
| Text-only Coletti BF16 GGUF already on disk | 53,808,282,560 | File exists. A previous load left about 0.1 GiB free and is not repeated |
| Unsloth BF16 GGUF pair | 54,657,735,616 | Third-party conversion, not the reference |
| This PC physical RAM | 16.3 GB | MEASURED at the audit |
| This GPU | 8151 MiB | MEASURED |

A llama.cpp attempt with `-ngl 99` asked CUDA for 48,880 MiB and failed
before any calibration chunk. That is recorded in [IMATRIX.md](IMATRIX.md).

`E:` is a partition of NVMe disk `311CD0512GB`, 477 GB, the same disk as
`D:`. Sequential throughput was not measured. The USB disk is not used.

## llama.cpp on this build

Build b10982 can map a GGUF or read it with `--load-mode dio`. Mapping does
not keep the working set small once every token touches every weight. Direct
I/O still has to bring those bytes into RAM. Partial GPU offload leaves the
remaining layers in host RAM. Neither mode is a layer-discard loop. Stock
llama.cpp does not stream one BF16 layer from SSD and free it before the
next layer.

## Layer streaming that does exist

AirLLM (`lyogavin/airllm`) is an external project. Its README says frozen
weights can be streamed one layer at a time, and it names Qwen3.8-27B
training at about 2 GB of VRAM for sequence 512. That sentence is upstream.
It was not reproduced here, the code was not copied, and it is a training
claim rather than a P9 inference result. A llama.cpp discussion of the same
idea (ggml-org/llama.cpp #20045) reports that layer swapping waits on disk
and is slower than ordinary offload. RabbitLLM, mentioned there, was not
described as supporting the Qwen3.5 hybrid.

Using that method here would still start by downloading and splitting the
55.6 GB checkpoint. Decoder bytes after the two BF16 embedding matrices are
50,477,262,304, or 788,707,223 bytes per layer across 64 layers. Each
embedding matrix is about 2.37 GiB, larger than one decoder layer. The GPU
could in principle hold one layer. The disk would still move the whole
weight file once per generated token.

The 305 stored IQ2 calls produced 43,751 output tokens. Replaying that many
tokens by rereading 55,562,855,904 bytes each time is 2,430,930,508,655,904
bytes, 2.16 PiB, before the prompts. That volume is arithmetic on measured token
counts and the published weight size. It is not a timed benchmark. It is
enough to reject the method for this desktop: it is a multi-terabyte SSD
workload, and the first step is still a 55.6 GB download into a machine
with 5.3 GB free RAM.

Windows swap is not an acceptable way around that. The commit limit is
larger than physical RAM because of the pagefile. That headroom is not a
budget.

## Conclusion

No maintained runtime that this tree already runs can execute official 27B
BF16 inside 16 GB RAM and 8 GB VRAM without paging. Layer streaming exists
upstream and does not make a 395-item suite safe or practical here. The
reference stays unavailable.

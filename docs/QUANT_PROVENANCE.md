# Quant provenance

`IQ2_XXS` is a GGML type name, not a unique file. The local bartowski artifact and the
Unsloth `UD-IQ2_XXS` artifact are different physical representations.

Sizes below are Hugging Face LFS byte counts or a local GGUF header parse. They are not
runtime VRAM.

## Local bartowski IQ2_XXS

| Field | Value |
|---|---|
| Repository | `bartowski/Qwen3.8-27B-GGUF` |
| Revision | `0c92138c51f112d2f0dc84d5f6d1ebe4b9912b9a` |
| Quantizer | `quantized_by: bartowski` (repo card). Imatrix file is set. |
| File | `Qwen3.8-27B-IQ2_XXS.gguf` |
| File bytes | 8,881,268,448 |
| Architecture | `qwen35` (`general.architecture`) |
| `general.file_type` | 19 (`LLAMA_FTYPE_MOSTLY_IQ2_XXS`) |
| `general.quantization_version` | 2 |
| `qwen35.block_count` | 65 |
| Embedding length | 5120 |
| Heads / KV heads | 24 / 4 |
| Context metadata | 262144 |
| Tensor count | 866 |
| Vision tensors inside this GGUF | none |
| MTP | present as `blk.64.*`, including `blk.64.nextn.*` |

`mmproj` is a separate file on the Unsloth repo (`mmproj-BF16.gguf`, 931,146,432 bytes).
It is not inside the local GGUF.

Tensor payload from the local header, using ggml type-size/block-size. Sum of tensor
bytes is 8,870,273,024. The remaining 10,995,424 bytes are GGUF metadata and alignment.

| Group | Tensors | Elements | Payload bytes |
|---|---:|---:|---:|
| decoder `blk.0`–`blk.63` | 848 | 24,353,196,544 | 7,210,878,976 |
| MTP `blk.64` | 15 | 424,699,392 | 238,983,168 |
| `token_embd.weight` | 1 | 1,271,398,400 | 546,304,000 (`q3_k`) |
| `output.weight` + `output_norm.weight` | 2 | 1,271,403,520 | 874,106,880 (`q5_k` + `f32`) |

Payload by GGML type:

| Type | Bytes |
|---|---:|
| `iq2_xxs` | 4,974,182,400 |
| `iq3_xxs` | 957,358,080 |
| `q5_k` | 888,504,320 |
| `iq3_s` | 738,918,400 |
| `q3_k` | 546,304,000 |
| `iq4_xs` | 412,221,440 |
| `q4_0` | 238,878,720 |
| `f32` | 105,058,304 |
| `q4_k` | 8,847,360 |

The card label is mostly `iq2_xxs`. Embeddings are `q3_k` and `lm_head` (`output.weight`)
is `q5_k`. A previous llama.cpp load printed `IQ2_XXS - 2.0625 bpw`. That is the nominal
rate of the `iq2_xxs` type, not the average rate of this mixed file.

## Why 8.88 GB is not the 7.27 GB UD-IQ2_XXS

Unsloth `unsloth/Qwen3.8-27B-GGUF` file `Qwen3.8-27B-UD-IQ2_XXS.gguf`:

| Field | Value |
|---|---|
| LFS sha256 | `e792d8fb3142fe6d9171876d6da0f71f05a71028718debc72dbec93ff645e67d` |
| Bytes | 7,266,070,528 |
| License on the repo card | `apache-2.0` |
| Base model | `Qwen/Qwen3.8-27B` |
| Vision | separate `mmproj-*.gguf`, not in this file |
| MTP | separate directory; the only published MTP sidecar found is `MTP/mtp-Qwen3.8-27B-Q4_0.gguf` (1,369,590,656 bytes). It is not inside `UD-IQ2_XXS`. |

Local header parse after sha256 check
`e792d8fb3142fe6d9171876d6da0f71f05a71028718debc72dbec93ff645e67d`
(file bytes 7,266,070,528):

| Field | Value |
|---|---|
| Architecture | `qwen35` |
| `general.file_type` | 19 |
| `qwen35.block_count` | 64 |
| Tensors | 851 |
| Vision tensors | none |
| MTP `blk.64` | absent |

| Group | Tensors | Payload bytes | Types |
|---|---:|---:|---|
| decoder `blk.0`–`blk.63` | 848 | 6,291,572,736 | mixed, see below |
| `token_embd.weight` | 1 | 417,177,600 | `q2_k` |
| `output.weight` + norm | 2 | 546,324,480 | `q3_k` + `f32` |

Largest type payloads: `iq2_xxs` 2,418,155,520; `iq1_s` 1,500,160,000; `iq3_xxs` 678,379,520;
`iq2_s` 675,102,720; `q3_k` 584,601,600; `iq2_xs` 437,985,280; `q2_k` 441,262,080;
`iq1_m` 276,398,080. The file name is `IQ2_XXS`, but a large share of the payload is `iq1_s`.

Difference versus the bartowski file: 8,881,268,448 − 7,266,070,528 = 1,615,197,920 bytes
(1.504 GiB).

MTP in the bartowski file is only 238,983,168 bytes (227.9 MiB). Removing that block
still leaves the bartowski payload about 1.28 GiB larger than the Unsloth file. The rest
of the gap is a different mixed-precision recipe (bartowski imatrix layout versus Unsloth
Dynamic), not a second copy of the same `IQ2_XXS` tensor list.

Sibling Unsloth files, same repo, metadata only:

| File | Bytes | LFS sha256 |
|---|---:|---|
| `Qwen3.8-27B-UD-IQ1_M.gguf` | 6,729,166,848 | `1b5165a7149ea51e683c8eaf23372188ad9fc9d1a795386f7a1b558acf847dc6` |
| `Qwen3.8-27B-UD-IQ1_S.gguf` | 6,192,222,208 | `3895b6eaa91e705c06ad1938d16c22e86f073c6a67df86260a1da79be3d1f887` |

`F:` is a fixed NTFS disk (DriveType 3) with about 899 GiB free. One 7.27 GB download fits.
`UD-IQ1_M` and `UD-IQ1_S` stay un-downloaded until the `UD-IQ2_XXS` runtime fit is known.

## EXL3 metadata, no weight download

Repository `turboderp/Qwen3.8-27B-exl3`. Text-only H3 branches (no `_V` suffix). Tensor
byte counts are safetensors `data_offsets` from an HTTP range read of the header, not a
full download.

Embeddings are 2,542,796,800 bytes on every branch below, which is exactly
248,320 × 5,120 × 2 (BF16). They are not part of the low-bit decoder.

| Branch | Decoder | lm_head | Decoder + head | MTP | Vision | Embeddings |
|---|---:|---:|---:|---:|---:|---:|
| `SC_2.20bpw_H3` | 6,755,992,128 | 477,281,284 | 7,233,273,412 | 106,468,384 | 921,460,192 | 2,542,796,800 |
| `SC_2.00bpw_H3` | 6,147,818,048 | 477,281,284 | 6,625,099,332 | 106,468,384 | 921,460,192 | 2,542,796,800 |
| `SC_1.80bpw_H3` | 5,538,988,608 | 477,281,284 | 6,016,269,892 | 106,468,384 | 921,460,192 | 2,542,796,800 |
| `SC_1.60bpw_H3` | 4,930,814,528 | 477,281,284 | 5,408,095,812 | 106,468,384 | 921,460,192 | 2,542,796,800 |

`SC_2.20` decoder+head is 6.74 GiB. `SC_2.00` is 6.17 GiB. `SC_1.80` is 5.60 GiB.
A previous llama.cpp load left on the order of 1.7 GiB of non-model GPU memory
(device used minus GPU model buffer). Adding that overhead to `SC_2.00` exceeds a
7.5 GiB usable budget. `SC_1.80` decoder+head plus that overhead lands near 7.3 GiB,
inside the 7.2–7.5 GiB band, and only if embeddings, vision, and MTP stay off the GPU.
That is the first EXL3 candidate. It is not downloaded yet.

Total `.safetensors` size is the wrong comparator: it adds BF16 embeddings, vision, and MTP.

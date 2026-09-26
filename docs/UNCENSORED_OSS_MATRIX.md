# Uncensored open-source matrix

Checked 2026-09-24 from the Hugging Face model API and the GitHub API. No weights were
downloaded. Byte counts are sibling `size` fields. Refusal and quality figures below are
upstream card claims, not ExVRAM runs. ExVRAM does not describe any of these checkpoints as
completely uncensored.

Two layers stay separate:

- **Model behaviour.** The checkpoint should stay steerable, with substantially reduced
  built-in refusal. That is what this matrix records.
- **Application policy.** Any product restriction lives outside the weights and is
  configurable. It is not part of the runtime, the quantizer, or this scorecard.

Abliteration and quantization are different steps. The acceptable pipelines are:

```text
official BF16 -> abliteration on BF16/FP16 -> quality check -> quantization -> runtime
ready abliterated BF16 -> existing low-bit quantizer -> runtime
```

Abliterating a checkpoint that is already heavily quantized is recorded when an upstream
repo did it, and it is not the path ExVRAM will copy.

## Toolchains

| Project | Revision checked | License | Role | Copied into ExVRAM |
|---|---|---|---|---|
| [p-e-w/heretic](https://github.com/p-e-w/heretic) | `3521f8648a0dccf6e12a92666862632235fac7e6` (master, 2026-09-05). Latest tag `v1.4.0` is older (2026-06-14) | AGPL-3.0 | External directional-ablation optimizer. Co-minimizes refusal count and KL divergence. CLI: `heretic MODEL` or `heretic --model BASE --evaluate-model CANDIDATE` | No. AGPL code stays an external process |
| [nanofatdog/LLM-abliterate](https://github.com/nanofatdog/LLM-abliterate) | `f01cec9633a591657eea5934d1ce4a5999800844` (2026-08-15) | Not declared. No `LICENSE` file in that tree. GitHub license field is null | Training-free directional ablation used by the hotdogs checkpoint | No |
| [Sumandora/remove-refusals-with-transformers](https://github.com/Sumandora/remove-refusals-with-transformers) | `7786b0a8c50f4e7c16a0e300e697b2876decc0c6` (2025-11-27) | Apache-2.0 | Method cited by the Huihui card | No |
| [PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp) | `842b1880415d6f508f03b789e5ce70194def7bfd`, release `prism-b10735-842b188` (2026-09-24) | MIT | Required by Huihui's own card for the ternary GGUF series. Stock llama.cpp does not load those files | No |

Heretic's own README reports a Gemma-3-12B example (97/100 refusals down to 3/100, KL 0.16).
That table is not a Qwen3.8-27B result.

PrismML publishes a Windows x64 CUDA 13.3 binary,
`llama-prism-b10735-842b188-bin-win-cuda-13.3-x64.zip` (146,021,367 bytes), and a Windows
CUDA workflow file. The same release list did not include an x64 CUDA 13.4 binary. ExVRAM
has not built or run this fork. There is no SM120 timing in this repository. A 2026-09-24
commit on the fork fixes a CPU `PQ2_0` build; that is not evidence of an SM120 kernel.

## Behaviour sources

All three BF16 repos use `Qwen3_5ForConditionalGeneration`. On 2026-09-24 each safetensors
index had 1,199 tensor names, identical to `Qwen/Qwen3.8-27B` revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`: 15 `mtp.*`, 333 vision tensors, 1 `lm_head`.
Name equality is not value equality. Official index `total_size` is 55,562,855,904 bytes.
Huihui and hotdogs match that byte total. Coletti's index total is 55,562,857,544 bytes
(+1,640).

| Source | HF revision | License | Method | MTP | Vision | Upstream refusal | Upstream quality |
|---|---|---|---|---|---|---|---|
| [JonathanColetti/Qwen3.8-27B-Uncensored](https://huggingface.co/JonathanColetti/Qwen3.8-27B-Uncensored) | `5bb7aa90f0efef548e87005b1fb7658e522b6b7f` | Apache-2.0 | Heretic on BF16. Edits `attn.o_proj` and `mlp.down_proj` only. Card says all 15 `mtp.*` tensors were copied back from the base after the merge | Names match base. Card: copied verbatim, not abliterated | Present. Pipeline `image-text-to-text` | 98/100 base, **12/100** this checkpoint, on 100 held-out `mlabonne/harmful_behaviors` prompts, non-thinking. KL 0.1191. Card says this is not an over-refusal rate | 0-shot lm-eval mean delta **-0.5** (MMLU, ARC-Challenge, HellaSwag, Winogrande). Card: within or near standard error. No GSM8K or HumanEval in that table. Separate GGUF wikitext-2 PPL: base BF16 6.5129, this model BF16 6.5563 (**+0.0434**) |
| [hotdogs/Qwen3.8-27B-abliterated](https://huggingface.co/hotdogs/Qwen3.8-27B-abliterated) | `b3867530df40c9025188d67bcb7dcac113d4f932` | Apache-2.0 | LLM-abliterate, training-free, λ=1.2. Card says vision and `lm_head` are untouched and `mtp.*` is edited in the state dict | Names match base. Card: edited, not a verbatim copy | Present. Card: byte-identical visual tower | Same 100-prompt Heretic harness: 98/100 base, **39/100** this checkpoint. KL **0.0001**. A separate 20-prompt note (6/20) is not the comparable figure | Card A/B, not the same protocol as Coletti: MMLU −0.005 (limit 1000), GSM8K strict −0.03 (limit 100), ARC-Challenge +0.01 (300 items) |
| [huihui-ai/Huihui-Qwen3.8-27B-abliterated](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated) | `739e3c5b89849f6c238ce1e5b70008612ae42cdd` | Apache-2.0 | `remove-refusals-with-transformers`. Card: layers 18–51 ablated; MTP and visual not modified | Names match base. Card: not modified | Present. Card: not modified | Not stated on the inspected card | Not stated on the inspected card |

The shared 98/100 base figure is one upstream harness, not an ExVRAM measurement.
`mlabonne/harmful_behaviors` revision `01cead01398926d81f7c52bdb790ee8cf77ebba7` and
`mlabonne/harmless_alpaca` revision `02c6a92cfcf11bb0c387334f8146d149d65b587f` were inspected.
Neither card declared a license. Those prompt sets are not copied into Git.

Coletti also publishes the Heretic search front (23 non-dominated points, from 12/100 at KL
0.1191 up to 98/100 at KL 0.0004). A less aggressive point can be re-exported with Heretic.
That is still the upstream tool, not a new ExVRAM method.

## Compact artifacts

`FULL_GPU` on the detected RTX 5060 means weights and runtime state inside 8151 MiB
(7.960 GiB). A file smaller than that can still miss once KV, compute buffers, and allocator
reserve are included. Context 8192 was not run for any row.

| Artifact | HF revision | Method lineage | Quant | Bytes | GiB | MTP | Vision | Runtime | 8 GB file headroom |
|---|---|---|---|---:|---:|---|---|---|---|
| Coletti `Qwen3.8-27B-Uncensored-IQ2_M.gguf` | `45d0fc0ad6cfcf9eeee6008ddaea925536409b18` | Heretic BF16, then llama.cpp imatrix. Card cites llama.cpp `a94d563ed` | IQ2_M, fused | 10,624,771,968 | 9.895 | inline | separate `mmproj` 927,606,912 bytes | stock llama.cpp | No. File is larger than 8151 MiB |
| Coletti `...-noMTP-IQ2_M.gguf` | same | same | IQ2_M, no MTP | 10,173,451,200 | 9.475 | no | separate mmproj | stock llama.cpp | No |
| Huihui `...-Q2_K.gguf` | `3f101cd22b7999228bbd5d79a33975414eb9758b` | Huihui BF16 GGUF, then `llama-quantize` | Q2_K | 10,864,592,160 | 10.118 | card: MTP not modified on the BF16 update | `mmproj-model-bf16.gguf` 931,145,888 | card points at stock llama.cpp | No |
| Huihui `...-UD-IQ2_S.gguf` | same | Abliteration applied to an Unsloth UD GGUF. Layers 17–52 on that series | UD-IQ2_S | 8,424,398,848 | 7.846 | no in this filename | separate mmproj | likely stock llama.cpp; not run here | File is 116 MiB under 8151 MiB. No practical FULL_GPU margin |
| Huihui `...-UD-IQ2_S-MTP.gguf` | same | same UD lineage | UD-IQ2_S + MTP | 8,775,408,544 | 8.173 | filename says MTP | separate mmproj | not run | File exceeds 8151 MiB |
| Huihui `...-Ternary-Bonsai-PTQ1_0.gguf` | same | Abliteration applied to `prism-ml` ternary GGUF. Layers 22–52. Card: test/validation; some weights converted away from PTQ1 | ternary PTQ1_0 | 6,608,603,616 | 6.155 | not stated on the filename | separate mmproj | **PrismML-Eng/llama.cpp only** | Weights-only headroom 1.80 GiB. 8k fit is not measured |
| Huihui `...-Ternary-Bonsai-PQ2_0.gguf` | same | same ternary lineage | ternary PQ2_0 | 7,704,693,216 | 7.176 | not stated | separate mmproj | PrismML fork | File fits the raw VRAM number. 8k fit is not measured |
| WatchDG EXL3 2.25 bpw shard sum | `f7fb5b58fc349334f7e32c56a6ff7fe35b37fc15` | Base `Blackfrost-AI/Qwen3.8-27B-ABLITERATED-BF16`, not Coletti | EXL3 2.25 bpw | 11,536,664,983 | 10.744 | not split out | pipeline `image-text-to-text` | ExLlamaV3, not run | No |
| grimlee EXL3 3.0 bpw shard sum | `64371e7543b20705b491b4a1ab4b2ba42722b3a8` | Base `huihui-ai/Huihui-Qwen3.8-27B-abliterated` | EXL3 3.0 bpw | 13,468,688,185 | 12.543 | not split out | pipeline `image-text-to-text` | ExLlamaV3, not run | No |
| orcarouter Uncensored NVFP4 shard sum | `96d4d0b66d94314909d1af73f955e94bf68335ac` | Card base is official Qwen, not a named abliterated repo | NVFP4 shards plus a 849,400,424-byte extra | 24,688,489,848 | 22.992 | extra shard, not inventoried here | pipeline `image-text-to-text` | not identified as an ExVRAM runtime | No |
| sakamakismile Huihui NVFP4 | `36276309d30211d9babf72f22d1605dc2dc6357c` | Huihui BF16 | NVFP4 plus BF16 MTP sidecar | 20,559,284,232 | 19.147 | `model-mtp-bf16.safetensors` 849,400,424 | text-generation pipeline | not run | No |
| HauhauCS Aggressive `IQ2_M` | `993a5971fda8f30dd1b7eb2654792ba4415c7460` | Card does not name Heretic, abliteration, or a fine-tune. It calls the release an aggressive uncensoring profile | IQ2_M | 10,319,906,944 | 9.611 | card: native NextN kept; extra FastMTP sidecar 903,453,952 | separate BF16 mmproj 931,146,624 | stock llama.cpp assumed, not run | No |

`turboderp/Huihui-Qwen3.8-27B-abliterated-exl3` revision `bdbe86bc337e53e2fbee812043a9eb063bcb8c8f`
lists the Huihui BF16 model as its base and Apache-2.0 metadata. The default branch had no
weight siblings in the API response, so no EXL3 byte count is claimed for it.

NVFP4 is relevant to SM120 in principle. The two inspected NVFP4 repos are about 19–23 GiB
on disk. They are not 8 GB candidates. MXFP4 was not a separate artifact in the repos sized
here.

## Class C

`HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF` is a distinct, more aggressive
published line. The card claims **0/465 refusals** and does not define the prompt set, the
judge, or the quantization used for that count. That number is not comparable to the 100-prompt
Heretic harness. The smallest full GGUF inspected, IQ2_M, is 9.611 GiB. It is not the selected
source.

## Selected source

**JonathanColetti/Qwen3.8-27B-Uncensored**, Heretic on BF16.

It is the named Heretic checkpoint, the lowest published refusal on the shared 100-prompt
harness, and the one that documents a verbatim MTP restore. hotdogs remains the
quality-preservation alternative (KL 0.0001, 39/100 residual refusals) if a later local
quality gate rejects the Coletti point. Huihui remains the place to read sub-8 GB filenames,
not the behaviour source, because those small files were abliterated after quantization and
the ternary series needs an unmeasured fork.

No first-party uncensoring code is justified. The missing piece is a compact quant of this
BF16 checkpoint, produced with llama.cpp, Unsloth, or ExLlamaV3.

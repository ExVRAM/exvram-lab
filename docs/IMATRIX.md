# Importance matrix

Status: **not generated**. The Coletti BF16 snapshot is the calibration source.
No imatrix SHA256 exists yet, and no original-Qwen imatrix has been reused.

## Runtime

| Field | Value |
|---|---|
| Binary | local llama.cpp `llama-imatrix.exe`, build 10982 |
| Build | 10982 |
| Commit | `fc82583e6` |
| Version string | `0.4.1-dev` |
| CUDA | prebuilt Windows CUDA 13.4 bundle; `llama-cli --list-devices` lists CUDA0 RTX 5060 (8150 MiB) |
| SM120 | the prebuilt enumerates this GPU. There is no ExVRAM cmake log that proves every kernel was compiled as native `sm_120` |

`llama-quantize.exe` in the same directory accepts `IQ2_XXS` (nominal 2.06 bpw), `IQ1_M` (nominal 1.75), and `IQ1_S` (nominal 1.56). Nominal bpw is not physical bpw.

## Corpus

| Field | Value |
|---|---|
| Dataset | `Salesforce/wikitext` |
| HF revision checked | `b08601e04326c79dfdd32d625aee71d232d685c3` |
| License | `cc-by-sa-3.0` and `gfdl` |
| Config | `wikitext-2-raw-v1` |
| Split used for calibration | `train` |
| Chunks | 200 |
| Seed | 17, only if a subset is sampled; the extracted file order is the reproducible input |
| Why this corpus | plain language, not a refusal set. The test split stays unused so a later perplexity run is not the calibration text |

The corpus text stays outside Git. ShareAlike and GFDL terms are not imported into this Apache-2.0 tree.

## Planned command

After the BF16 snapshot is converted to a GGUF with the same llama.cpp commit:

```text
llama-imatrix.exe -m <coletti-bf16-or-f16.gguf> -f <wikitext-2-raw-train.txt> -o <imatrix.dat> --output-format dat --chunks 200 -ngl 99
```

The F16/BF16 GGUF of this 27B model does not fit in 8151 MiB. Layer offload during imatrix generation is calibration I/O, not the FULL_GPU decode result. The decode benchmark still requires CPU decoder offload of 0.

Output to record when the run exists: imatrix SHA256, exact argv, llama.cpp commit, corpus revision, seed, and chunk count.

Coletti publishes an imatrix next to their GGUF repo. That file is not the first input. This project generates the matrix from the verified Coletti weights. Reuse of an official Qwen3.8-27B imatrix is only a fallback if this generation cannot finish, and any quant that uses it must be labeled as reused.

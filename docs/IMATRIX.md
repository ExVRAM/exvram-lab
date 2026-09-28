# Importance matrix

Status: **not generated**. The verified Coletti BF16 snapshot has been converted to a text-only GGUF. A 200-chunk run was started and did not finish. No imatrix SHA256 exists, and no original-Qwen imatrix has been reused.

## Runtime

| Field | Value |
|---|---|
| Binary | local llama.cpp `llama-imatrix.exe`, build 10982 |
| Build | 10982 |
| Commit | `fc82583e6` |
| Version string | `0.4.1-dev` |
| CUDA | prebuilt Windows CUDA 13.4 bundle; `llama-cli --list-devices` lists CUDA0 RTX 5060 (8150 MiB) |
| SM120 | `llama-imatrix` system_info on this GPU printed `CUDA : ARCHS = 750,800,860,890,900,1200,1210` and `BLACKWELL_NATIVE_FP4 = 1`. That is the binary's own arch list, not an ExVRAM cmake log |

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

The train parquet (`6,357,543` bytes) was saved under `work/calibration/wikitext-2-raw-v1/` and expanded to `work/calibration/wikitext-2-raw-v1-train.txt` (`10,964,514` bytes, `36,718` rows). No subset was sampled, so seed 17 was not applied.

## Command

The text GGUF is `artifacts/derived/qwen38-27b-uncensored-coletti-5bb7aa90-text-bf16.gguf`: 851 tensors, layers 0–63, no vision tensors, no MTP tensors (`--no-mtp`). File size `53,808,282,560` bytes. The last tensor ends at EOF.

Passing `-ngl 99` on this build makes `--fit` abort (`n_gpu_layers already set`) and then `cudaMalloc` asks for the whole BF16 model (`48,880` MiB). That attempt failed before any chunk. The calibration command leaves layer placement to `--fit` and keeps the chunk context at 512:

```text
llama-imatrix.exe -m artifacts/derived/qwen38-27b-uncensored-coletti-5bb7aa90-text-bf16.gguf -f work/calibration/wikitext-2-raw-v1-train.txt -o artifacts/derived/qwen38-27b-uncensored-coletti-5bb7aa90-wikitext2-chunks200.dat --output-format dat --chunks 200 -c 512 -b 512 -ub 512 --fit on --fit-ctx 512
```

The BF16 GGUF does not fit in 8151 MiB. Layer offload during imatrix generation is calibration I/O, not the FULL_GPU decode result. The decode benchmark still requires CPU decoder offload of 0.

A `llama-quantize --dry-run` without an imatrix (this IQ type warns that a real quant needs one) reported IQ2_XXS at `8032.03` MiB (`2.51` BPW in the quantizer's mixture: `output.weight` `q5_K`, `token_embd.weight` `q2_K`) and IQ1_M at `7267.08` MiB (`2.27` BPW). Those figures are size estimates, not a FULL_GPU measurement.

One calibration attempt with `--fit on` reached the first chunk: `74.59` seconds per pass, ETA about 4 hours, first-chunk perplexity `[1]6.9562`. Physical RAM on this 16 GB machine fell to about `0.1` GB free while that process ran, and the process exited before chunk 2 was saved. A later file at the imatrix path was only `20,579` bytes and is not a usable matrix. The quantizer must not run on it. The P6 llama-server was on this GPU again after that exit, so the calibration was not restarted on top of it. A later detached run was stopped on purpose: it again left about `0.1` GiB of RAM free and made the workstation unusable. Do not resume a BF16 imatrix on this 16 GiB machine. The dat file is not a finished matrix.

Output to record when the run exists: imatrix SHA256, exact argv, llama.cpp commit, corpus revision, seed, and chunk count.

Coletti publishes an imatrix next to their GGUF repo. That file is not the first input. This project generates the matrix from the verified Coletti weights. Reuse of an official Qwen3.8-27B imatrix is only a fallback if this generation cannot finish, and any quant that uses it must be labeled as reused.

# Requantization disk plan

Measured 2026-09-25 before any Coletti weight download. Byte totals for the source
come from the Hugging Face model API at revision
`5bb7aa90f0efef548e87005b1fb7658e522b6b7f` (22 files, API `sha` matches that
revision). Free space is `GetDiskFreeSpaceExW` user-available bytes.

## Volume decision

| Volume | DriveType | Filesystem | Free | Role in this pipeline |
|---|---|---|---:|---|
| `C:` | 3, fixed | NTFS | 33.25 GiB | Rejected. Smaller than the 51.77 GiB source. The default Hugging Face cache is `%USERPROFILE%\.cache\huggingface` on this volume. |
| `D:` | 3, fixed | NTFS | 50.97 GiB | Rejected. Smaller than the source. |
| `E:` | 3, fixed | NTFS | 280.27 GiB | Selected. The project tree is already on this volume. |
| `F:` | 3, reported fixed | NTFS | 894.57 GiB | **Rejected.** The operator identified this volume as removable. `GetDriveTypeW` returning 3 is not sufficient. No download, conversion, imatrix, quant, or benchmark path may use `F:`. |

The default user Hugging Face hub cache exists and currently holds only
`sentence-transformers/all-MiniLM-L6-v2`. It does not contain the Coletti
checkpoint. Downloads must not fall through to that cache.

No Coletti blob was present under the project `hf_cache` or `models`
directory before this plan. `models` already holds an unrelated Unsloth
`UD-IQ2_XXS` file (7,266,070,528 bytes). That file is not an input to this
conversion.

## Budget

API file sum for the pinned revision: **55,583,133,236 bytes (51.77 GiB)**.
Safetensors sum: **55,563,007,192 bytes**. The index `total_size` recorded
earlier, 55,562,857,544, is tensor payload, not the download size.

Windows cannot create symlinks here (`WinError 1314`). `snapshot_download`
with `local_dir` does not also fill the hub blob cache, so the source is one
copy, not two.

| Resident set | Planned bytes | Notes |
|---|---:|---|
| Coletti BF16 `local_dir` | 55,583,133,236 | immutable source |
| BF16/F16 GGUF intermediate | about 52–56 GiB | one conversion output |
| first IQ quant | under 10 GiB | IQ2_XXS target |
| imatrix + WikiText-2 calibration text | under 1 GiB | corpus is not committed |
| **Peak if all are kept** | **about 115–120 GiB** | source stays |

The GGUF planning floor is 130–150 GB of free space before the first download.
`E:` has 280.27 GiB free, which covers that floor with the source, the
intermediate GGUF, and the quant resident at the same time.

EXL3 would avoid the second full-size GGUF, but that saving is not required
on `E:`. The first pipeline stays GGUF.

## Paths

- Source, one copy: `models/sources/JonathanColetti-Qwen3.8-27B-Uncensored-5bb7aa90`
- Derived GGUF and imatrix: `artifacts/derived/`
- Fetcher: `benchmark/fetch_coletti_bf16.py`
- Runtime already on the selected volume: llama.cpp build 10982, commit `fc82583e6`, CUDA 13.4 prebuilt under the local llama cache

`HF_HOME` and `HF_HUB_CACHE` must not point at `C:` or `F:` for this work.
The fetcher passes `local_dir` and does not use a second cache copy.

The pinned download was started on 2026-09-25 into that `E:` directory.
`benchmark/fetch_coletti_bf16.py` writes
`experiments/manifests/coletti_qwen38_uncensored_bf16.json` only after every
expected file matches its API size and a local SHA256 is recorded. Until that
manifest exists, the snapshot is incomplete and must not be converted.

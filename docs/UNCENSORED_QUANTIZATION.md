# Coletti uncensored quantization

Status: **BF16 GGUF present locally; IQ2_XXS not produced**.

The required source is `JonathanColetti/Qwen3.8-27B-Uncensored` at revision
`5bb7aa90f0efef548e87005b1fb7658e522b6b7f`. A local text-only BF16 GGUF of that
revision is on disk at 53,808,282,560 bytes. No matching IQ2_XXS file was
produced. An imatrix `.dat` from a WikiText-2 pass is only 13,582,676 bytes
and is not treated as a finished calibration.

The next step is to finish imatrix calibration on that local BF16 GGUF and
requantize with llama.cpp `llama-quantize` to the same UD-IQ2_XXS pipeline as
the base model. Heretic and fine-tuning remain out of scope. The artifact is
not uploaded.

Required provenance before any benchmark:

1. source revision and per-file hashes;
2. quantizer build/commit and imatrix provenance;
3. exact quant command;
4. output bytes, SHA256 and physical bpw;
5. tensor topology verification.

# P9 Coletti imatrix

Status: **not finished**. The partial file is not a calibration.

Source: `JonathanColetti/Qwen3.8-27B-Uncensored` revision
`5bb7aa90f0efef548e87005b1fb7658e522b6b7f`. Local text-only BF16 GGUF:
53,808,282,560 bytes. Corpus, chunk count, seed, and the failed command are
in [IMATRIX.md](IMATRIX.md). That document's measured outcome still holds:
on this 15.93 GiB host, `llama-imatrix` with `--fit on` reached one chunk,
left about 0.1 GiB of RAM free, and did not save a usable matrix. A later
dat file of 13,582,676 bytes is not a finished matrix. No output SHA-256
exists.

Do not resume that command on this machine. A quant that needs this imatrix
stays unproduced. Reusing someone else's imatrix would be a different
artifact and is not recorded here.

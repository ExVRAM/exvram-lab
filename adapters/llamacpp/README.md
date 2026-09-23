# llama.cpp adapter boundary

This directory is reserved for an adapter to an externally installed llama.cpp executable or API.
Use it as an independent baseline and record the exact binary/source revision in results.

The local command builder targets the upstream `llama-bench -o jsonl -r N -ngl N` interface. It
does not download models or execute an unpinned binary automatically.

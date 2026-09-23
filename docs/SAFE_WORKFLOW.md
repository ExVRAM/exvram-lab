# Safe workflow for ExVRAM measurements

Normal development does not need any removable volume. Keep the Git working tree separate from
model weights and external runtime binaries.

## Safe by default

- Run lint, compile checks, unit tests, synthetic benchmarks, and documentation work from the
  project directory.
- Keep external model files and runtime binaries outside Git and do not download them implicitly.
- Keep full-model benchmarks in the foreground with a bounded timeout.
- Stop a run if the removable drive disconnects, Windows reports I/O delays, or the system starts
  lagging.
- The P2 runners reject Windows removable-volume paths by default. `--allow-removable-storage`
  is an explicit opt-in for a controlled experiment, not a normal setting.

## Before a GPU run

1. Confirm the model and runtime are on a stable local/data volume.
2. Confirm enough free space for temporary files and model memory mapping.
3. Run a short device probe first.
4. Run one context point at a time and preserve the JSONL result.
5. Do not infer 8k fit, quality, or target throughput from a c128 smoke run.

The repository does not vendor external runtimes. A missing or unavailable runtime must produce a
clear `INCONCLUSIVE`/unavailable result rather than triggering an automatic download.

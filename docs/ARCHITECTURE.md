# Architecture

ExVRAM Lab is a measurement and planning layer, not a new inference engine.

```text
experiment JSON
      |
      v
config + memory planner ----> plan JSON
      |
      +----> optional adapter registry ----> external runtime (not vendored)
      |
      +----> synthetic/real benchmark ----> versioned result JSON
                                      |
                                      v
                               quality gate scaffold
```

The P2 path adds a provenance manifest and an append/upsert research store around the same
boundary:

```text
model/backend manifest -> P2 queue -> external runner -> raw JSONL -> SQLite research DB
                                      |                         |
                                      +---- P2/Pareto/gap ------+---- P3 branch selector
```

## Contracts

- `ExperimentConfig` is the input contract. It records model shape, weight/KV assumptions,
  hardware budget, target, and candidate adapters.
- `AdapterProtocol` is intentionally narrow: metadata, availability, and planning. An adapter
  must not import a heavy runtime merely to list itself.
- `ResultRecord` is the persisted output contract. It always identifies whether a result is
  synthetic and carries hardware telemetry plus a quality-gate status.
- `MemoryBudget` exposes raw packed-weight math, scale metadata, KV estimate, reserve, total, and
  assumptions so a plan can be challenged rather than treated as a fact.
- `QueueItem` and the research-record schema use explicit `PLANNED`, `RUNNING`, `PASS`, `FAIL`,
  and `INCONCLUSIVE` states. SQLite rows are upserted by stable experiment ID, while JSONL keeps
  an exchangeable raw record format.
- `benchmark/run_exllama_p2.py` and `benchmark/run_llamacpp_p2.py` are integration-boundary
  runners. They invoke optional installed runtimes; they do not copy or import their kernels.

## Dependency policy

The base package uses the Python standard library. CUDA/Torch and all external runtimes are
optional. The first-party code contains orchestration, schemas, planning, and measurement glue;
it does not vendor low-level kernels.

## Failure behavior

No CUDA, PyTorch, `nvidia-smi`, or external adapter is a supported state. Detection returns an
explicit unavailable state and the CPU/synthetic path remains usable. Runtime-specific failures
must be surfaced in result metadata rather than converted into fabricated throughput numbers.
Large model files and runtime bundles are external caches, never repository dependencies or Git
artifacts.

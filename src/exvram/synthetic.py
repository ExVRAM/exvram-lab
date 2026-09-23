from __future__ import annotations

import time

from .contracts import ExperimentConfig
from .hardware import HardwareInfo
from .results import new_result


def _work(seed: int, work_units: int) -> int:
    state = seed & 0xFFFFFFFF
    for index in range(work_units):
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        state = (state + index * 2654435761) & 0xFFFFFFFF
    return state


def run_synthetic_microbenchmark(
    config: ExperimentConfig,
    hardware: HardwareInfo,
    *,
    iterations: int = 20,
    work_units: int = 5_000,
) -> object:
    if iterations <= 0 or work_units <= 0:
        raise ValueError("iterations and work_units must be positive")
    started = time.perf_counter()
    checksum = 0
    for iteration in range(iterations):
        checksum ^= _work(config.seed + iteration, work_units)
    elapsed = time.perf_counter() - started
    total_operations = iterations * work_units
    return new_result(
        config=config,
        hardware=hardware,
        benchmark_kind="synthetic_microbenchmark",
        synthetic=True,
        metrics={
            "iterations": iterations,
            "work_units_per_iteration": work_units,
            "elapsed_seconds": elapsed,
            "iterations_per_second": iterations / elapsed if elapsed else None,
            "pseudo_operations": total_operations,
            "pseudo_operations_per_second": total_operations / elapsed if elapsed else None,
            "checksum": checksum,
        },
        notes=[
            "Synthetic CPU workload only; this is not model inference.",
            "Do not interpret pseudo_operations_per_second as tokens/second or VRAM usage.",
        ],
    )


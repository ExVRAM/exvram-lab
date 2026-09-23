from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .adapters import all_adapters, get_adapter
from .config import load_experiment
from .errors import ExVRAMError
from .hardware import detect_hardware
from .layer_benchmark import DEFAULT_BACKENDS, DEFAULT_LAYER_SHAPES, run_layer_benchmark
from .memory import compare_memory_configurations, estimate_memory
from .research import (
    default_p2_queue,
    load_manifest,
    upsert_items,
    write_queue_jsonl,
)
from .results import append_jsonl_payload, write_result
from .synthetic import run_synthetic_microbenchmark


def _dump(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="exvram", description="ExVRAM Lab research workflow CLI")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("list-adapters", help="list optional external runtime integrations")
    commands.add_parser("validate-environment", help="report CUDA/GPU and integration availability")

    manifest = commands.add_parser(
        "validate-research-manifest", help="validate model/backend provenance for P2 planning"
    )
    manifest.add_argument("--manifest", required=True)

    p2 = commands.add_parser("plan-p2", help="create the machine-readable P2 experiment queue")
    p2.add_argument("--manifest", required=True)
    p2.add_argument("--queue-output", required=True, help="JSONL queue output")
    p2.add_argument("--database", help="optional SQLite research database")

    record = commands.add_parser(
        "record-research", help="upsert a completed/failed research record into SQLite"
    )
    record.add_argument("--result", required=True, help="JSON result object")
    record.add_argument("--database", required=True)

    plan = commands.add_parser(
        "plan-experiment", help="calculate a memory plan without running a model"
    )
    plan.add_argument("--config", help="JSON experiment config or matrix")
    plan.add_argument("--index", type=int, default=0, help="matrix entry index (default: 0)")

    compare = commands.add_parser(
        "compare-memory-configurations", help="rank memory-saving scenarios against the baseline"
    )
    compare.add_argument("--config", help="JSON experiment config or matrix")
    compare.add_argument("--index", type=int, default=0)

    synthetic = commands.add_parser(
        "run-synthetic-microbenchmark", help="run the CPU-only synthetic workflow"
    )
    synthetic.add_argument("--config", help="JSON experiment config or matrix")
    synthetic.add_argument("--index", type=int, default=0)
    synthetic.add_argument("--iterations", type=int, default=20)
    synthetic.add_argument("--work-units", type=int, default=5_000)
    synthetic.add_argument("--output", help="write the result JSON to this path")

    layer = commands.add_parser(
        "run-layer-benchmark", help="run real CUDA layer timing or record unavailable backends"
    )
    layer.add_argument("--backend", action="append", help="repeat to select specific backends")
    layer.add_argument(
        "--layer",
        action="append",
        dest="layers",
        help="repeat to select layer names from the default shape set",
    )
    layer.add_argument("--warmup", type=int, default=10)
    layer.add_argument("--repeats", type=int, default=50)
    layer.add_argument(
        "--batch-size",
        action="append",
        type=int,
        dest="batch_sizes",
        help="repeat to measure M values; default is M=1",
    )
    layer.add_argument("--seed", type=int, default=17)
    layer.add_argument("--output", help="append raw result records to a JSONL path")
    return parser


def _list_adapters() -> int:
    hardware = detect_hardware()
    _dump(
        {
            "schema_version": 1,
            "adapters": [
                {
                    **adapter.info.to_dict(),
                    "availability": adapter.availability().to_dict(),
                }
                for adapter in all_adapters()
            ],
            "hardware": hardware.to_dict(),
        }
    )
    return 0


def _validate_environment() -> int:
    hardware = detect_hardware()
    _dump(
        {
            "schema_version": 1,
            "status": (
                "cuda_runtime_available"
                if hardware.cuda_runtime_available
                else "cuda_driver_visible_runtime_unavailable"
                if hardware.cuda_available
                else "cuda_unavailable"
            ),
            "hardware": hardware.to_dict(),
            "adapters": [adapter.plan(load_experiment()) for adapter in all_adapters()],
        }
    )
    return 0


def _validate_research_manifest(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    _dump(
        {
            "schema_version": 1,
            "status": "valid",
            "manifest": args.manifest,
            "model": manifest["model"],
            "candidate_count": len(manifest["candidates"]),
            "candidates": manifest["candidates"],
        }
    )
    return 0


def _plan_p2(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    queue = default_p2_queue(manifest)
    queue_path = write_queue_jsonl(queue, args.queue_output)
    database_rows = upsert_items(queue, args.database) if args.database else 0
    _dump(
        {
            "schema_version": 1,
            "phase": "P2",
            "status": "planned",
            "manifest": args.manifest,
            "queue_output": queue_path,
            "queue_records": len(queue),
            "database_rows": database_rows,
            "experiments": [item.to_dict() for item in queue],
        }
    )
    return 0


def _record_research(args: argparse.Namespace) -> int:
    result_path = args.result
    try:
        payload = json.loads(Path(result_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExVRAMError(f"cannot read research result {result_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ExVRAMError("research result must be a JSON object")
    rows = upsert_items([payload], args.database)
    _dump({"schema_version": 1, "database": args.database, "rows_upserted": rows})
    return 0


def _plan_experiment(args: argparse.Namespace) -> int:
    config = load_experiment(args.config, args.index)
    hardware = detect_hardware()
    candidates = [config.adapter, *config.candidate_adapters]
    unique = list(dict.fromkeys(candidates))
    plans = []
    for name in unique:
        try:
            plans.append(get_adapter(name).plan(config))
        except KeyError as exc:
            plans.append({"adapter": name, "status": "unknown", "error": str(exc)})
    _dump(
        {
            "schema_version": 1,
            "config": config.to_dict(),
            "hardware": hardware.to_dict(),
            "memory_budget": estimate_memory(config).to_dict(),
            "adapter_plans": plans,
        }
    )
    return 0


def _compare_memory(args: argparse.Namespace) -> int:
    config = load_experiment(args.config, args.index)
    _dump(compare_memory_configurations(config))
    return 0


def _run_synthetic(args: argparse.Namespace) -> int:
    config = load_experiment(args.config, args.index)
    hardware = detect_hardware()
    record = run_synthetic_microbenchmark(
        config,
        hardware,
        iterations=args.iterations,
        work_units=args.work_units,
    )
    output = write_result(record, args.output)
    payload = record.to_dict()
    if output:
        payload["saved_to"] = output
    _dump(payload)
    return 0


def _select_layers(names: list[str] | None) -> tuple[dict[str, int | str], ...]:
    if not names:
        return DEFAULT_LAYER_SHAPES
    known = {str(layer["name"]): layer for layer in DEFAULT_LAYER_SHAPES}
    selected = []
    for name in names:
        if name not in known:
            raise ExVRAMError(f"unknown layer {name}; known layers: {', '.join(known)}")
        selected.append(known[name])
    return tuple(selected)


def _run_layer(args: argparse.Namespace) -> int:
    hardware = detect_hardware()
    backends = tuple(args.backend) if args.backend else DEFAULT_BACKENDS
    records = run_layer_benchmark(
        hardware,
        backends=backends,
        layers=_select_layers(args.layers),
        warmup=args.warmup,
        repeats=args.repeats,
        seed=args.seed,
        batch_sizes=tuple(args.batch_sizes) if args.batch_sizes else (1,),
    )
    saved_to = None
    if args.output:
        for record in records:
            saved_to = append_jsonl_payload(record, args.output)
    counts: dict[str, int] = {}
    for record in records:
        status = str(record["measurement_status"])
        counts[status] = counts.get(status, 0) + 1
    _dump(
        {
            "schema_version": 1,
            "benchmark_kind": "layer_microbenchmark",
            "synthetic": False,
            "hardware": hardware.to_dict(),
            "summary": {"records": len(records), "by_status": counts},
            "results": records,
            **({"saved_to": saved_to} if saved_to else {}),
        }
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "list-adapters":
            return _list_adapters()
        if args.command == "validate-environment":
            return _validate_environment()
        if args.command == "validate-research-manifest":
            return _validate_research_manifest(args)
        if args.command == "plan-p2":
            return _plan_p2(args)
        if args.command == "record-research":
            return _record_research(args)
        if args.command == "plan-experiment":
            return _plan_experiment(args)
        if args.command == "compare-memory-configurations":
            return _compare_memory(args)
        if args.command == "run-synthetic-microbenchmark":
            return _run_synthetic(args)
        if args.command == "run-layer-benchmark":
            return _run_layer(args)
        raise ExVRAMError(f"unknown command: {args.command}")
    except (ExVRAMError, ValueError, OSError) as exc:
        print(json.dumps({"error": str(exc), "type": type(exc).__name__}), file=sys.stderr)
        return 2

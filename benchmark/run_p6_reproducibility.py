"""Repeatable llama.cpp measurements for the P6 context/quality gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

try:
    from run_p5_context_scaling import (
        _hardware,
        _parse_gpu,
        _parse_log,
        _percentile,
        _pick_port,
        _sample_system,
        _start_gpu_monitor,
        _stop_process,
        _utc_now,
        _wait_health,
        assert_model_fits_host_ram,
        desktop_safe_launch_args,
        windows_launch_flags,
    )
except ModuleNotFoundError:
    from benchmark.run_p5_context_scaling import (
        _hardware,
        _parse_gpu,
        _parse_log,
        _percentile,
        _pick_port,
        _sample_system,
        _start_gpu_monitor,
        _stop_process,
        _utc_now,
        _wait_health,
        assert_model_fits_host_ram,
        desktop_safe_launch_args,
        windows_launch_flags,
    )

from exvram.storage import StorageSafetyError, assert_storage_safe


def _lock_handle(handle) -> None:
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        return
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_handle(handle) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class _OutputLock:
    def __init__(self, output: Path) -> None:
        self.path = output.with_name(f"{output.name}.lock")
        self.handle = None

    def __enter__(self):
        self.handle = self.path.open("a+b")
        try:
            self.handle.seek(0, os.SEEK_END)
            if self.handle.tell() == 0:
                self.handle.write(b"\0")
                self.handle.flush()
            self.handle.seek(0)
            _lock_handle(self.handle)
        except OSError as exc:
            self.handle.close()
            self.handle = None
            raise RuntimeError(
                f"another P6 run already holds {self.path}"
            ) from exc
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(f"pid={os.getpid()}\n".encode("ascii"))
        self.handle.flush()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.handle is None:
            return
        try:
            _unlock_handle(self.handle)
        finally:
            self.handle.close()
            self.handle = None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--contexts", default="256,512,1024,2048,4096,8192")
    parser.add_argument("--cache-type-k", default="q4_0")
    parser.add_argument("--cache-type-v", default="q4_0")
    parser.add_argument("--decode-tokens", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--prompt-mode", choices=("short", "filled"), default="short")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--ubatch-size", type=int, default=1)
    parser.add_argument("--max-background-vram-mib", type=int, default=1024)
    parser.add_argument("--allow-contended-gpu", action="store_true")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--model-sha256", default=None)
    parser.add_argument("--allow-removable-storage", action="store_true")
    return parser


def _token_count(tokenizer: str, model: str, prompt_path: Path) -> int:
    command = [tokenizer, "-m", model, "-f", str(prompt_path), "--show-count"]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    text = f"{completed.stdout}\n{completed.stderr}"
    marker = "Total number of tokens:"
    for line in reversed(text.splitlines()):
        if marker in line:
            return int(line.split(marker, 1)[1].strip())
    raise RuntimeError(f"tokenizer failed: {text[-1000:]}")


def _short_prompt() -> str:
    return "Explain quantized inference in one concise paragraph."


def _filled_prompt(
    tokenizer: str,
    model: str,
    context: int,
    decode_tokens: int,
    probe_path: Path,
) -> tuple[str, int, int]:
    target = max(1, context - decode_tokens)
    if target <= 4:
        prompt = "x"
        probe_path.write_text(prompt, encoding="utf-8", newline="\n")
        return prompt, _token_count(tokenizer, model, probe_path), target

    seed = (
        "This is deterministic context filler for ExVRAM Lab. Quantized inference "
        "trades numerical precision for lower memory traffic while preserving the "
        "same model computation. "
    )
    suffix = "\n\nNow explain the central idea in one concise paragraph."

    def build(body_chars: int) -> str:
        body = (seed * math.ceil(body_chars / len(seed)))[:body_chars]
        return body + suffix

    body_chars = max(32, target * 6)
    best_prompt = build(body_chars)
    best_count = 0
    best_distance = sys.maxsize
    for _ in range(12):
        prompt = build(body_chars)
        probe_path.write_text(prompt, encoding="utf-8", newline="\n")
        count = _token_count(tokenizer, model, probe_path)
        distance = abs(count - target)
        if distance < best_distance:
            best_prompt, best_count, best_distance = prompt, count, distance
        if count == target:
            return prompt, count, target
        delta = target - count
        body_chars = max(1, body_chars + delta * 6)
    return best_prompt, best_count, target


def _completion(
    port: int,
    prompt: str,
    decode_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    payload = json.dumps(
        {
            "prompt": prompt,
            "n_predict": decode_tokens,
            "temperature": 0.0,
            "seed": 42,
            "stream": True,
            "cache_prompt": False,
            "ignore_eos": True,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    arrivals: list[float] = []
    final_payload: dict[str, Any] = {}
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue
            if event.get("content"):
                arrivals.append(time.perf_counter())
            if isinstance(event.get("timings"), dict):
                final_payload = event["timings"]
    intervals = [
        (right - left) * 1000.0 for left, right in zip(arrivals, arrivals[1:])
    ]
    return {
        "stream_events": len(arrivals),
        "ttft_ms": (arrivals[0] - started) * 1000.0 if arrivals else None,
        "inter_token_p50_ms": _percentile(intervals, 50),
        "inter_token_p95_ms": _percentile(intervals, 95),
        "inter_token_p99_ms": _percentile(intervals, 99),
        "server_timings": final_payload,
    }


def _background_gpu_processes() -> list[dict[str, str]]:
    command = [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    processes: list[dict[str, str]] = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) >= 3:
            processes.append({"pid": fields[0], "name": fields[1], "used_memory": fields[2]})
    return processes


def _gpu_snapshot() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.used,memory.free,temperature.gpu,clocks.sm,clocks.mem,pstate,power.draw,utilization.gpu,utilization.memory",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    fields = [field.strip() for field in completed.stdout.strip().split(",")]
    if len(fields) < 10:
        return {"available": False, "raw": completed.stdout.strip()}
    return {
        "available": True,
        "gpu": fields[0],
        "memory_used_mib": int(float(fields[1])),
        "memory_free_mib": int(float(fields[2])),
        "temperature_c": float(fields[3]),
        "sm_clock_mhz": float(fields[4]),
        "memory_clock_mhz": float(fields[5]),
        "power_state": fields[6],
        "power_draw_w": float(fields[7]),
        "gpu_utilization_percent": float(fields[8]),
        "memory_utilization_percent": float(fields[9]),
    }


def _server_command(args: argparse.Namespace, context: int, port: int) -> list[str]:
    return [
        args.server,
        "--model",
        args.model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--no-webui",
        "--ctx-size",
        str(context),
        "--gpu-layers",
        "999",
        "--device",
        "CUDA0",
        "--split-mode",
        "none",
        "--fit",
        "off",
        *desktop_safe_launch_args(),
        "--batch-size",
        str(args.batch_size),
        "--ubatch-size",
        str(args.ubatch_size),
        "--flash-attn",
        "on",
        "--cache-type-k",
        args.cache_type_k,
        "--cache-type-v",
        args.cache_type_v,
        "--parallel",
        "1",
        "--reasoning",
        "off",
        "--ctx-checkpoints",
        "0",
        "--cache-ram",
        "0",
        "--verbosity",
        "4",
    ]


def _stop_server(process: subprocess.Popen[str] | None) -> None:
    """Stop the exact llama-server process and any Windows child processes."""
    if process is None:
        return
    pid = process.pid
    _stop_process(process)
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            text=True,
            check=False,
        )


def _apply_timings(metrics: dict[str, Any]) -> None:
    timings = metrics.get("server_timings")
    if not isinstance(timings, dict):
        return
    prompt_tokens = timings.get("prompt_n")
    prompt_ms = timings.get("prompt_ms")
    generated_tokens = timings.get("predicted_n")
    decode_ms = timings.get("predicted_ms")
    if isinstance(prompt_tokens, (int, float)) and isinstance(prompt_ms, (int, float)):
        metrics["prompt_tokens"] = int(prompt_tokens)
        metrics["prompt_eval_ms"] = float(prompt_ms)
        metrics["prefill_tokens_per_second"] = float(
            timings.get("prompt_per_second", 0.0)
        )
    if isinstance(generated_tokens, (int, float)) and isinstance(decode_ms, (int, float)):
        metrics["generated_tokens"] = int(generated_tokens)
        metrics["decode_eval_ms"] = float(decode_ms)
        metrics["decode_tokens_per_second"] = float(
            timings.get("predicted_per_second", 0.0)
        )


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "stddev": None,
            "min": None,
            "max": None,
            "cv": None,
        }
    mean = statistics.fmean(values)
    stddev = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "count": len(values),
        "mean": mean,
        "median": statistics.median(values),
        "stddev": stddev,
        "min": min(values),
        "max": max(values),
        "cv": stddev / mean if mean else None,
    }


def _run_series(
    args: argparse.Namespace,
    context: int,
    work_dir: Path,
    hardware: dict[str, Any],
    on_record,
) -> list[dict[str, Any]]:
    occupancy_target = context if args.prompt_mode == "filled" else None
    if args.prompt_mode == "filled":
        context = context + args.decode_tokens
    tokenizer = str(Path(args.server).with_name("llama-tokenize.exe"))
    prompt_path = work_dir / f"p6_prompt_{args.prompt_mode}_c{context}.txt"
    if args.prompt_mode == "short":
        prompt = _short_prompt()
        prompt_path.write_text(prompt, encoding="utf-8", newline="\n")
        prompt_tokens_target = None
        prompt_tokens_local = None
    else:
        prompt, prompt_tokens_local, prompt_tokens_target = _filled_prompt(
            tokenizer, args.model, context, args.decode_tokens, prompt_path
        )
    prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    condition = (
        f"c{context}-{args.cache_type_k}-{args.cache_type_v}-{args.prompt_mode}"
        f"-b{args.batch_size}-u{args.ubatch_size}"
    )
    log_path = work_dir / f"p6_{condition}.log"
    port = _pick_port()
    background_processes = _background_gpu_processes()
    gpu_state_before = _gpu_snapshot()
    if (
        not args.allow_contended_gpu
        and gpu_state_before.get("available")
        and gpu_state_before.get("memory_used_mib", 0) > args.max_background_vram_mib
    ):
        guard = {
            "schema_version": 1,
            "benchmark_id": f"p6-{condition}-environment-guard",
            "benchmark_kind": "full_gpu_reproducibility_environment_guard",
            "synthetic": False,
            "timestamp_utc": _utc_now(),
            "config": {
                "model_file": Path(args.model).name,
                "configured_context_tokens": context,
                "prompt_mode": args.prompt_mode,
                "cache_type_k": args.cache_type_k,
                "cache_type_v": args.cache_type_v,
                "repeats_requested": args.repeats,
                "warmup_runs": args.warmup_runs,
                "max_background_vram_mib": args.max_background_vram_mib,
                "background_gpu_process_count": len(background_processes),
                "gpu_memory_used_mib_before_server": gpu_state_before.get("memory_used_mib"),
            },
            "hardware": hardware,
            "metrics": {"measurement_status": "not_run"},
            "quality_gate": {
                "status": "INCONCLUSIVE",
                "reason": "GPU environment is contended; no model process was started.",
            },
            "notes": [
                "Close GPU-heavy applications, then rerun with the same recipe.",
            ],
        }
        on_record(guard)
        return [guard]
    process = None
    server_log = log_path.open("w", encoding="utf-8", newline="\n")
    records: list[dict[str, Any]] = []
    try:
        assert_model_fits_host_ram(Path(args.model))
        process = subprocess.Popen(
            _server_command(args, context, port),
            stdin=subprocess.DEVNULL,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            text=True,
            creationflags=windows_launch_flags(),
        )
        _wait_health(port, process, min(args.timeout, 600))
        for _ in range(args.warmup_runs):
            _completion(port, prompt, min(args.decode_tokens, 32), args.timeout)
        for run_index in range(1, args.repeats + 1):
            gpu_path = work_dir / f"p6_{condition}_run{run_index}.csv"
            monitor = None
            stop_system = threading.Event()
            system_samples: list[dict[str, float | int]] = []
            system_thread = threading.Thread(
                target=_sample_system,
                args=(stop_system, system_samples),
                daemon=True,
            )
            started = time.perf_counter()
            failed_type = None
            try:
                monitor = _start_gpu_monitor(gpu_path)
                system_thread.start()
                try:
                    metrics = _completion(port, prompt, args.decode_tokens, args.timeout)
                except (urllib.error.URLError, TimeoutError, OSError, ConnectionError) as exc:
                    if process.poll() is None:
                        try:
                            metrics = _completion(port, prompt, args.decode_tokens, args.timeout)
                        except (
                            urllib.error.URLError,
                            TimeoutError,
                            OSError,
                            ConnectionError,
                        ) as retry_exc:
                            failed_type = type(retry_exc).__name__
                            metrics = {}
                    else:
                        failed_type = type(exc).__name__
                        metrics = {}
            finally:
                stop_system.set()
                system_thread.join(timeout=2)
                _stop_process(monitor)
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
            metrics.update(_parse_log(log_text))
            _apply_timings(metrics)
            prompt_tokens = int(metrics.get("prompt_tokens", prompt_tokens_local or 0))
            generated_tokens = int(metrics.get("generated_tokens", 0))
            buffers = metrics.get("model_buffers_mib") or {}
            offloaded = metrics.get("offloaded_layers")
            full_gpu = offloaded == "65/65"
            metrics.update(
                {
                    "measurement_status": "failed" if failed_type else "measured",
                    "error_type": failed_type,
                    "configured_context_tokens": context,
                    "occupancy_target_tokens": occupancy_target,
                    "tokens_in_cache_before_decode": prompt_tokens,
                    "final_context_occupancy": min(context, prompt_tokens + generated_tokens),
                    "prompt_tokens_local_tokenizer": prompt_tokens_local,
                    "prompt_target_tokens": prompt_tokens_target,
                    "wall_seconds": time.perf_counter() - started,
                    "gpu_telemetry": _parse_gpu(gpu_path),
                    "full_gpu": full_gpu,
                    "cpu_decoder_weight_offload": not full_gpu,
                    "host_mapped_model_mib": buffers.get("CPU_Mapped"),
                    "cuda_model_buffer_mib": buffers.get("CUDA0"),
                }
            )
            if system_samples:
                cpu = [
                    float(item["system_cpu_percent"])
                    for item in system_samples
                    if "system_cpu_percent" in item
                ]
                ram = [
                    int(item["available_ram_mib"])
                    for item in system_samples
                    if "available_ram_mib" in item
                ]
                metrics["system_cpu_max_percent"] = max(cpu) if cpu else None
                metrics["system_ram_min_available_mib"] = min(ram) if ram else None
            room = max(0, context - prompt_tokens)
            metrics["generation_capped_by_context"] = (
                generated_tokens < args.decode_tokens and room <= args.decode_tokens
            )
            record = {
                "schema_version": 1,
                "benchmark_id": f"p6-{condition}-run{run_index}",
                "benchmark_kind": "full_gpu_reproducibility_run",
                "synthetic": False,
                "timestamp_utc": _utc_now(),
                "config": {
                    "runner": "llama-server",
                    "model_file": Path(args.model).name,
                    "model_sha256": args.model_sha256,
                    "configured_context_tokens": context,
                    "occupancy_target_tokens": occupancy_target,
                    "prompt_mode": args.prompt_mode,
                    "decode_tokens_requested": args.decode_tokens,
                    "repeats": args.repeats,
                    "warmup_runs": args.warmup_runs,
                    "cache_type_k": args.cache_type_k,
                    "cache_type_v": args.cache_type_v,
                    "gpu_layers": 999,
                    "device": "CUDA0",
                    "fit": "off",
                    "load_mode": "dio",
                    "process_priority": "low",
                    "flash_attention_requested": True,
                    "logical_batch": args.batch_size,
                    "physical_batch": args.ubatch_size,
                    "batch_size": args.batch_size,
                    "ubatch_size": args.ubatch_size,
                    "seed": 42,
                    "temperature": 0.0,
                    "ctx_checkpoints": 0,
                    "cache_ram_mib": 0,
                    "prompt_sha256": prompt_sha,
                    "prompt_chars": len(prompt),
                    "run_index": run_index,
                    "background_gpu_process_count": len(background_processes),
                    "gpu_memory_used_mib_before_server": gpu_state_before.get("memory_used_mib"),
                },
                "hardware": hardware,
                "metrics": metrics,
                "quality_gate": {"status": "NOT_RUN", "reason": "P6 reproducibility only."},
                "notes": [
                    "Measured after one or more warmup requests.",
                    "cache_prompt=false; tokens_in_cache_before_decode means prompt tokens "
                    "processed into the fresh KV/state.",
                ],
            }
            records.append(record)
            print(
                f"{record['benchmark_id']} decode={metrics.get('decode_tokens_per_second')} "
                f"prompt={prompt_tokens} gen={generated_tokens} "
                f"occupancy={metrics.get('final_context_occupancy')} full_gpu={full_gpu}",
                flush=True,
            )
            on_record(record)
            if failed_type is not None and process.poll() is not None:
                break
    finally:
        _stop_server(process)
        server_log.close()
    def _eligible(record: dict[str, Any]) -> bool:
        metrics = record["metrics"]
        if metrics.get("measurement_status") != "measured" or metrics.get("full_gpu") is not True:
            return False
        generated = int(metrics.get("generated_tokens") or 0)
        prompt = int(metrics.get("prompt_tokens") or 0)
        room = max(1, context - prompt)
        needed = min(args.decode_tokens, room)
        return generated >= max(1, needed - 1)

    eligible = [record for record in records if _eligible(record)]
    decode_values = [
        float(record["metrics"]["decode_tokens_per_second"])
        for record in eligible
        if record["metrics"].get("decode_tokens_per_second")
    ]
    p50_values = [
        float(record["metrics"]["inter_token_p50_ms"])
        for record in eligible
        if record["metrics"].get("inter_token_p50_ms") is not None
    ]
    p95_values = [
        float(record["metrics"]["inter_token_p95_ms"])
        for record in eligible
        if record["metrics"].get("inter_token_p95_ms") is not None
    ]
    p99_values = [
        float(record["metrics"]["inter_token_p99_ms"])
        for record in eligible
        if record["metrics"].get("inter_token_p99_ms") is not None
    ]
    decode_summary = _summary(decode_values)
    summary = {
        "schema_version": 1,
        "benchmark_id": f"p6-{condition}-summary",
        "benchmark_kind": "full_gpu_reproducibility_summary",
        "synthetic": False,
        "timestamp_utc": _utc_now(),
        "config": {
            "model_file": Path(args.model).name,
            "model_sha256": args.model_sha256,
            "configured_context_tokens": context,
            "prompt_mode": args.prompt_mode,
            "prompt_sha256": prompt_sha,
            "prompt_tokens_local_tokenizer": prompt_tokens_local,
            "prompt_target_tokens": prompt_tokens_target,
            "cache_type_k": args.cache_type_k,
            "cache_type_v": args.cache_type_v,
            "repeats_requested": args.repeats,
            "warmup_runs": args.warmup_runs,
            "batch_size": args.batch_size,
            "ubatch_size": args.ubatch_size,
        },
        "hardware": hardware,
        "metrics": {
            "runs_completed": len(records),
            "decode_tok_s": decode_summary,
            "inter_token_p50_ms": _summary(p50_values),
            "inter_token_p95_ms": _summary(p95_values),
            "inter_token_p99_ms": _summary(p99_values),
            "eligible_full_gpu_runs": len(eligible),
            "prompt_tokens": _summary([
                float(record["metrics"]["prompt_tokens"])
                for record in eligible
                if record["metrics"].get("prompt_tokens") is not None
            ]),
            "final_context_occupancy": _summary([
                float(record["metrics"]["final_context_occupancy"])
                for record in eligible
                if record["metrics"].get("final_context_occupancy") is not None
            ]),
            "stability_cv_pass": (
                decode_summary.get("cv") is not None and decode_summary["cv"] <= 0.05
            ),
            "quality_gate": "NOT_RUN",
        },
        "quality_gate": {"status": "NOT_RUN", "reason": "P6 reproducibility only."},
        "notes": ["Headline speed is median decode tok/s, not the best run."],
    }
    records.append(summary)
    on_record(summary)
    return records


def main() -> int:
    args = _parser().parse_args()
    if (
        args.repeats <= 0
        or args.warmup_runs < 0
        or args.decode_tokens <= 0
        or args.batch_size <= 0
        or args.ubatch_size <= 0
    ):
        print(
            "repeats and decode-tokens must be positive; "
            "warmup-runs cannot be negative",
            file=sys.stderr,
        )
        return 2
    contexts = [int(item) for item in args.contexts.split(",") if item.strip()]
    work_dir = Path(args.work_dir)
    output = Path(args.output)
    work_dir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        assert_storage_safe(
            [args.server, args.model, args.output, args.work_dir],
            allow_removable=args.allow_removable_storage,
        )
    except StorageSafetyError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    hardware = _hardware()
    try:
        with _OutputLock(output):
            with output.open("a", encoding="utf-8", newline="\n") as handle:
                def on_record(record: dict[str, Any]) -> None:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                    handle.flush()

                for context in contexts:
                    try:
                        _run_series(args, context, work_dir, hardware, on_record)
                    except Exception as exc:
                        on_record(
                            {
                                "schema_version": 1,
                                "benchmark_id": f"p6-c{context}-series-error",
                                "benchmark_kind": "full_gpu_reproducibility_series_error",
                                "synthetic": False,
                                "timestamp_utc": _utc_now(),
                                "config": {
                                    "configured_context_tokens": context,
                                    "prompt_mode": args.prompt_mode,
                                },
                                "metrics": {
                                    "measurement_status": "failed",
                                    "error_type": type(exc).__name__,
                                },
                                "quality_gate": {
                                    "status": "NOT_RUN",
                                    "reason": "series stopped on a runtime error",
                                },
                                "notes": [
                                    "Earlier flushed runs in this file remain valid measurements."
                                ],
                            }
                        )
                        print(
                            f"context {context} stopped: {type(exc).__name__}",
                            file=sys.stderr,
                        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

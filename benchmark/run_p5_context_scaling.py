"""Measure deterministic llama.cpp context scaling with official llama-server."""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import math
import os
import re
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from exvram.storage import StorageSafetyError, assert_storage_safe

# Process creation flag. Applied before llama.cpp reads the GGUF, so the
# desktop keeps a core while weights are loading.
_BELOW_NORMAL_PRIORITY_CLASS = 0x00004000


def desktop_safe_launch_args() -> list[str]:
    """Keep the workstation responsive while a multi-gigabyte GGUF is opened.

    Default mmap fills the Windows standby list and evicts the desktop working
    set. Direct I/O skips that cache. Low priority and two reserved threads
    leave the UI a way to run during the load.
    """
    threads = max(1, (os.cpu_count() or 1) - 2)
    return [
        "--load-mode",
        "dio",
        "--prio",
        "-1",
        "--threads",
        str(threads),
        "--threads-batch",
        str(threads),
    ]


def windows_launch_flags() -> int:
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    if sys.platform == "win32":
        flags |= _BELOW_NORMAL_PRIORITY_CLASS
    return flags


def assert_model_fits_host_ram(model: Path) -> None:
    """Refuse a GGUF larger than physical RAM.

    Direct I/O keeps a model that fits from flushing the desktop cache. A file
    larger than RAM still forces the working set into the pagefile.
    """
    if sys.platform != "win32":
        return
    size = model.stat().st_size

    class _MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = _MemoryStatus()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return
    if size > status.ullTotalPhys:
        raise RuntimeError(
            f"{model.name} is {size} bytes and physical RAM is {status.ullTotalPhys} bytes. "
            "Opening it pages the workstation out."
        )


_PROMPT_RE = re.compile(r"prompt eval time\s*=\s*([0-9.]+) ms /\s*([0-9]+) tokens")
_DECODE_RE = re.compile(r"eval time\s*=\s*([0-9.]+) ms /\s*([0-9]+) tokens")
_RATE_RE = re.compile(
    r"\[\s*Prompt:\s*([0-9.]+)\s*t/s\s*\|\s*Generation:\s*([0-9.]+)\s*t/s\s*\]",
    re.IGNORECASE,
)
_LAYERS_RE = re.compile(r"offloaded\s+(\d+)\s*/\s*(\d+)\s+layers to GPU", re.IGNORECASE)
_MODEL_BUFFER_RE = re.compile(
    r"(?P<kind>CPU_Mapped|CUDA\d+) model buffer size\s*=\s*(?P<mib>[0-9.]+) MiB",
    re.IGNORECASE,
)
_RESOURCE_RE = re.compile(
    r"(?P<kind>CUDA\d+|CUDA_Host)\s+"
    r"(?P<resource>KV|RS|compute|output) buffer size\s*=\s*"
    r"(?P<mib>[0-9.]+) MiB",
    re.IGNORECASE,
)
_CONTEXT_RE = re.compile(r"n_ctx\s*=\s*(\d+)")
_BATCH_RE = re.compile(r"n_batch\s*=\s*(\d+)")
_UBATCH_RE = re.compile(r"n_ubatch\s*=\s*(\d+)")
_FLASH_RE = re.compile(r"flash_attn\s*=\s*(enabled|disabled)", re.IGNORECASE)
_UNIFIED_RE = re.compile(r"kv_unified\s*=\s*(true|false)", re.IGNORECASE)
_GRAPH_RE = re.compile(
    r"graph nodes\s*=\s*(\d+).*?graph splits\s*=\s*(\d+)",
    re.IGNORECASE | re.DOTALL,
)


class _FILETIME(ctypes.Structure):
    _fields_ = [("low", ctypes.c_uint32), ("high", ctypes.c_uint32)]


class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("length", ctypes.c_uint32),
        ("memory_load", ctypes.c_uint32),
        ("total_phys", ctypes.c_uint64),
        ("avail_phys", ctypes.c_uint64),
        ("total_page", ctypes.c_uint64),
        ("avail_page", ctypes.c_uint64),
        ("total_virtual", ctypes.c_uint64),
        ("avail_virtual", ctypes.c_uint64),
        ("avail_extended", ctypes.c_uint64),
    ]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True, help="path to llama-server.exe")
    parser.add_argument("--model", required=True, help="path to the GGUF artifact")
    parser.add_argument("--output", required=True, help="append one JSON record per context")
    parser.add_argument(
        "--work-dir",
        required=True,
        help="ignored directory for prompts/logs/telemetry",
    )
    parser.add_argument("--contexts", default="256,512,1024,2048,4096,8192")
    parser.add_argument("--cache-type-k", default="q4_0")
    parser.add_argument("--cache-type-v", default="q4_0")
    parser.add_argument("--decode-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--model-sha256", default=None)
    parser.add_argument("--allow-removable-storage", action="store_true")
    return parser


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _pick_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _prompt_text(context: int) -> str:
    if context <= 256:
        return "Explain quantized inference in one concise paragraph."
    target_prompt_tokens = max(32, context - 256)
    seed = (
        "This is deterministic context filler for ExVRAM Lab. Quantized inference "
        "trades numerical precision for lower memory traffic while preserving the "
        "same model computation. "
    )
    target_chars = target_prompt_tokens * 4
    body = (seed * math.ceil(target_chars / len(seed)))[:target_chars]
    return body + "\n\nNow explain the central idea in one concise paragraph."


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return float(statistics.quantiles(values, n=100, method="inclusive")[int(percentile) - 1])


def _system_times() -> tuple[int, int] | None:
    if os.name != "nt":
        return None
    idle = _FILETIME()
    kernel = _FILETIME()
    user = _FILETIME()
    if not ctypes.windll.kernel32.GetSystemTimes(
        ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)
    ):
        return None

    def value(item: _FILETIME) -> int:
        return (item.high << 32) | item.low

    return value(kernel) + value(user), value(idle)


def _system_memory() -> int | None:
    if os.name != "nt":
        return None
    state = _MEMORYSTATUSEX()
    state.length = ctypes.sizeof(_MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):
        return None
    return int(state.avail_phys // (1024 * 1024))


def _sample_system(stop: threading.Event, samples: list[dict[str, float | int]]) -> None:
    previous = _system_times()
    while not stop.wait(0.5):
        current = _system_times()
        sample: dict[str, float | int] = {"timestamp": time.time()}
        available = _system_memory()
        if available is not None:
            sample["available_ram_mib"] = available
        if previous is not None and current is not None:
            total_delta = current[0] - previous[0]
            idle_delta = current[1] - previous[1]
            if total_delta > 0:
                sample["system_cpu_percent"] = max(
                    0.0,
                    min(100.0, 100.0 * (1.0 - idle_delta / total_delta)),
                )
        samples.append(sample)
        previous = current


def _start_gpu_monitor(path: Path) -> subprocess.Popen[str]:
    handle = path.open("w", encoding="utf-8", newline="")
    command = [
        "nvidia-smi",
        "--query-gpu=timestamp,name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,clocks.sm,power.draw",
        "--format=csv,noheader,nounits",
        "-lms",
        "500",
    ]
    process = subprocess.Popen(
        command,
        stdout=handle,
        stderr=subprocess.DEVNULL,
        text=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    process._exvram_log_handle = handle  # type: ignore[attr-defined]
    return process


def _stop_process(process: subprocess.Popen[str] | None) -> None:
    if process is None:
        return
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=15)
    handle = getattr(process, "_exvram_log_handle", None)
    if handle is not None:
        handle.close()


def _wait_health(port: int, process: subprocess.Popen[str], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"llama-server exited with code {process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.HTTPError):
            pass
        time.sleep(0.5)
    raise TimeoutError("llama-server health check timed out")


def _stream_completion(
    port: int, prompt: str, decode_tokens: int, timeout: float
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
    intervals = [(right - left) * 1000.0 for left, right in zip(arrivals, arrivals[1:])]
    return {
        "stream_events": len(arrivals),
        "ttft_ms": (arrivals[0] - started) * 1000.0 if arrivals else None,
        "inter_token_p50_ms": _percentile(intervals, 50),
        "inter_token_p95_ms": _percentile(intervals, 95),
        "server_timings": final_payload,
    }


def _parse_gpu(path: Path) -> dict[str, Any]:
    rows: list[dict[str, float]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 9:
                continue
            try:
                rows.append(
                    {
                        "memory_total_mib": float(row[2]),
                        "memory_used_mib": float(row[3]),
                        "memory_free_mib": float(row[4]),
                        "gpu_utilization_percent": float(row[5]),
                        "memory_utilization_percent": float(row[6]),
                        "sm_clock_mhz": float(row[7]),
                        "power_w": float(row[8]),
                    }
                )
            except ValueError:
                continue
    if not rows:
        return {"samples": 0}
    return {
        "samples": len(rows),
        "peak_vram_used_mib": max(row["memory_used_mib"] for row in rows),
        "minimum_vram_free_mib": min(row["memory_free_mib"] for row in rows),
        "max_gpu_utilization_percent": max(row["gpu_utilization_percent"] for row in rows),
        "max_memory_utilization_percent": max(row["memory_utilization_percent"] for row in rows),
        "max_sm_clock_mhz": max(row["sm_clock_mhz"] for row in rows),
        "max_power_w": max(row["power_w"] for row in rows),
    }


def _parse_log(text: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    prompt_matches = list(_PROMPT_RE.finditer(text))
    decode_matches = list(_DECODE_RE.finditer(text))
    prompt = prompt_matches[-1] if prompt_matches else None
    decode = decode_matches[-1] if decode_matches else None
    rate = _RATE_RE.search(text)
    if prompt:
        metrics["prompt_eval_ms"] = float(prompt.group(1))
        metrics["prompt_tokens"] = int(prompt.group(2))
        metrics["prefill_tokens_per_second"] = int(prompt.group(2)) / (
            float(prompt.group(1)) / 1000.0
        )
    if decode:
        metrics["decode_eval_ms"] = float(decode.group(1))
        metrics["generated_tokens"] = int(decode.group(2))
        metrics["decode_tokens_per_second"] = int(decode.group(2)) / (
            float(decode.group(1)) / 1000.0
        )
    if rate:
        metrics["prefill_tokens_per_second"] = float(rate.group(1))
        metrics["decode_tokens_per_second"] = float(rate.group(2))
    layers = _LAYERS_RE.search(text)
    if layers:
        metrics["offloaded_layers"] = f"{layers.group(1)}/{layers.group(2)}"
    model_buffers = {
        match.group("kind"): float(match.group("mib")) for match in _MODEL_BUFFER_RE.finditer(text)
    }
    if model_buffers:
        metrics["model_buffers_mib"] = model_buffers
    resources = {
        f"{match.group('kind')}_{match.group('resource').lower()}_mib": float(match.group("mib"))
        for match in _RESOURCE_RE.finditer(text)
    }
    if resources:
        metrics["resource_buffers_mib"] = resources
    for name, expression in (
        ("actual_context_tokens", _CONTEXT_RE),
        ("logical_batch", _BATCH_RE),
        ("physical_batch", _UBATCH_RE),
    ):
        match = expression.search(text)
        if match:
            metrics[name] = int(match.group(1))
    flash = _FLASH_RE.search(text)
    unified = _UNIFIED_RE.search(text)
    if flash:
        metrics["flash_attention"] = flash.group(1).lower() == "enabled"
    if unified:
        metrics["kv_unified"] = unified.group(1).lower() == "true"
    graph = _GRAPH_RE.search(text)
    if graph:
        metrics["graph_nodes"] = int(graph.group(1))
        metrics["graph_splits"] = int(graph.group(2))
    return metrics


def _hardware() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,compute_cap,memory.total,power.limit,pcie.link.gen.current,pcie.link.width.current",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    fields = [field.strip() for field in completed.stdout.strip().split(",")]
    if len(fields) < 7:
        return {"nvidia_smi": False}
    return {
        "gpu": fields[0],
        "driver": fields[1],
        "compute_capability": fields[2],
        "vram_total_mib": int(float(fields[3])),
        "power_limit_w": float(fields[4]),
        "pcie_link_gen_current": int(float(fields[5])),
        "pcie_link_width_current": int(float(fields[6])),
    }


def _run_one(
    args: argparse.Namespace,
    context: int,
    work_dir: Path,
    hardware: dict[str, Any],
) -> dict[str, Any]:
    prompt = _prompt_text(context)
    prompt_path = work_dir / f"p5_prompt_c{context}.txt"
    prompt_path.write_text(prompt, encoding="utf-8", newline="\n")
    prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    log_path = work_dir / f"p5_context_c{context}_{args.cache_type_k}_{args.cache_type_v}.log"
    gpu_path = work_dir / f"p5_context_c{context}_{args.cache_type_k}_{args.cache_type_v}.csv"
    port = _pick_port()
    command = [
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
        "32",
        "--ubatch-size",
        "1",
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
        "--verbosity",
        "3",
    ]
    base: dict[str, Any] = {
        "schema_version": 1,
        "benchmark_id": f"p5-context-c{context}-{args.cache_type_k}-{args.cache_type_v}",
        "benchmark_kind": "full_gpu_context_scaling",
        "synthetic": False,
        "timestamp_utc": _utc_now(),
        "config": {
            "runner": "llama-server",
            "model_file": Path(args.model).name,
            "requested_context_tokens": context,
            "decode_tokens_requested": args.decode_tokens,
            "cache_type_k": args.cache_type_k,
            "cache_type_v": args.cache_type_v,
            "gpu_layers": 999,
            "device": "CUDA0",
            "fit": "off",
            "flash_attention_requested": True,
            "warmup": True,
            "logical_batch": 32,
            "physical_batch": 1,
            "seed": 42,
            "temperature": 0.0,
            "prompt_sha256": prompt_sha,
            "prompt_chars": len(prompt),
        },
        "hardware": hardware,
        "metrics": {},
        "quality_gate": {"status": "NOT_RUN", "reason": "Context scaling only."},
        "notes": [],
    }
    server_log = log_path.open("w", encoding="utf-8", newline="\n")
    server: subprocess.Popen[str] | None = None
    monitor: subprocess.Popen[str] | None = None
    stop_system = threading.Event()
    system_samples: list[dict[str, float | int]] = []
    system_thread = threading.Thread(
        target=_sample_system,
        args=(stop_system, system_samples),
        daemon=True,
    )
    started = time.perf_counter()
    try:
        assert_model_fits_host_ram(Path(args.model))
        server = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            text=True,
            creationflags=windows_launch_flags(),
        )
        monitor = _start_gpu_monitor(gpu_path)
        system_thread.start()
        _wait_health(port, server, min(args.timeout, 600))
        stream = _stream_completion(port, prompt, args.decode_tokens, args.timeout)
        elapsed = time.perf_counter() - started
        base["metrics"].update(stream)
        base["metrics"]["wall_seconds"] = elapsed
        base["notes"].append("Streaming events are used for TTFT and inter-token p50/p95.")
    except (OSError, RuntimeError, TimeoutError, urllib.error.URLError) as exc:
        base["notes"].append(f"run_failed: {type(exc).__name__}: {exc}")
        base["quality_gate"] = {"status": "INCONCLUSIVE", "reason": "Runtime did not complete."}
    finally:
        if server is not None:
            _stop_process(server)
        stop_system.set()
        system_thread.join(timeout=2)
        _stop_process(monitor)
        server_log.close()
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    base["metrics"].update(_parse_log(log_text))
    server_timings = base["metrics"].get("server_timings")
    if isinstance(server_timings, dict):
        prompt_tokens = server_timings.get("prompt_n")
        prompt_ms = server_timings.get("prompt_ms")
        generated_tokens = server_timings.get("predicted_n")
        decode_ms = server_timings.get("predicted_ms")
        if isinstance(prompt_tokens, (int, float)) and isinstance(prompt_ms, (int, float)):
            base["metrics"]["prompt_tokens"] = int(prompt_tokens)
            base["metrics"]["prompt_eval_ms"] = float(prompt_ms)
            base["metrics"]["prefill_tokens_per_second"] = float(
                server_timings.get("prompt_per_second", 0.0)
            )
        if isinstance(generated_tokens, (int, float)) and isinstance(decode_ms, (int, float)):
            base["metrics"]["generated_tokens"] = int(generated_tokens)
            base["metrics"]["decode_eval_ms"] = float(decode_ms)
            base["metrics"]["decode_tokens_per_second"] = float(
                server_timings.get("predicted_per_second", 0.0)
            )
    base["metrics"]["gpu_telemetry"] = _parse_gpu(gpu_path) if gpu_path.exists() else {"samples": 0}
    if system_samples:
        cpu_values = [
            float(item["system_cpu_percent"])
            for item in system_samples
            if "system_cpu_percent" in item
        ]
        ram_values = [
            int(item["available_ram_mib"])
            for item in system_samples
            if "available_ram_mib" in item
        ]
        base["metrics"]["system_cpu_max_percent"] = max(cpu_values) if cpu_values else None
        base["metrics"]["system_ram_min_available_mib"] = min(ram_values) if ram_values else None
    base["metrics"]["log_path"] = str(log_path)
    base["metrics"]["gpu_telemetry_path"] = str(gpu_path)
    if args.model_sha256:
        base["config"]["model_sha256"] = args.model_sha256
    return base


def main() -> int:
    args = _parser().parse_args()
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
    exit_code = 0
    with output.open("a", encoding="utf-8", newline="\n") as handle:
        for context in contexts:
            record = _run_one(args, context, work_dir, hardware)
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            print(json.dumps(record, indent=2, sort_keys=True))
            if "run_failed" in " ".join(record.get("notes", [])):
                exit_code = 2
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

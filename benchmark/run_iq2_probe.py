"""Reproduce four stored IQ2 math ids. Does not start a server without --launch."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from exvram.iq2_loops import PROBE_IDS, gsm8k_prompt, loop_shape
from exvram.p9_eval import gsm8k_correct, sha256_text

DATASET = Path("work/p9/datasets/gsm8k_test.jsonl")
STORED = Path("experiments/results/p9_qwen35_2b_iq2.jsonl")
OUTPUT = Path("experiments/results/p9_iq2_probe.jsonl")
MODEL = Path("models/qwen35-2b/Qwen3.5-2B-UD-IQ2_XXS.gguf")
SERVER = Path(".cache/llama/runtime-b10982-cuda13.4/llama-server.exe")


def prompts() -> list[dict[str, object]]:
    stored = {
        row["sample_id"]: row
        for row in (
            json.loads(line)
            for line in STORED.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if row.get("sample_id") in PROBE_IDS
    }
    lines = DATASET.read_text(encoding="utf-8").splitlines()
    items = []
    for sample_id in PROBE_IDS:
        question = json.loads(lines[int(sample_id.split(":")[1])])["question"]
        prompt = gsm8k_prompt(question)
        row = stored[sample_id]
        if sha256_text(prompt) != row["prompt_sha256"]:
            raise SystemExit(f"{sample_id} prompt does not match the stored row")
        items.append(
            {
                "sample_id": sample_id,
                "prompt": prompt,
                "gold": row["gold"],
                "stored_correct": row["correct"],
                "stored_predicted_n": (row.get("timings") or {}).get("predicted_n"),
            }
        )
    return items


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    items = prompts()
    if not args.launch:
        print(json.dumps({"launch": False, "ids": [item["sample_id"] for item in items]}))
        return 0
    if _server_running():
        raise SystemExit("llama-server is already running")
    if _ram_available_mib() < 2048:
        raise SystemExit("available RAM is below 2048 MiB")
    _run(items)
    return 0


def _run(items: list[dict[str, object]]) -> None:
    from benchmark.run_p5_context_scaling import _pick_port, _wait_health, windows_launch_flags
    from benchmark.run_p6_reproducibility import _server_command, _stop_server
    from benchmark.run_p9_quality import _erase_slot, _limit_threads, _post_json

    port = _pick_port()
    command = _limit_threads(
        _server_command(
            argparse.Namespace(
                server=str(SERVER),
                model=str(MODEL),
                batch_size=32,
                ubatch_size=1,
                cache_type_k="q4_0",
                cache_type_v="q4_0",
            ),
            8448,
            port,
        ),
        4,
    )
    log_path = Path("work/p9-iq2-probe/p9_server.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
        creationflags=windows_launch_flags(),
    )
    try:
        _wait_health(port, process, timeout=600)
        for item in items:
            _erase_slot(port)
            data = _post_json(
                port,
                "/v1/chat/completions",
                {
                    "messages": [{"role": "user", "content": item["prompt"]}],
                    "max_tokens": 768,
                    "temperature": 0,
                    "seed": 42,
                    "stream": False,
                    "cache_prompt": False,
                },
                300,
            )
            text = str(data["choices"][0]["message"].get("content") or "")
            timings = data.get("timings") if isinstance(data.get("timings"), dict) else {}
            row = {
                "protocol": "p9-2026-10-02-1",
                "record_type": "probe",
                "sample_id": item["sample_id"],
                "measurement_status": "measured",
                "prediction": text,
                "correct": gsm8k_correct(text, str(item["gold"])),
                "loop_shape": loop_shape(text),
                "stored_correct": item["stored_correct"],
                "stored_predicted_n": item["stored_predicted_n"],
                "timings": timings,
                "decoding": {"temperature": 0, "seed": 42, "n_predict": 768, "mode": "chat"},
                "server_extra": [],
            }
            _append(row)
            print(
                f"{item['sample_id']} correct={row['correct']} shape={row['loop_shape']}",
                flush=True,
            )
    finally:
        _stop_server(process)
        handle.close()


def _append(row: dict[str, object]) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _server_running() -> bool:
    if sys.platform != "win32":
        return False
    completed = subprocess.run(
        ["tasklist", "/FI", "IMAGENAME eq llama-server.exe", "/NH"],
        capture_output=True,
        text=True,
        check=False,
    )
    return "llama-server.exe" in completed.stdout


def _ram_available_mib() -> int:
    if sys.platform != "win32":
        return 2048
    import ctypes

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
        return 0
    return int(status.ullAvailPhys // (1024 * 1024))


if __name__ == "__main__":
    sys.exit(main())

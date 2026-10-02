"""Run the locked P9 quality protocol against one llama-server model.

Speed is recorded only as a sanity field. This runner does not tune it.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from exvram.p9_eval import (
    BASE_IQ2_BYTES,
    BASE_IQ2_SHA256,
    CODING_STOP,
    DATASETS,
    DECODE_SEED,
    LONG_CONTEXT_LENGTHS,
    LONG_CONTEXT_PER_LENGTH,
    N_PREDICT,
    PROTOCOL_ID,
    REASONING_PER_TASK,
    SAMPLE_COUNTS,
    TEMPERATURE,
    accuracy,
    aggregate_score,
    assemble_humaneval,
    bbh_correct,
    choice_correct,
    gsm8k_correct,
    instruction_correct,
    instruction_eligible,
    needle_code,
    needle_correct,
    python_program_passes,
    quality_verdict,
    select_ids,
    sha256_text,
    truthfulqa_gold_letter,
)
from exvram.storage import assert_storage_safe

try:
    from run_p5_context_scaling import (
        _parse_log,
        _pick_port,
        _wait_health,
        assert_model_fits_host_ram,
        windows_launch_flags,
    )
    from run_p6_reproducibility import _gpu_snapshot, _server_command, _stop_server
except ModuleNotFoundError:
    from benchmark.run_p5_context_scaling import (
        _parse_log,
        _pick_port,
        _wait_health,
        assert_model_fits_host_ram,
        windows_launch_flags,
    )
    from benchmark.run_p6_reproducibility import _gpu_snapshot, _server_command, _stop_server

LOCAL_NAMES = {
    "bbh_logical_deduction_three_objects": "bbh_logical_deduction_three_objects.json",
    "bbh_tracking_shuffled_objects_three_objects": (
        "bbh_tracking_shuffled_objects_three_objects.json"
    ),
    "bbh_date_understanding": "bbh_date_understanding.json",
    "gsm8k_test": "gsm8k_test.jsonl",
    "humaneval": "humaneval.jsonl.gz",
    "truthfulqa": "truthfulqa.csv",
    "ifeval_input": "ifeval_input.jsonl",
}
FILLER = "The decoy note says nothing useful. "
QUESTION = "\n\nQuestion: What is the secret code? Reply with the code only."
POSITIONS = ("early", "middle", "late")
POSITION_FRACTION = {"early": 0.10, "middle": 0.50, "late": 0.90}


def _dataset_file(cache: Path, key: str) -> Path:
    destination = cache / LOCAL_NAMES[key]
    payload = DATASETS[key]
    if destination.is_file():
        digest = _sha256_file(destination)
        if digest != payload["sha256"]:
            raise RuntimeError(f"{destination.name} sha256 {digest} != {payload['sha256']}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(payload["url"], destination)  # noqa: S310
    digest = _sha256_file(destination)
    if digest != payload["sha256"]:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"downloaded {key} sha256 {digest} != {payload['sha256']}")
    return destination


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _item(
    *,
    sample_id: str,
    category: str,
    task: str,
    prompt: str,
    gold: str,
    mode: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "sample_id": sample_id,
        "category": category,
        "task": task,
        "prompt": prompt,
        "gold": gold,
        "mode": mode,
        "n_predict": N_PREDICT[category],
    }
    if extra:
        row.update(extra)
    return row


def load_static_items(cache: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    reasoning_files = (
        ("bbh_logical_deduction_three_objects", "logical_deduction_three_objects"),
        ("bbh_tracking_shuffled_objects_three_objects", "tracking_shuffled_objects_three_objects"),
        ("bbh_date_understanding", "date_understanding"),
    )
    for key, task in reasoning_files:
        document = json.loads(_dataset_file(cache, key).read_text(encoding="utf-8"))
        examples = document["examples"]
        identifiers = [f"{task}:{index}" for index in range(len(examples))]
        for sample_id in select_ids(identifiers, REASONING_PER_TASK):
            index = int(sample_id.rsplit(":", 1)[1])
            example = examples[index]
            prompt = (
                example["input"].rstrip()
                + "\nAnswer with the final answer only, with no explanation."
            )
            items.append(
                _item(
                    sample_id=sample_id,
                    category="reasoning",
                    task=task,
                    prompt=prompt,
                    gold=str(example["target"]),
                    mode="chat",
                )
            )

    math_rows = [
        json.loads(line)
        for line in _dataset_file(cache, "gsm8k_test").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    math_ids = [f"gsm8k:{index}" for index in range(len(math_rows))]
    for sample_id in select_ids(math_ids, SAMPLE_COUNTS["math"]):
        row = math_rows[int(sample_id.rsplit(":", 1)[1])]
        prompt = (
            row["question"].rstrip()
            + "\nSolve the problem. Put the final number on its own line as #### <number>."
        )
        items.append(
            _item(
                sample_id=sample_id,
                category="math",
                task="gsm8k",
                prompt=prompt,
                gold=row["answer"],
                mode="chat",
            )
        )

    with _dataset_file(cache, "truthfulqa").open(encoding="utf-8", newline="") as handle:
        factual_rows = list(csv.DictReader(handle))
    by_question = {row["Question"]: row for row in factual_rows}
    for question in select_ids(list(by_question), SAMPLE_COUNTS["factual"]):
        row = by_question[question]
        gold = truthfulqa_gold_letter(question)
        best, wrong = row["Best Answer"], row["Best Incorrect Answer"]
        choice_a, choice_b = (best, wrong) if gold == "A" else (wrong, best)
        prompt = (
            f"Question: {question}\nA. {choice_a}\nB. {choice_b}\n"
            "Which answer is true? Reply with A or B only."
        )
        items.append(
            _item(
                sample_id=f"truthfulqa:{sha256_text(question)[:16]}",
                category="factual",
                task="truthfulqa_binary_v1",
                prompt=prompt,
                gold=gold,
                mode="chat",
            )
        )

    instruction_rows = [
        json.loads(line)
        for line in _dataset_file(cache, "ifeval_input").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    eligible = [row for row in instruction_rows if instruction_eligible(row["instruction_id_list"])]
    instruction_ids = [f"ifeval:{row['key']}" for row in eligible]
    for sample_id in select_ids(instruction_ids, SAMPLE_COUNTS["instruction"]):
        row = next(item for item in eligible if f"ifeval:{item['key']}" == sample_id)
        items.append(
            _item(
                sample_id=sample_id,
                category="instruction",
                task="ifeval_subset_v1",
                prompt=row["prompt"],
                gold=row["instruction_id_list"][0],
                mode="chat",
                extra={"kwargs": row["kwargs"][0] if row["kwargs"] else {}},
            )
        )

    with gzip.open(_dataset_file(cache, "humaneval"), "rt", encoding="utf-8") as handle:
        coding_rows = [json.loads(line) for line in handle if line.strip()]
    by_task = {row["task_id"]: row for row in coding_rows}
    for task_id in select_ids(list(by_task), SAMPLE_COUNTS["coding"]):
        row = by_task[task_id]
        items.append(
            _item(
                sample_id=task_id,
                category="coding",
                task="humaneval",
                prompt=row["prompt"],
                gold=row["entry_point"],
                mode="completion",
                extra={"test": row["test"], "entry_point": row["entry_point"]},
            )
        )
    return items


def _post_json(port: int, path: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _token_count(port: int, text: str) -> int:
    data = _post_json(port, "/tokenize", {"content": text}, timeout=60)
    tokens = data.get("tokens")
    if not isinstance(tokens, list):
        raise RuntimeError("tokenize response has no tokens")
    return len(tokens)


def _needle_prompt(port: int, index: int, length: int) -> tuple[str, str, str, int]:
    code = needle_code(index)
    needle = f"The secret code is {code}. "
    position = POSITIONS[index % len(POSITIONS)]
    fraction = POSITION_FRACTION[position]
    low, high = 1, max(2, length)
    best_prompt = needle + QUESTION
    best_count = _token_count(port, best_prompt)
    for _ in range(16):
        if low > high:
            break
        repeats = (low + high) // 2
        body = FILLER * repeats
        cut = min(len(body), max(0, int(len(body) * fraction)))
        prompt = body[:cut] + needle + body[cut:] + QUESTION
        count = _token_count(port, prompt)
        if abs(count - length) < abs(best_count - length):
            best_prompt, best_count = prompt, count
        if abs(count - length) <= 8:
            break
        if count > length:
            high = repeats - 1
        else:
            low = repeats + 1
    return best_prompt, code, position, best_count


def _generate(port: int, item: dict[str, Any]) -> dict[str, Any]:
    timeout = 900 if item["category"] == "long_context" else 180
    if item["mode"] == "completion":
        payload = {
            "prompt": item["prompt"],
            "n_predict": item["n_predict"],
            "temperature": TEMPERATURE,
            "seed": DECODE_SEED,
            "stream": False,
            "cache_prompt": False,
            "ignore_eos": False,
            "stop": CODING_STOP,
        }
        data = _post_json(port, "/completion", payload, timeout)
        text = str(data.get("content") or "")
    else:
        payload = {
            "messages": [{"role": "user", "content": item["prompt"]}],
            "max_tokens": item["n_predict"],
            "temperature": TEMPERATURE,
            "seed": DECODE_SEED,
            "stream": False,
            "cache_prompt": False,
        }
        data = _post_json(port, "/v1/chat/completions", payload, timeout)
        text = str(data["choices"][0]["message"].get("content") or "")
    timings = data.get("timings") if isinstance(data.get("timings"), dict) else {}
    return {"text": text, "timings": timings}


def _score(item: dict[str, Any], prediction: str) -> bool:
    if item["category"] == "reasoning":
        return bbh_correct(prediction, item["gold"])
    if item["category"] == "math":
        return gsm8k_correct(prediction, item["gold"])
    if item["category"] == "factual":
        return choice_correct(prediction, item["gold"])
    if item["category"] == "instruction":
        return instruction_correct(prediction, item["gold"], item.get("kwargs") or {})
    if item["category"] == "coding":
        source = assemble_humaneval(item["prompt"], prediction, item["test"], item["entry_point"])
        return python_program_passes(source)
    if item["category"] == "long_context":
        return needle_correct(prediction, item["gold"])
    raise ValueError(item["category"])


def _read_done(paths: list[Path]) -> set[str]:
    done: set[str] = set()
    for path in paths:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("measurement_status") == "measured" and row.get("sample_id"):
                done.add(str(row["sample_id"]))
    return done


def _append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores: dict[str, float | None] = {}
    for category in ("reasoning", "math", "coding", "factual", "instruction"):
        flags = [
            bool(row["correct"])
            for row in rows
            if row.get("category") == category and row.get("measurement_status") == "measured"
        ]
        scores[f"{category}_score"] = accuracy(flags)
    length_scores = []
    for length in LONG_CONTEXT_LENGTHS:
        flags = [
            bool(row["correct"])
            for row in rows
            if row.get("category") == "long_context"
            and row.get("context_target") == length
            and row.get("measurement_status") == "measured"
        ]
        length_score = accuracy(flags)
        scores[f"long_context_{length}"] = length_score
        if length_score is not None:
            length_scores.append(length_score)
    if len(length_scores) == len(LONG_CONTEXT_LENGTHS):
        scores["long_context_score"] = sum(length_scores) / len(LONG_CONTEXT_LENGTHS)
    else:
        scores["long_context_score"] = None
    mapped = {key: scores.get(key) for key in (
        "reasoning_score",
        "math_score",
        "coding_score",
        "factual_score",
        "instruction_score",
        "long_context_score",
    )}
    return {
        "protocol": PROTOCOL_ID,
        "record_type": "summary",
        "scores": scores,
        "aggregate": aggregate_score(mapped),
        "base_quality_verdict": quality_verdict(mapped, None),
        "reference_kind": "UNAVAILABLE",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--model-role", default="base_iq2_xxs")
    parser.add_argument("--output", default="experiments/results/p9_base_quality.jsonl")
    parser.add_argument("--long-output", default="experiments/results/p9_long_context.jsonl")
    parser.add_argument("--cache", default="work/p9/datasets")
    parser.add_argument("--work-dir", default="work/p9")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--ubatch-size", type=int, default=1)
    parser.add_argument("--cache-type-k", default="q4_0")
    parser.add_argument("--cache-type-v", default="q4_0")
    parser.add_argument("--context", type=int, default=8448)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-contended-gpu", action="store_true")
    args = parser.parse_args()

    model = Path(args.model)
    assert_storage_safe([model])
    assert_model_fits_host_ram(model)
    if args.model_role == "base_iq2_xxs":
        if args.model_sha256 != BASE_IQ2_SHA256 or model.stat().st_size != BASE_IQ2_BYTES:
            raise SystemExit("base IQ2 file does not match the locked sha256 and byte size")

    cache = Path(args.cache)
    items = load_static_items(cache)
    counts: dict[str, int] = {}
    for item in items:
        counts[item["category"]] = counts.get(item["category"], 0) + 1
    expected = {key: value for key, value in SAMPLE_COUNTS.items() if key != "long_context"}
    if counts != expected:
        raise SystemExit(f"protocol sample counts {counts} != {expected}")
    if args.dry_run:
        print(json.dumps({"protocol": PROTOCOL_ID, "counts": counts, "long_context": "runtime"}))
        return 0

    output = Path(args.output)
    long_output = Path(args.long_output)
    done = _read_done([output, long_output])
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    gpu = _gpu_snapshot()
    background_mib = gpu.get("memory_used_mib", 0)
    if not args.allow_contended_gpu and gpu.get("available") and background_mib > 1024:
        raise SystemExit(f"background VRAM {gpu.get('memory_used_mib')} MiB is above 1024")

    log_path = work / "p9_server.log"
    log_handle = log_path.open("w", encoding="utf-8")
    port = _pick_port()
    command = _server_command(args, args.context, port)
    import subprocess

    process = subprocess.Popen(
        command,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        creationflags=windows_launch_flags(),
    )
    measured: list[dict[str, Any]] = []
    try:
        _wait_health(port, process, timeout=600)
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        parsed = _parse_log(log_text)
        full_gpu = parsed.get("offloaded_layers") == "65/65"
        pending = [item for item in items if item["sample_id"] not in done]
        for item in pending:
            _run_item(args, port, item, full_gpu, output, measured)
        for length in LONG_CONTEXT_LENGTHS:
            for index in range(LONG_CONTEXT_PER_LENGTH):
                sample_id = f"needle:{length}:{index}"
                if sample_id in done:
                    continue
                prompt, code, position, count = _needle_prompt(port, index, length)
                item = _item(
                    sample_id=sample_id,
                    category="long_context",
                    task="synthetic_needle_v1",
                    prompt=prompt,
                    gold=code,
                    mode="chat",
                    extra={
                        "context_target": length,
                        "needle_position": position,
                        "prompt_tokens_local": count,
                    },
                )
                _run_item(args, port, item, full_gpu, long_output, measured)
    finally:
        _stop_server(process)
        log_handle.close()

    rows = []
    for path in (output, long_output):
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            rows.extend(json.loads(line) for line in text.splitlines() if line.strip())
    summary = _summary(rows)
    summary["model_role"] = args.model_role
    summary["model_sha256"] = args.model_sha256
    summary["full_gpu"] = any(row.get("full_gpu") for row in rows)
    _append(output, summary)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


def _run_item(
    args: argparse.Namespace,
    port: int,
    item: dict[str, Any],
    full_gpu: bool,
    output: Path,
    measured: list[dict[str, Any]],
) -> None:
    record: dict[str, Any] = {
        "protocol": PROTOCOL_ID,
        "record_type": "sample",
        "model_role": args.model_role,
        "model_sha256": args.model_sha256,
        "model_bytes": Path(args.model).stat().st_size,
        "category": item["category"],
        "task": item["task"],
        "sample_id": item["sample_id"],
        "gold": item["gold"],
        "prompt_sha256": sha256_text(item["prompt"]),
        "context_target": item.get("context_target"),
        "needle_position": item.get("needle_position"),
        "prompt_tokens": item.get("prompt_tokens_local"),
        "full_gpu": full_gpu,
        "decoding": {
            "temperature": TEMPERATURE,
            "seed": DECODE_SEED,
            "n_predict": item["n_predict"],
            "system_prompt": None,
            "mode": item["mode"],
        },
    }
    try:
        generated = _generate(port, item)
        prediction = generated["text"]
        record.update(
            {
                "measurement_status": "measured",
                "prediction": prediction,
                "correct": _score(item, prediction),
                "timings": generated["timings"],
            }
        )
    except (
        urllib.error.URLError,
        TimeoutError,
        OSError,
        KeyError,
        RuntimeError,
        json.JSONDecodeError,
    ) as exc:
        record.update(
            {
                "measurement_status": "failed",
                "error_type": type(exc).__name__,
                "correct": None,
            }
        )
    _append(output, record)
    measured.append(record)
    status = record["measurement_status"]
    print(f"{item['sample_id']} {status} correct={record.get('correct')}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    sys.exit(main())

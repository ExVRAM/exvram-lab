"""LONGTASK-V1 checkers. This module does not start a model."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Mapping

PROTOCOL_ID = "longtask-v1"
MODEL_RUN = "NOT_RUN"
CONTEXT_TOKENS = (2048, 4096, 8192)

_UNSAFE_CODE = re.compile(
    r"\b(?:os|subprocess|socket|shutil|ctypes|winreg|pathlib|requests)\b"
    r"|__import__|\beval\s*\(|\bexec\s*\(|\bcompile\s*\(|\bopen\s*\("
)

DOCUMENT = """\
Permit P-17 covers the North Hall.
The original capacity was 120 people. Amendment A closes the east fire exit, so the capacity is 96.
The fee is 40 per person at the amended capacity and is due 2026-11-02.
A 10 percent discount applies only when the fee is paid on or before 2026-10-28.
The noise curfew is 22:00, except the rehearsal on 2026-11-01, which may continue until 23:30.
The vendor is Lumen Audio. The invoice must name permit P-17 and must not include catering.
The deposit is 500. It stays refundable because the east exit is closed and therefore is not used.
The contact is ada@north.example. The retired address ada@hall.example is not the contact.
Load-in starts at 16:00. The stage plot lists 4 speakers and 2 returns.
Cancellation before 2026-10-30 returns the deposit and half the fee.
After that date the deposit is kept.
"""

DOCUMENT_LINES = (
    ("PERMIT", "P-17"),
    ("CAPACITY", "96"),
    ("FEE", "3840"),
    ("EARLY_FEE", "3456"),
    ("DUE", "2026-11-02"),
    ("EARLY_PAY", "2026-10-28"),
    ("CURFEW", "22:00"),
    ("REHEARSAL_UNTIL", "23:30"),
    ("REHEARSAL_DATE", "2026-11-01"),
    ("VENDOR", "Lumen Audio"),
    ("CATERING", "no"),
    ("DEPOSIT", "500"),
    ("DEPOSIT_REFUNDABLE", "yes"),
    ("CONTACT", "ada@north.example"),
    ("LOAD_IN", "16:00"),
    ("SPEAKERS", "4"),
    ("RETURNS", "2"),
    ("CANCEL_BEFORE", "2026-10-30"),
)

ROUTE_LINES = (
    ("CANCELLED", "C"),
    ("A_QTY", "10"),
    ("B_QTY", "12"),
    ("C_QTY", "0"),
    ("TRUCK", "15"),
    ("RULE", "smallest-first"),
    ("TRIP1", "A"),
    ("TRIP2", "B"),
    ("TRIP_COUNT", "2"),
    ("LEFTOVER", "0"),
    ("C_SHIPPED", "no"),
    ("OLD_TRUCK_USED", "no"),
    ("LARGEST_FIRST_USED", "no"),
    ("UNLOADED_A", "10"),
    ("UNLOADED_B", "12"),
    ("FINAL", "0"),
)

LEDGER_FILES = ("ledger/money.py", "ledger/book.py", "ledger/report.py")


def _labeled(text: str) -> dict[str, str]:
    found: dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        found[key.strip().upper()] = value.strip()
    return found


def score_labeled(text: str, expected: tuple[tuple[str, str], ...]) -> dict[str, bool]:
    found = _labeled(text)
    return {key: found.get(key) == value for key, value in expected}


def _ledger_runner() -> str:
    return r"""
import importlib.util
import json
from pathlib import Path

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

root = Path(__file__).resolve().parent
money = load(root / "ledger" / "money.py", "money")
book_mod = load(root / "ledger" / "book.py", "book")
report = load(root / "ledger" / "report.py", "report")
checks = {}
checks["parse_plain"] = money.parse_cents("12.30") == 1230
checks["parse_grouped"] = money.parse_cents("$1,000.00") == 100000
checks["parse_negative"] = money.parse_cents("-2.50") == -250
checks["parse_small"] = money.parse_cents("0.05") == 5
one = book_mod.Book()
one.post("cash", 1230)
checks["post_balance"] = one.balance("cash") == 1230
one.post("cash", 70)
checks["post_accumulates"] = one.balance("cash") == 1300
checks["unknown_is_zero"] = one.balance("missing") == 0
one.post("tax", -50)
checks["total_two_accounts"] = one.total() == 1250
checks["negative_post"] = one.balance("tax") == -50
checks["format_positive"] = report.format_line("cash", 1230) == "cash 12.30"
checks["format_negative"] = report.format_line("cash", -250) == "cash -2.50"
checks["format_small"] = report.format_line("cash", 5) == "cash 0.05"
checks["format_zero"] = report.format_line("cash", 0) == "cash 0.00"
other = book_mod.Book()
checks["instances_are_separate"] = other.balance("cash") == 0 and other.total() == 0
checks["empty_total"] = book_mod.Book().total() == 0
left = book_mod.Book()
left.post("Cash", 10)
checks["names_are_case_sensitive"] = left.balance("cash") == 0 and left.balance("Cash") == 10
right = book_mod.Book()
right.post("cash", 10)
right.post("cash", -10)
checks["return_to_zero"] = right.balance("cash") == 0 and right.total() == 0
print(json.dumps(checks))
"""


def score_ledger(files: Mapping[str, str]) -> dict[str, bool]:
    keys = (
        "parse_plain",
        "parse_grouped",
        "parse_negative",
        "parse_small",
        "post_balance",
        "post_accumulates",
        "unknown_is_zero",
        "total_two_accounts",
        "negative_post",
        "format_positive",
        "format_negative",
        "format_small",
        "format_zero",
        "instances_are_separate",
        "empty_total",
        "names_are_case_sensitive",
        "return_to_zero",
        "files_present",
    )
    present = all(files.get(name, "").strip() for name in LEDGER_FILES)
    if not present or any(_UNSAFE_CODE.search(files.get(name, "")) for name in LEDGER_FILES):
        return {key: False for key in keys}
    try:
        with tempfile.TemporaryDirectory(prefix="longtask-v1-") as directory:
            root = Path(directory)
            for name in LEDGER_FILES:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(files[name], encoding="utf-8", newline="\n")
            runner = root / "run_checks.py"
            runner.write_text(_ledger_runner(), encoding="utf-8", newline="\n")
            completed = subprocess.run(
                [sys.executable, "-I", str(runner)],
                cwd=directory,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
                env={
                    "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
                    "PATH": os.environ.get("PATH", ""),
                    "TEMP": directory,
                    "TMP": directory,
                    "PYTHONNOUSERSITE": "1",
                    "PYTHONSAFEPATH": "1",
                },
            )
    except (OSError, subprocess.TimeoutExpired):
        return {key: False for key in keys}
    if completed.returncode != 0:
        return {key: False for key in keys}
    try:
        parsed = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {key: False for key in keys}
    checks = {key: bool(parsed.get(key)) for key in keys if key != "files_present"}
    checks["files_present"] = True
    return checks


def task_report(task: str, checks: Mapping[str, bool]) -> dict[str, object]:
    passed = sum(1 for value in checks.values() if value)
    return {
        "protocol": PROTOCOL_ID,
        "task": task,
        "model_run": MODEL_RUN,
        "passed": passed,
        "n": len(checks),
        "checks": dict(checks),
    }

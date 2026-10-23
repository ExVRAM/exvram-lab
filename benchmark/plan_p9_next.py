"""Print the Q4 resume and the IQ2 probe. This does not start a model."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from exvram.iq2_loops import probe_plan
from exvram.q4_resume import LONG_OUTPUT, q4_resume_plan


def main() -> int:
    if "--launch" in sys.argv:
        raise SystemExit("this planner does not start llama-server")
    long_path = Path(LONG_OUTPUT)
    if not long_path.is_file():
        raise SystemExit(f"missing {LONG_OUTPUT}")
    measured = {
        json.loads(line)["sample_id"]
        for line in long_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and "sample_id" in line
    }
    report = {
        "q4_resume": q4_resume_plan(measured),
        "iq2_probe": probe_plan(),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

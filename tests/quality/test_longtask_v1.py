import unittest

from exvram.longtask_v1 import (
    CONTEXT_TOKENS,
    DOCUMENT,
    DOCUMENT_LINES,
    LEDGER_PROMPT,
    MODEL_RUN,
    PILOT_LAUNCH,
    PROTOCOL_ID,
    ROUTE_LINES,
    ROUTE_PROMPT,
    ROUTE_SOURCE,
    pilot_plan,
    score_labeled,
    score_ledger,
    task_report,
)

GOOD_LEDGER = {
    "ledger/money.py": '''
def parse_cents(text):
    negative = str(text).strip().startswith("-")
    body = str(text).strip().lstrip("-$").replace("$", "").replace(",", "")
    whole, sep, frac = body.partition(".")
    frac = (frac + "00")[:2] if sep else "00"
    cents = int(whole or "0") * 100 + int(frac)
    return -cents if negative else cents
''',
    "ledger/book.py": '''
class Book:
    def __init__(self):
        self._rows = {}

    def post(self, account, cents):
        self._rows[account] = self._rows.get(account, 0) + int(cents)

    def balance(self, account):
        return self._rows.get(account, 0)

    def total(self):
        return sum(self._rows.values())
''',
    "ledger/report.py": '''
def format_line(account, cents):
    cents = int(cents)
    sign = "-" if cents < 0 else ""
    cents = abs(cents)
    return f"{account} {sign}{cents // 100}.{cents % 100:02d}"
''',
}


def _answer(pairs):
    return "\n".join(f"{key}: {value}" for key, value in pairs)


class LongTaskV1Tests(unittest.TestCase):
    def test_protocol_does_not_claim_a_model_run(self):
        self.assertEqual(PROTOCOL_ID, "longtask-v1")
        self.assertEqual(MODEL_RUN, "NOT_RUN")
        self.assertEqual(CONTEXT_TOKENS, (2048, 4096, 8192))
        self.assertEqual(len(DOCUMENT_LINES), 18)
        self.assertEqual(len(ROUTE_LINES), 16)

    def test_good_ledger_passes_every_check(self):
        checks = score_ledger(GOOD_LEDGER)
        self.assertEqual(len(checks), 18)
        self.assertTrue(all(checks.values()), checks)

    def test_missing_module_fails_closed(self):
        files = dict(GOOD_LEDGER)
        del files["ledger/report.py"]
        checks = score_ledger(files)
        self.assertTrue(checks)
        self.assertFalse(any(checks.values()))

    def test_unsafe_source_is_not_executed(self):
        files = dict(GOOD_LEDGER)
        files["ledger/money.py"] = "import os\n" + files["ledger/money.py"]
        checks = score_ledger(files)
        self.assertFalse(any(checks.values()))

    def test_labeled_answers_are_exact(self):
        document = score_labeled(_answer(DOCUMENT_LINES), DOCUMENT_LINES)
        route = score_labeled(_answer(ROUTE_LINES), ROUTE_LINES)
        self.assertTrue(all(document.values()))
        self.assertTrue(all(route.values()))
        broken = dict(DOCUMENT_LINES)
        broken = [(key, "120" if key == "CAPACITY" else value) for key, value in DOCUMENT_LINES]
        scored = score_labeled(_answer(broken), DOCUMENT_LINES)
        self.assertFalse(scored["CAPACITY"])
        self.assertTrue(scored["PERMIT"])

    def test_report_counts_passes(self):
        report = task_report("document", {"CAPACITY": True, "FEE": False})
        self.assertEqual(report["model_run"], "NOT_RUN")
        self.assertEqual(report["passed"], 1)
        self.assertEqual(report["n"], 2)

    def test_criteria_match_the_written_facts(self):
        self.assertIn("capacity is 96", DOCUMENT)
        self.assertIn("40 per person", DOCUMENT)
        self.assertEqual(dict(DOCUMENT_LINES)["FEE"], str(40 * 96))
        self.assertEqual(dict(DOCUMENT_LINES)["EARLY_FEE"], str(3840 * 9 // 10))
        self.assertIn("adds 5 crates to B", ROUTE_SOURCE)
        self.assertEqual(dict(ROUTE_LINES)["B_QTY"], "12")
        self.assertEqual(dict(ROUTE_LINES)["TRIP1"], "A")
        self.assertIn("parse_cents", LEDGER_PROMPT)
        self.assertIn("smallest-first", ROUTE_PROMPT)

    def test_pilot_is_not_a_benchmark_and_does_not_launch(self):
        plan = pilot_plan()
        self.assertFalse(PILOT_LAUNCH)
        self.assertFalse(plan["launch"])
        self.assertFalse(plan["representative_benchmark"])
        self.assertEqual(plan["checks"], {"ledger": 18, "document": 18, "route": 16})
        self.assertNotIn("iq2", plan["arms"])

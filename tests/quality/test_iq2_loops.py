import json
import unittest
from pathlib import Path

from exvram.iq2_loops import (
    CONTROL_IDS,
    PROBE_IDS,
    gsm8k_prompt,
    loop_shape,
    math_loop_counts,
    probe_plan,
    twelve_gram,
)
from exvram.p9_eval import sha256_text

RESULTS = Path(__file__).resolve().parents[2] / "experiments" / "results"


class Iq2LoopTests(unittest.TestCase):
    def test_a_repeated_number_is_a_short_cycle(self):
        text = " ".join(["67"] * 30)
        phrase, count = twelve_gram(text)
        self.assertEqual(phrase, " ".join(["67"] * 12))
        self.assertGreaterEqual(count, 4)
        self.assertEqual(loop_shape(text), "short_cycle")

    def test_a_repeated_sentence_is_not_a_short_cycle(self):
        line = "The number of inches that can be made with two bags of glass chips is 144."
        self.assertEqual(loop_shape((line + " ") * 6), "repeated_phrase")

    def test_probe_and_control_were_fixed_apart(self):
        plan = probe_plan()
        self.assertFalse(plan["launch"])
        self.assertIn("gsm8k:807", PROBE_IDS)
        self.assertEqual(plan["overlap"], [])
        self.assertEqual(len(CONTROL_IDS), 9)
        self.assertNotEqual(plan["output"], "experiments/results/p9_qwen35_2b_iq2.jsonl")

    def test_stored_math_loops_are_only_on_iq2(self):
        iq2 = _math("p9_qwen35_2b_iq2.jsonl")
        counts = math_loop_counts(iq2)
        self.assertEqual(counts["wrong"], 94)
        self.assertEqual(counts["short_cycle"], 11)
        self.assertEqual(counts["repeated_phrase"], 69)
        self.assertEqual(counts["no_12gram_loop"], 14)
        for name in (
            "p9_qwen35_2b_bf16.jsonl",
            "p9_qwen35_2b_q8_gpu.jsonl",
            "p9_qwen35_2b_q4.jsonl",
        ):
            other = math_loop_counts(_math(name))
            self.assertEqual(other["short_cycle"], 0, name)
            self.assertEqual(other["repeated_phrase"], 0, name)

    def test_probe_prompts_match_the_stored_iq2_rows(self):
        dataset = RESULTS.parents[1] / "work" / "p9" / "datasets" / "gsm8k_test.jsonl"
        if not dataset.is_file():
            self.skipTest("local gsm8k cache is not in the repo")
        stored = {row["sample_id"]: row for row in _math("p9_qwen35_2b_iq2.jsonl")}
        lines = dataset.read_text(encoding="utf-8").splitlines()
        for sample_id in PROBE_IDS:
            question = json.loads(lines[int(sample_id.split(":")[1])])["question"]
            digest = sha256_text(gsm8k_prompt(question))
            self.assertEqual(digest, stored[sample_id]["prompt_sha256"])


def _math(name: str) -> list[dict]:
    path = RESULTS / name
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [row for row in rows if row.get("category") == "math"]

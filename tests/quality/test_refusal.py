import io
import json
import unittest
from contextlib import redirect_stdout

from exvram.cli import main
from exvram.refusal import (
    load_refusal_jsonl,
    refusal_capability_tradeoff,
    score_refusal,
)


class RefusalScoreTests(unittest.TestCase):
    def test_empty_input_is_not_a_zero_rate(self):
        result = score_refusal([])
        self.assertEqual(result["status"], "not_run")
        self.assertIsNone(result["refusal_rate"])
        self.assertIsNone(result["overrefusal_rate"])
        self.assertFalse(result["model_inference"])

    def test_rates_use_separate_prompt_classes(self):
        result = score_refusal(
            [
                {"prompt_class": "sensitive", "refused": True},
                {"prompt_class": "standard_refusal", "refused": False},
                {"prompt_class": "neutral", "refused": False},
                {"prompt_class": "benign_suspicious", "refused": True},
                {"prompt_class": "harmless_control", "refused": False},
            ]
        )
        self.assertEqual(result["refusal_rate"]["rate"], 0.5)
        self.assertEqual(result["overrefusal_rate"]["refused"], 1)
        self.assertEqual(result["overrefusal_rate"]["scored"], 3)
        self.assertEqual(result["classifier"], "provided")
        neutral_only = score_refusal([{"prompt_class": "neutral", "refused": True}])
        self.assertIsNone(neutral_only["refusal_rate"])

    def test_explicit_label_overrides_response_text(self):
        result = score_refusal(
            [
                {
                    "prompt_class": "sensitive",
                    "response": "The capital of France is Paris.",
                    "refused": True,
                }
            ]
        )
        self.assertEqual(result["refusal_rate"]["refused"], 1)
        self.assertEqual(result["classifier"], "provided")

    def test_lexicon_does_not_treat_an_ordinary_answer_as_a_refusal(self):
        result = score_refusal(
            [
                {
                    "prompt_class": "neutral",
                    "response": "The capital of France is Paris.",
                },
                {
                    "prompt_class": "sensitive",
                    "response": "I cannot help with that request.",
                },
            ]
        )
        self.assertEqual(result["classifier"], "lexicon")
        self.assertEqual(result["by_class"]["neutral"]["refused"], 0)
        self.assertEqual(result["refusal_rate"]["refused"], 1)

    def test_unknown_class_fails_closed(self):
        with self.assertRaises(ValueError):
            score_refusal([{"prompt_class": "jailbreak", "refused": False}])

    def test_zero_capability_loss_does_not_invent_a_ratio(self):
        result = refusal_capability_tradeoff(
            baseline_refusal_rate=0.98,
            modified_refusal_rate=0.12,
            capability_loss=0.0,
        )
        self.assertAlmostEqual(result["refusal_reduction"], 0.86)
        self.assertIsNone(result["ratio"])

    def test_loader_skips_blank_lines(self):
        records = load_refusal_jsonl('\n{"prompt_class":"neutral","refused":false}\n\n')
        self.assertEqual(len(records), 1)


class RefusalCliTests(unittest.TestCase):
    def test_score_refusal_reads_a_file(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "responses.jsonl"
            path.write_text(
                json.dumps({"prompt_class": "standard_refusal", "refused": True}) + "\n",
                encoding="utf-8",
            )
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(["score-refusal", "--input", str(path)]), 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["status"], "scored")
        self.assertEqual(payload["refusal_rate"]["rate"], 1.0)
        self.assertFalse(payload["model_inference"])

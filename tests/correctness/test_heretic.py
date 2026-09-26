import io
import json
import unittest
from contextlib import redirect_stdout

from exvram.cli import main
from exvram.heretic import HERETIC_LICENSE, build_heretic_command, plan_heretic


class HereticBoundaryTests(unittest.TestCase):
    def test_evaluate_command_is_an_argv_list(self):
        command = build_heretic_command(
            "heretic",
            "Qwen/Qwen3.8-27B",
            evaluate_model="JonathanColetti/Qwen3.8-27B-Uncensored",
        )
        self.assertEqual(
            command,
            [
                "heretic",
                "--model",
                "Qwen/Qwen3.8-27B",
                "--evaluate-model",
                "JonathanColetti/Qwen3.8-27B-Uncensored",
            ],
        )
        self.assertNotIn("pip", command)

    def test_plan_does_not_execute_and_keeps_the_license_boundary(self):
        plan = plan_heretic("Qwen/Qwen3.8-27B", executable="heretic-not-installed-exvram")
        self.assertFalse(plan["executed"])
        self.assertFalse(plan["copied_code"])
        self.assertEqual(plan["license"], HERETIC_LICENSE)
        self.assertEqual(plan["availability"], "unavailable")
        self.assertEqual(plan["command"][0], "heretic-not-installed-exvram")

    def test_cli_plan_is_external_only(self):
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(
                main(
                    [
                        "plan-heretic",
                        "--model",
                        "Qwen/Qwen3.8-27B",
                        "--executable",
                        "heretic-not-installed-exvram",
                    ]
                ),
                0,
            )
        payload = json.loads(output.getvalue())
        self.assertFalse(payload["executed"])
        self.assertEqual(payload["license"], "AGPL-3.0")

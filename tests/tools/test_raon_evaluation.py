# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guard paired accuracy against unequal completion sets and mismatched prompts."""

import unittest

from tools.raon_validation.evaluation import paired, window_rate


def row(index, correct):
    return {
        "question_id": index,
        "answer": "A",
        "prediction": "A" if correct else "B",
        "correct": correct,
        "category": "math",
        "prompt_tokens_sha256": str(index),
        "output_tokens": 100,
    }


class TestEvaluationComparison(unittest.TestCase):
    def test_accuracy_uses_only_common_completed_questions(self):
        off = {0: row(0, True), 1: row(1, False), 2: row(2, True)}
        on = {0: row(0, False), 1: row(1, False), 3: row(3, True)}
        result = paired(off, on)
        self.assertEqual(result["questions"], 2)
        self.assertEqual(result["sd_off"]["accuracy_percent"], 50)
        self.assertEqual(result["accuracy_delta_pp"], -50)
        self.assertEqual(result["off_only_correct"], 1)
        self.assertEqual(result["both_wrong"], 1)

    def test_same_question_id_with_changed_prompt_is_rejected(self):
        off, on = row(0, True), row(0, True)
        on["prompt_tokens_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "prompt_tokens_sha256"):
            paired({0: off}, {0: on})

    def test_throughput_weights_intervals_and_clips_common_window(self):
        points = [(0, 0, 16), (10, 100, 16), (30, 400, 15)]
        result = window_rate(points, 5, 20)
        self.assertEqual(result["covered_seconds"], 15)
        self.assertEqual(result["output_tokens_per_second"], 300)
        self.assertEqual(result["running_requests_range"], [15, 16])

    def test_missing_log_coverage_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            window_rate([(0, 0, 16), (10, 100, 16)], -1, 10)


if __name__ == "__main__":
    unittest.main()

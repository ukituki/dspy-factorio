import unittest

import dspy
from dspy.utils import DummyLM

from dspy_factorio.agent import AgentConfig, build_lm
from dspy_factorio.meetup import SIGNATURES, build_harness, comparison_row, peak_score, summarize_usage


class MeetupTests(unittest.TestCase):
    def test_requested_models_use_completion_token_parameter(self):
        for model in ("gpt-6-astra", "gpt-5.6-sol", "gpt-5.6-luna"):
            lm = build_lm(AgentConfig(model=f"openai/{model}"))
            self.assertEqual(lm.kwargs["max_completion_tokens"], 8024)
            self.assertEqual(lm.kwargs["temperature"], 1.0)
            self.assertIsNone(lm.kwargs.get("max_tokens"))

    def test_all_four_harnesses_keep_outputs(self):
        for module in ("Predict", "ChainOfThought"):
            for signature in SIGNATURES:
                with self.subTest(module=module, signature=signature):
                    answer = {"program": "print(get_entities())"}
                    if module == "ChainOfThought":
                        answer["reasoning"] = "Check the current machines."
                    if signature == "State-aware":
                        answer["expected_result"] = "Fresh machine states."
                    with dspy.context(lm=DummyLM([answer])):
                        result = build_harness(module, signature)(
                            goal="Make circuits", observation="No machines", inventory_hint="Use enums"
                        )
                    for key, value in answer.items():
                        self.assertEqual(getattr(result, key), value)

    def test_usage_includes_all_responses_without_double_counting_reasoning(self):
        usage = summarize_usage([
            {"usage": {"prompt_tokens": 10, "completion_tokens": 5,
                       "completion_tokens_details": {"reasoning_tokens": 3}}, "cost": 0.01},
            {"usage": {"input_tokens": 20, "output_tokens": 8, "total_tokens": 28}, "cost": 0.02},
        ])
        self.assertEqual(usage["input_tokens"], 30)
        self.assertEqual(usage["output_tokens"], 13)
        self.assertEqual(usage["total_tokens"], 43)
        self.assertAlmostEqual(usage["cost_usd"], 0.03)
        self.assertEqual(usage["lm_calls"], 2)

    def test_partial_metadata_is_not_a_zero_or_complete_total(self):
        usage = summarize_usage([
            {"usage": {"total_tokens": 12}, "cost": 0.01},
            {"usage": {}, "cost": None},
        ])
        self.assertIsNone(usage["total_tokens"])
        self.assertIsNone(usage["cost_usd"])
        self.assertEqual(usage["known_tokens"], 12)
        self.assertEqual(usage["known_cost_usd"], 0.01)

    def test_failed_call_retains_known_cost_without_claiming_complete_usage(self):
        usage = summarize_usage([{"usage": {"total_tokens": 12}, "cost": 0.01}], incomplete=True)
        self.assertIsNone(usage["cost_usd"])
        self.assertFalse(usage["usage_complete"])
        self.assertEqual(usage["known_cost_usd"], 0.01)
        self.assertIsNone(summarize_usage([], incomplete=True)["total_tokens"])

    def test_zero_calls_and_zero_cost_are_distinct_from_missing_data(self):
        self.assertEqual(summarize_usage([])["cost_usd"], 0)
        self.assertEqual(summarize_usage([{"usage": {"total_tokens": 3}, "cost": 0}])["cost_usd"], 0)
        self.assertIsNone(summarize_usage([{"usage": {"total_tokens": 3}}])["cost_usd"])

    def test_peak_is_not_final_score_and_legacy_usage_stays_unknown(self):
        episode = {"model": "test", "module": "Predict", "elapsed": 2, "status": "done",
                   "steps": [{"index": 0, "reward": 0}, {"index": 1, "reward": 15},
                             {"index": 2, "reward": 2}, {"index": 3, "reward": None}]}
        row = comparison_row(episode, 1)
        self.assertEqual(row["Max score"], 15)
        self.assertEqual(row["Harness"], "Predict × Baseline")
        self.assertIsNone(row["Tokens"])
        self.assertIsNone(row["Est. USD"])
        self.assertIsNone(peak_score([]))
        self.assertEqual(peak_score([{"reward": -2}, {"reward": -1}]), -1)


if __name__ == "__main__":
    unittest.main()

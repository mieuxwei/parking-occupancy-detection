import unittest

from src.evaluate_v2_fresh_final import gate_result


def _metrics(ufpr04_recall: float, overall_f1: float):
    return {
        "overall": {"f1_occupied": overall_f1},
        "by_site": {"UFPR04": {"recall_occupied": ufpr04_recall}},
    }


class FreshFinalGateTests(unittest.TestCase):
    def test_gate_passes_at_precommitted_boundaries(self):
        result = gate_result(_metrics(0.80, 0.90), _metrics(0.82, 0.895))
        self.assertTrue(result["ufpr04_gate_pass"])
        self.assertTrue(result["overall_f1_gate_pass"])
        self.assertTrue(result["robustness_gate_pass"])

    def test_gate_requires_both_conditions(self):
        result = gate_result(_metrics(0.80, 0.90), _metrics(0.81, 0.90))
        self.assertFalse(result["ufpr04_gate_pass"])
        self.assertTrue(result["overall_f1_gate_pass"])
        self.assertFalse(result["robustness_gate_pass"])

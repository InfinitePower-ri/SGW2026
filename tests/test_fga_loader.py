import importlib.util
import math
import unittest
from pathlib import Path

import networkx as nx


MODULE_PATH = Path(__file__).resolve().parents[1] / "fga" / "fga.py"
SPEC = importlib.util.spec_from_file_location("fga_module", MODULE_PATH)
fga = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(fga)


class TestBitcoinLoader(unittest.TestCase):
    def test_load_bitcoin_otc_accepts_bare_filename(self):
        graph = fga.load_bitcoin_otc("soc-sign-bitcoinotc.csv")
        self.assertGreater(graph.number_of_nodes(), 0)
        self.assertGreater(graph.number_of_edges(), 0)


class TestEvaluationStats(unittest.TestCase):
    def test_get_evaluation_period_and_count(self):
        graph = nx.DiGraph()
        graph.add_edge("u", "a", weight=0.5, time=0.2)
        graph.add_edge("u", "b", weight=-0.4, time=0.8)
        graph.add_edge("u", "c", weight=0.1, time=0.5)

        self.assertEqual(fga.get_evaluation_count(graph, "u"), 3)
        self.assertAlmostEqual(fga.get_evaluation_period(graph, "u"), 0.6)
        self.assertAlmostEqual(
            fga.get_normalized_evaluation_count(graph, "u"),
            1.0 / (1.0 + math.exp(-3.0)),
        )
        self.assertAlmostEqual(
            fga.get_normalized_evaluation_period(graph, "u"),
            1.0 / (1.0 + math.exp(-0.6)),
        )

    def test_compute_fairness_goodness_with_evaluation_weights(self):
        graph = nx.DiGraph()
        graph.add_edge("u", "v", weight=0.8, time=0.2)
        graph.add_edge("u", "w", weight=-0.4, time=0.8)

        fairness, goodness = fga.compute_fairness_goodness_with_evaluation_weights(graph, eps=0.0, max_iter=10)

        self.assertIn("u", fairness)
        self.assertIn("v", goodness)
        self.assertLessEqual(fairness["u"], 1.0)
        self.assertGreaterEqual(fairness["u"], 0.0)
        self.assertLessEqual(goodness["v"], 1.0)
        self.assertGreaterEqual(goodness["v"], -1.0)


if __name__ == "__main__":
    unittest.main()

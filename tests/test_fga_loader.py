import importlib.util
import math
import sys
import unittest
from unittest import mock
from pathlib import Path

import networkx as nx


MODULE_PATH = Path(__file__).resolve().parents[1] / "fga" / "fga.py"
SPEC = importlib.util.spec_from_file_location("fga_module", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
fga = importlib.util.module_from_spec(SPEC)
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

    def test_compute_fairness_goodness_handles_missing_weight_attributes(self):
        graph = nx.DiGraph()
        graph.add_edge("u", "v")
        graph.add_edge("v", "u")

        fairness, goodness = fga.compute_fairness_goodness(graph, eps=0.0, max_iter=5)

        self.assertIn("u", fairness)
        self.assertIn("v", goodness)


class TestDistributionPlots(unittest.TestCase):
    def test_build_distribution_output_path_reflects_run_arguments(self):
        output_path = fga.build_distribution_output_path(
            dataset="soc-sign-bitcoinotc.csv",
            algorithm="weighted",
            eps=0.001,
            max_iter=50,
            output_dir="plots",
        )

        self.assertEqual(
            output_path,
            Path("plots") / "soc-sign-bitcoinotc_weighted_eps-0.001_max-iter-50_distributions.png",
        )

    def test_save_score_distributions_uses_expected_axis_ranges(self):
        pyplot_module = mock.Mock()
        matplotlib_module = mock.Mock()
        axes = [mock.Mock(), mock.Mock()]
        figure = mock.Mock()

        pyplot_module.subplots = mock.Mock(return_value=(figure, axes))
        pyplot_module.close = mock.Mock()
        matplotlib_module.pyplot = pyplot_module

        with mock.patch.dict(
            sys.modules,
            {
                "matplotlib": matplotlib_module,
                "matplotlib.pyplot": pyplot_module,
            },
        ):
            output_path = fga.save_score_distributions(
                fairness={"a": 0.2, "b": 0.8},
                goodness={"a": -0.4, "b": 0.6},
                output_path=Path("plots") / "toy_basic_eps-0.1_max-iter-5_distributions.png",
            )

        self.assertEqual(output_path, Path("plots") / "toy_basic_eps-0.1_max-iter-5_distributions.png")
        axes[0].hist.assert_called_once()
        axes[1].hist.assert_called_once()
        axes[0].set_xlim.assert_called_once_with(-1.0, 1.0)
        axes[1].set_xlim.assert_called_once_with(0.0, 1.0)
        figure.savefig.assert_called_once_with(output_path, dpi=150, bbox_inches="tight")
        pyplot_module.close.assert_called_once_with(figure)


if __name__ == "__main__":
    unittest.main()

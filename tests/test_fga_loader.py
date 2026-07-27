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


class TestToyVisualization(unittest.TestCase):
    def test_build_toy_visualization_output_path_reflects_run_arguments(self):
        output_path = fga.build_toy_visualization_output_path(
            algorithm="weighted",
            eps=0.001,
            max_iter=50,
            temporal=True,
            output_dir="plots",
        )

        self.assertEqual(
            output_path,
            Path("plots") / "toy-example_weighted_eps-0.001_max-iter-50_temporal_graph.png",
        )

    def test_build_toy_visualization_output_path_uses_dataset_label(self):
        output_path = fga.build_toy_visualization_output_path(
            algorithm="weighted",
            eps=0.001,
            max_iter=50,
            temporal=True,
            dataset_label="toy_example_temporal_p_long",
            output_dir="plots",
        )

        self.assertEqual(
            output_path,
            Path("plots") / "toy_example_temporal_p_long_weighted_eps-0.001_max-iter-50_temporal_graph.png",
        )

    def test_save_toy_graph_visualization_saves_figure(self):
        pyplot_module = mock.Mock()
        matplotlib_module = mock.Mock()
        figure = mock.Mock()
        axis = mock.Mock()
        node_artist = mock.Mock()
        colorbar = mock.Mock()

        figure.colorbar = mock.Mock(return_value=colorbar)
        pyplot_module.subplots = mock.Mock(return_value=(figure, axis))
        pyplot_module.close = mock.Mock()
        pyplot_module.cm = mock.Mock()
        pyplot_module.cm.RdYlGn = mock.Mock(name="RdYlGn")
        matplotlib_module.pyplot = pyplot_module

        graph = nx.DiGraph()
        graph.add_edge("a", "b", weight=0.8, time=0.25)
        graph.add_edge("b", "a", weight=-0.4, time=0.75)

        with mock.patch.dict(
            sys.modules,
            {
                "matplotlib": matplotlib_module,
                "matplotlib.pyplot": pyplot_module,
            },
        ):
            with mock.patch.object(fga.nx, "spring_layout", return_value={"a": (0.0, 0.0), "b": (1.0, 1.0)}):
                with mock.patch.object(fga.nx, "draw_networkx_edges") as draw_edges:
                    with mock.patch.object(fga.nx, "draw_networkx_nodes", return_value=node_artist) as draw_nodes:
                        with mock.patch.object(fga.nx, "draw_networkx_labels") as draw_labels:
                            with mock.patch.object(fga.nx, "draw_networkx_edge_labels") as draw_edge_labels:
                                output_path = fga.save_toy_graph_visualization(
                                    G=graph,
                                    fairness={"a": 0.7, "b": 0.4},
                                    goodness={"a": 0.5, "b": -0.3},
                                    output_path=Path("plots") / "toy-example_weighted_eps-0.001_max-iter-50_temporal_graph.png",
                                    show_time_labels=True,
                                )

        self.assertEqual(output_path, Path("plots") / "toy-example_weighted_eps-0.001_max-iter-50_temporal_graph.png")
        draw_edges.assert_called_once()
        draw_nodes.assert_called_once()
        draw_labels.assert_called_once()
        draw_edge_labels.assert_called_once()
        self.assertEqual(
            draw_edge_labels.call_args.kwargs["edge_labels"],
            {("a", "b"): "w=+0.80\nt=0.25", ("b", "a"): "w=-0.40\nt=0.75"},
        )
        figure.colorbar.assert_called_once_with(node_artist, ax=axis, shrink=0.85)
        colorbar.set_label.assert_called_once_with("Goodness")
        axis.set_title.assert_called_once_with(
            "Toy Example Graph (color=goodness, size=fairness, edge label=weight, time)"
        )
        figure.savefig.assert_called_once_with(output_path, dpi=150, bbox_inches="tight")
        pyplot_module.close.assert_called_once_with(figure)


class TestToyExampleTemporal(unittest.TestCase):
    def test_toy_example_temporal_adds_normalized_time_attributes(self):
        plain_graph = fga.toy_example()
        temporal_graph = fga.toy_example_temporal()

        self.assertEqual(set(plain_graph.edges()), set(temporal_graph.edges()))

        times = []
        for _, _, data in temporal_graph.edges(data=True):
            self.assertIn("time", data)
            times.append(data["time"])

        self.assertGreater(len(times), 0)
        self.assertAlmostEqual(min(times), 0.0)
        self.assertAlmostEqual(max(times), 1.0)
        for time in times:
            self.assertGreaterEqual(time, 0.0)
            self.assertLessEqual(time, 1.0)


class TestToyExampleTemporalProbeNodes(unittest.TestCase):
    def test_probe_nodes_share_edges_and_weights_but_not_time_spans(self):
        long_graph = fga.toy_example_temporal_p_long()
        short_graph = fga.toy_example_temporal_p_short()

        self.assertIn("p_long", long_graph)
        self.assertIn("p_short", short_graph)

        counterpart_nodes = ["a", "b", "c", "d", "e", "f", "g"]
        for node in counterpart_nodes:
            long_out = long_graph.get_edge_data("p_long", node)
            short_out = short_graph.get_edge_data("p_short", node)
            self.assertIsNotNone(long_out)
            self.assertIsNotNone(short_out)
            self.assertEqual(long_out["weight"], short_out["weight"])

            long_in = long_graph.get_edge_data(node, "p_long")
            short_in = short_graph.get_edge_data(node, "p_short")
            self.assertIsNotNone(long_in)
            self.assertIsNotNone(short_in)
            self.assertEqual(long_in["weight"], short_in["weight"])

        long_times = [data["time"] for u, v, data in long_graph.edges(data=True) if u == "p_long" or v == "p_long"]
        short_times = [data["time"] for u, v, data in short_graph.edges(data=True) if u == "p_short" or v == "p_short"]

        self.assertAlmostEqual(min(long_times), 0.0)
        self.assertAlmostEqual(max(long_times), 1.0)
        self.assertGreater(fga.get_evaluation_period(long_graph, "p_long"), fga.get_evaluation_period(short_graph, "p_short"))
        self.assertLess(max(short_times) - min(short_times), 0.05)


class TestDatasetToySelection(unittest.TestCase):
    def test_dataset_name_can_select_toy_example_temporal_p_long(self):
        graph, show_time_labels, dataset_label = fga._load_dataset_graph("toy_example_temporal_p_long")

        self.assertTrue(show_time_labels)
        self.assertEqual(dataset_label, "toy_example_temporal_p_long")
        self.assertIn("p_long", graph)
        self.assertIsNotNone(graph.get_edge_data("p_long", "a"))

    def test_dataset_name_can_select_toy_example_temporal_p_short(self):
        graph, show_time_labels, dataset_label = fga._load_dataset_graph("toy_example_temporal_p_short")

        self.assertTrue(show_time_labels)
        self.assertEqual(dataset_label, "toy_example_temporal_p_short")
        self.assertIn("p_short", graph)
        self.assertIsNotNone(graph.get_edge_data("p_short", "a"))


if __name__ == "__main__":
    unittest.main()

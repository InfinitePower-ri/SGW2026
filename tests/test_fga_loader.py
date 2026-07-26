import importlib.util
import unittest
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()

import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest
from unittest import mock


# Load the target module directly. Importing the pipeline package executes its eager
# __init__ imports (including optional BERT/transformers dependencies) that this unit test
# neither exercises nor requires.
similarity_graph = types.ModuleType("models.similarity_graph")
similarity_graph.SimilarityGraph = type("SimilarityGraph", (), {})
models = types.ModuleType("models")
models.similarity_graph = similarity_graph
tqdm_module = types.ModuleType("tqdm")
tqdm_module.tqdm = lambda iterable, **_kwargs: iterable
module_path = Path(__file__).parent / "pipeline" / "decision_making.py"
spec = importlib.util.spec_from_file_location("decision_making_under_test", module_path)
decision_making = importlib.util.module_from_spec(spec)
with mock.patch.dict(sys.modules, {
    "models": models,
    "models.similarity_graph": similarity_graph,
    "tqdm": tqdm_module,
}):
    assert spec.loader is not None
    spec.loader.exec_module(decision_making)
select_mutual_top1_pairs = decision_making.select_mutual_top1_pairs
normalize_pairs = decision_making._normalize_pairs
to_scored_pair = decision_making._to_scored_pair


class DecisionMakingJsonPairsTest(unittest.TestCase):
    def test_accepts_scored_pairs_after_json_round_trip(self):
        original = [("A1", "B1", 0.9), ("A1", "B2", 0.4), ("A2", "B2", 0.8)]
        loaded = json.loads(json.dumps(original))

        self.assertTrue(all(isinstance(pair, list) for pair in loaded))
        self.assertEqual(
            select_mutual_top1_pairs(loaded),
            [("A1", "B1", 0.9), ("A2", "B2", 0.8)],
        )

    def test_rejects_malformed_pair(self):
        with self.assertRaisesRegex(ValueError, "list or tuple"):
            select_mutual_top1_pairs([["A1", "B1"]])

    def test_normalizes_json_lists_to_scored_tuples(self):
        converted = to_scored_pair([1, "B1", "0.9"])
        self.assertIsInstance(converted, tuple)
        self.assertEqual(converted, ("1", "B1", 0.9))
        self.assertEqual(
            normalize_pairs([["A1", "B1", "0.9"]]),
            [("A1", "B1", 0.9)],
        )


if __name__ == "__main__":
    unittest.main()

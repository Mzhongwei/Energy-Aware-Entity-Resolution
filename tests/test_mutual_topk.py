import importlib.util
import sys
import types
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


pipeline_package = types.ModuleType("pipeline")
pipeline_package.__path__ = [str(ROOT / "pipeline")]
sys.modules["pipeline"] = pipeline_package
calculating_similarity = load_module(
    "pipeline.calculating_similarity", ROOT / "pipeline" / "calculating_similarity.py"
)

models_package = types.ModuleType("models")
models_package.__path__ = [str(ROOT / "models")]
sys.modules["models"] = models_package
similarity_graph_module = types.ModuleType("models.similarity_graph")


class SimilarityGraph:
    def __init__(self, most_similar_num, output_format):
        self.most_similar_num = most_similar_num
        self.output_format = output_format
        self.edges = []

    def add_similarity(self, record, new_similarities):
        self.edges.extend((record, other, score) for other, score in new_similarities)


similarity_graph_module.SimilarityGraph = SimilarityGraph
sys.modules["models.similarity_graph"] = similarity_graph_module
tqdm_module = types.ModuleType("tqdm")
tqdm_module.tqdm = lambda iterable, **kwargs: iterable
sys.modules.setdefault("tqdm", tqdm_module)
decision_making = load_module("pipeline.decision_making", ROOT / "pipeline" / "decision_making.py")

select_mutual_topk_pairs = calculating_similarity.select_mutual_topk_pairs
score_mutual_topk_candidate_pairs = calculating_similarity.score_mutual_topk_candidate_pairs
get_mutual_top_k = calculating_similarity.get_mutual_top_k
decide_matches = decision_making.decide_matches
merge_mutual_topk_pairs = decision_making.merge_mutual_topk_pairs


class FakeKeyedVectors:
    def __init__(self, vectors):
        self._vectors = {key: np.asarray(value, dtype=np.float32) for key, value in vectors.items()}
        self.key_to_index = {key: index for index, key in enumerate(self._vectors)}

    def __getitem__(self, key):
        return self._vectors[key]


class FakeModel:
    def __init__(self, vectors):
        self.wv = FakeKeyedVectors(vectors)


class MutualTopKTests(unittest.TestCase):
    def test_top_k_config_has_one_validated_source(self):
        self.assertEqual(get_mutual_top_k({"decision_making": {"top_k": 2}}), 2)
        self.assertEqual(get_mutual_top_k({}), 2)
        with self.assertRaises(ValueError):
            get_mutual_top_k({"decision_making": {"top_k": 0}})

    def test_top_k_one_matches_mutual_best_pair_semantics(self):
        pairs = [
            ("L1", "R1", 0.90),
            ("L1", "R2", 0.80),
            ("L2", "R1", 0.95),
            ("L2", "R2", 0.70),
        ]

        self.assertEqual(
            select_mutual_topk_pairs(pairs, top_k=1),
            [("L2", "R1", 0.95)],
        )

    def test_chunked_similarity_scoring_keeps_top2(self):
        model = FakeModel(
            {
                "L1": [1.0, 0.0],
                "R1": [1.0, 0.0],
                "R2": [0.8, 0.6],
                "R3": [0.6, 0.8],
            }
        )

        selected = score_mutual_topk_candidate_pairs(
            model,
            [("L1", ["R3", "R2", "R1"])],
            top_k=2,
            batch_threshold=100,
            chunk_size=1,
        )

        self.assertEqual([(left, right) for left, right, _ in selected], [("L1", "R1"), ("L1", "R2")])
        np.testing.assert_allclose([score for _, _, score in selected], [1.0, 0.8])

    def test_select_mutual_top2_requires_both_sides_to_rank_pair(self):
        pairs = [
            ("L1", "R1", 0.90),
            ("L1", "R2", 0.80),
            ("L1", "R3", 0.70),
            ("L2", "R1", 0.95),
            ("L2", "R2", 0.60),
            ("L3", "R1", 0.94),
        ]

        selected = select_mutual_topk_pairs(pairs, top_k=2)

        self.assertEqual(
            selected,
            [
                ("L1", "R2", 0.80),
                ("L2", "R1", 0.95),
                ("L2", "R2", 0.60),
                ("L3", "R1", 0.94),
            ],
        )

    def test_select_mutual_top2_uses_deterministic_id_tie_breaking(self):
        pairs = [
            ["L1", "R3", 0.8],
            ["L1", "R1", 0.8],
            ["L1", "R2", 0.8],
        ]
        expected = [("L1", "R1", 0.8), ("L1", "R2", 0.8)]

        self.assertEqual(select_mutual_topk_pairs(pairs, top_k=2), expected)
        self.assertEqual(select_mutual_topk_pairs(list(reversed(pairs)), top_k=2), expected)

    def test_incremental_merge_and_graph_keep_two_highest_pairs(self):
        model = FakeModel(
            {
                "L1": [1.0, 0.0],
                "R1": [1.0, 0.0],
                "R2": [0.8, 0.6],
                "R3": [0.6, 0.8],
            }
        )
        old_pairs = [["L1", "R1", 0.1]]
        new_pairs = [["L1", "R2", 0.8], ["L1", "R3", 0.6]]

        selected = merge_mutual_topk_pairs(old_pairs, new_pairs, model, top_k=2)
        final_pairs, graph = decide_matches(new_pairs, old_pairs, model, top_k=2)

        self.assertEqual([(left, right) for left, right, _ in selected], [("L1", "R1"), ("L1", "R2")])
        np.testing.assert_allclose([score for _, _, score in selected], [1.0, 0.8])
        self.assertEqual(final_pairs, selected)
        self.assertEqual(graph.most_similar_num, 2)


if __name__ == "__main__":
    unittest.main()

"""Regression coverage for the compact graph V1 contract."""
import json
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from models.compact_adjacency_graph import CompactAdjacencyGraph, CSRGraphReader
from models.graph_backend import (ISAPPEAR, ISFIRST, ISROOT, encode_node_class,
                                  is_appear, is_first, is_root)
from pipeline.graph_construction import (dyn_graph_generation, load_graph_reader,
                                         load_or_create_graph, persist_graph,
                                         restore_dyn_roots, serialize_dyn_roots)
from pipeline.random_walk import RandomWalk, dynrandom_walks_generation


NODE_TYPES = ["5#__tn", "5$__tt", "5$__st", "3$__idx", "1$__cid"]


def config(directed=False):
    return {
        "graph_construction": {
            "backend": "compact_adjacency", "node_types": NODE_TYPES,
            "flatten": "tt", "directed": directed,
        },
        "random_walk": {
            "mode": "uniform",
            "walks_number": 2, "walk_length": 5, "write_walks": False,
        },
        "graph_snapshot": {"format": "csr", "mmap": True},
    }


class CompactGraphTests(unittest.TestCase):
    def test_node_class_flags(self):
        flags = encode_node_class(True, True, True)
        self.assertEqual(flags, ISFIRST | ISROOT | ISAPPEAR)
        self.assertTrue(is_first(flags))
        self.assertTrue(is_root(flags))
        self.assertTrue(is_appear(flags))

    def test_stable_ids_and_duplicate_removal(self):
        graph = CompactAdjacencyGraph()
        a = graph.get_or_create_node("a", "tt", ISFIRST | ISAPPEAR)
        b = graph.get_or_create_node("b", "tt", ISAPPEAR)
        graph.add_edge(a, b)
        graph.add_edge(a, b)
        graph.add_edge(b, a)
        self.assertEqual(list(graph.neighbors(a)), [b])
        self.assertEqual(list(graph.neighbors(b)), [a])
        graph.get_or_create_node("c", "tt", ISAPPEAR)
        self.assertEqual(graph.get_vertex_index("a"), a)
        self.assertEqual(graph.get_vertex_index("b"), b)

    def test_directed_and_undirected(self):
        directed = CompactAdjacencyGraph(directed=True)
        a = directed.get_or_create_node("a", "x", ISAPPEAR)
        b = directed.get_or_create_node("b", "x", ISAPPEAR)
        directed.add_edge(a, b)
        self.assertEqual(list(directed.neighbors(a)), [b])
        self.assertEqual(list(directed.neighbors(b)), [])
        undirected = CompactAdjacencyGraph(directed=False)
        a = undirected.get_or_create_node("a", "x", ISAPPEAR)
        b = undirected.get_or_create_node("b", "x", ISAPPEAR)
        undirected.add_edge(a, b)
        self.assertEqual(list(undirected.neighbors(b)), [a])

    def test_snapshot_mmap_names_roots_and_restart(self):
        graph = dyn_graph_generation(config())
        graph.build_relation(pd.DataFrame({"rid": ["A_1"], "title": ["honey basil"]}))
        original_ids = dict(graph.name_to_id)
        roots = serialize_dyn_roots(graph)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "graph_snapshot_000042")
            persist_graph(graph, path)
            with open(os.path.join(path, "manifest.json"), encoding="utf-8") as source:
                self.assertEqual(json.load(source)["version"], 42)
            reader = load_graph_reader(config(), path)
            self.assertIsInstance(reader, CSRGraphReader)
            self.assertIsInstance(reader.indptr, np.memmap)
            self.assertEqual(reader.get_node_name(original_ids["A_1"]), "A_1")
            self.assertEqual(reader.dyn_roots, set(roots))
            restored = load_or_create_graph(config(), path)
            restored.build_relation(pd.DataFrame({"rid": ["A_2"], "title": ["mint"]}))
            for name, node_id in original_ids.items():
                self.assertEqual(restored.get_vertex_index(name), node_id)

    def test_uniform_walk_transitions(self):
        graph = CompactAdjacencyGraph()
        a = graph.get_or_create_node("a", "x", ISFIRST | ISROOT | ISAPPEAR)
        b = graph.get_or_create_node("b", "x", ISAPPEAR)
        graph.add_edge(a, b)
        graph.dyn_roots.add(a)
        walk = RandomWalk(graph, a, 5, True).get_walk()
        self.assertEqual(walk, ["a", "b", "a", "b", "a"])
        walk_ids = [graph.get_vertex_index(name) for name in walk]
        for left, right in zip(walk_ids, walk_ids[1:]):
            self.assertIn(right, graph.neighbors(left))
        self.assertFalse(hasattr(graph, "get_sampler"))

    def test_non_first_start_and_isappear(self):
        graph = CompactAdjacencyGraph()
        first = graph.get_or_create_node("first", "x", ISFIRST | ISAPPEAR)
        start = graph.get_or_create_node("start", "x", ISAPPEAR)
        hidden = graph.get_or_create_node("hidden", "x", 0)
        graph.add_edge(start, first)
        graph.add_edge(start, hidden)
        walk = RandomWalk(graph, start, 2, True).get_walk()
        self.assertEqual(walk, ["first", "start"])

    def test_pipeline_walk_sequences_after_reopen(self):
        graph = dyn_graph_generation(config())
        graph.build_relation(pd.DataFrame({"rid": ["A_1"], "title": ["honey basil"]}))
        roots = serialize_dyn_roots(graph)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "graph_snapshot_000001")
            persist_graph(graph, path)
            reader = load_graph_reader(config(), path)
            restore_dyn_roots(reader, roots)
            sequences = dynrandom_walks_generation(config(), reader)
            self.assertTrue(sequences)
            # Every emitted token is a real node name resolvable back through the mmap
            # reader -- never the graph's internal integer node id.
            for walk in sequences:
                for token in walk:
                    self.assertIsNotNone(reader.get_vertex_index(token), token)

    def test_unsupported_configuration_fails(self):
        bad = config()
        bad["graph_construction"]["smoothing_method"] = "IDF"
        with self.assertRaisesRegex(NotImplementedError, "weighted"):
            dyn_graph_generation(bad)
        bad = config()
        bad["meta_path"] = ["rid", "title"]
        with self.assertRaisesRegex(NotImplementedError, "meta_path"):
            dyn_graph_generation(bad)


if __name__ == "__main__":
    unittest.main()

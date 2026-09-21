"""Exercise real stage processes, bounded handoffs and final incremental seeds."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "entries/batch/EmbTrai-training.py"
SPEC = importlib.util.spec_from_file_location("training_stages", ENTRY)
TRAINING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAINING)


class TrainingStagesTests(unittest.TestCase):
    def test_handoff_roundtrip_eos_and_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            bus = TRAINING.Handoff(directory, .001, .02)
            bus.put("input", 1, [("A_1", {"title": ["a", "b"]})])
            self.assertEqual(bus.take("input", 1), [("A_1", {"title": ["a", "b"]})])
            self.assertFalse(bus.path("input", 1).exists())
            bus.put("input", 2, None)
            self.assertIsNone(bus.take("input", 2))
            with self.assertRaises(TimeoutError):
                bus.take("input", 3)

    def run_pipeline(self, directory, missing_source=False):
        source = directory / "records.jsonl"
        if not missing_source:
            source.write_text("".join(json.dumps({"source_id": f"A_{i}", "title": f"shared token{i}"}) + "\n" for i in range(7)))
        config = {
            "data_source_A": str(source), "version_name": "test",
            "record_ids": {"source_field": "source_id", "left_prefix": "A_", "right_prefix": "B_"},
            "batch_processing": {"rows_per_batch": 3, "poll_interval_seconds": .01, "handoff_timeout_seconds": 30},
            "state_management": {"graph-dir": str(directory / "graph"),
                                 "embedding-dir": str(directory / "embedding"),
                                 "feature_index-dir": str(directory / "index")},
            "candidate_generation": {"method": "fullindexing"},
            "graph_construction": {"node_types": ["5#__tn", "5$__tt", "5$__st", "3$__idx", "1$__cid"],
                                   "flatten": "tt", "directed": False},
            "random_walk": {"walks_number": 2, "walk_length": 8, "write_walks": False},
            "embeddings_training": {"n_dimensions": 8, "epochs": 1, "inc_epochs": 1, "min_count": 0},
        }
        cfg_path = directory / "config.yaml"
        cfg_path.write_text(yaml.safe_dump(config))
        env = dict(os.environ, PYTHONPATH=str(ROOT), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
        processes, logs = {}, {}
        try:
            # Consumers start before the producer to exercise waiting and EOS propagation.
            for stage in reversed(TRAINING.STAGES):
                logs[stage] = (directory / f"{stage}.log").open("w")
                processes[stage] = subprocess.Popen(
                    [sys.executable, str(ENTRY), "--stage", stage, "--config", str(cfg_path),
                     "--workload", str(directory / "run")],
                    cwd=directory, env=env, stdout=logs[stage], stderr=subprocess.STDOUT,
                )
            codes = {stage: proc.wait(timeout=60) for stage, proc in processes.items()}
        finally:
            for proc in processes.values():
                if proc.poll() is None:
                    proc.kill()
                proc.wait()
            for stream in logs.values():
                stream.close()
        output = {stage: (directory / f"{stage}.log").read_text() for stage in processes}
        return codes, output

    def test_three_windows_six_processes_and_final_seeds(self):
        from igraph import Graph
        from models.embedding_model import EmbeddingModel
        from models.cg_index import CGIndex

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            codes, output = self.run_pipeline(directory)
            self.assertTrue(all(code == 0 for code in codes.values()), output)
            ids = {f"A_{i}" for i in range(7)}
            graph = Graph.Read_GraphML(str(directory / "graph/test/graph.graphml"))
            self.assertTrue(ids.issubset(set(graph.vs["name"])))
            model = EmbeddingModel.load(str(directory / "embedding/test/embedding.emb"))
            self.assertTrue(ids.issubset(model.wv.key_to_index))
            index = CGIndex.from_disk({"candidate_generation": {"method": "fullindexing"}}, str(directory / "index/test"))
            self.assertEqual(set(index.query(None)), ids)
            for stage in TRAINING.STAGES:
                self.assertEqual(output[stage].count("operation=compute start"), 3, output[stage])
            # No history of DataFrames, graphs or sequences accumulates on the handoff volume.
            self.assertEqual(list((directory / "run/communication/embedding-training").iterdir()), [])

    def test_producer_failure_stops_all_consumers(self):
        with tempfile.TemporaryDirectory() as tmp:
            codes, output = self.run_pipeline(Path(tmp), missing_source=True)
            self.assertTrue(all(code != 0 for code in codes.values()), output)
            for stage in TRAINING.STAGES[1:]:
                self.assertIn("Training peer failed", output[stage])

    def test_manifest_starts_six_independent_pods(self):
        repo = ROOT.parents[1]
        workflow = yaml.safe_load((repo / "k8s/pipeline/batch/pipeline.yaml").read_text())
        templates = {t["name"]: t for t in workflow["spec"]["templates"]}
        tasks = templates["embedding-training-dag"]["dag"]["tasks"]
        self.assertEqual({t["name"] for t in tasks}, set(TRAINING.STAGES))
        for task in tasks:
            self.assertNotIn("depends", task)
            self.assertNotIn("dependencies", task)
            container = templates[task["template"]]["container"]
            self.assertEqual(container["args"], ["--stage", task["name"], "--workload", "{{workflow.name}}"])


if __name__ == "__main__":
    unittest.main()

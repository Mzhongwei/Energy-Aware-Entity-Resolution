import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import yaml

from models.embedding_model import EmbeddingModel
from pipeline.embedding_training import initialize_embeddings, train_embeddings


class GensimEmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "embeddings_training": {
                "n_dimensions": 8,
                "window_size": 2,
                "negative": 2,
                "epochs": 1,
                "inc_epochs": 1,
                "min_count": 1,
                "workers": 1,
                "seed": 1729,
                "max_checkpoint_lead": 2,
            }
        }

    def test_incremental_training_roundtrip(self):
        model = train_embeddings(
            self.config,
            initialize_embeddings(self.config),
            [["A", "shared", "B"], ["A", "shared", "C"]],
        )
        self.assertEqual(model.framework, "gensim")
        self.assertEqual(model.train_calls, 1)

        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "embedding.emb")
            model.save(path)
            restored = EmbeddingModel.load(path)
            restored = train_embeddings(
                self.config,
                restored,
                [["D", "shared", "A"], ["D", "shared", "B"]],
            )
            self.assertEqual(restored.train_calls, 2)
            self.assertTrue({"A", "B", "C", "D"}.issubset(restored.wv.key_to_index))

    def test_rejects_pytorch_configuration_and_checkpoint(self):
        with self.assertRaisesRegex(ValueError, "removed PyTorch"):
            initialize_embeddings({"embeddings_training": {"device": "cuda"}})

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "embedding.emb"
            path.write_bytes(b"not a Gensim model")
            Path(f"{path}.meta.json").write_text(json.dumps({"framework": "pytorch"}))
            with self.assertRaisesRegex(ValueError, "retrain"):
                EmbeddingModel.load(str(path))

    def test_example_config_initializes_without_forwarding_stage_controls(self):
        path = Path(__file__).resolve().parents[1] / "config/examples/config-embedding.yaml"
        config = yaml.safe_load(path.read_text())
        self.assertIn("max_checkpoint_lead", config["embeddings_training"])
        model = initialize_embeddings(config)
        self.assertEqual(model.workers, config["embeddings_training"]["workers"])
        self.assertEqual(model.wv.vector_size, config["embeddings_training"]["n_dimensions"])

    def test_unknown_stage_key_still_rejected(self):
        self.config["embeddings_training"]["max_checkpoint_leed"] = 2
        with self.assertRaisesRegex(ValueError, "Unknown embeddings_training keys.*max_checkpoint_leed"):
            initialize_embeddings(self.config)

    # Three windows of walks over a vocabulary that grows each window, large enough that
    # training really moves the weights (Gensim's downsampling is off for these tests).
    WINDOWS = tuple(
        [
            [f"t{(start * 7 + step * 3) % (20 + 10 * window)}" for step in range(12)]
            for start in range(40 * window, 40 * window + 40)
        ]
        for window in range(3)
    )

    def fresh_model(self):
        self.config["embeddings_training"]["sampling_factor"] = 0
        return initialize_embeddings(self.config)

    def train_windows(self, model, windows):
        for sequences in windows:
            model = train_embeddings(self.config, model, sequences)
        return model

    def test_checkpoint_arrays_are_separate_and_memory_mappable(self):
        model = self.train_windows(self.fresh_model(), self.WINDOWS)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "embedding.emb")
            model.save(path)
            arrays = sorted(name for name in os.listdir(directory) if name.endswith(".npy"))
            self.assertIn("embedding.emb.wv.vectors.npy", arrays)
            self.assertIn("embedding.emb.syn1neg.npy", arrays)
            self.assertFalse([name for name in os.listdir(directory) if name.endswith(".tmp")])

            full = EmbeddingModel.load(path)
            mapped = EmbeddingModel.load(path, mmap="r")
            self.assertIsInstance(mapped.wv.vectors, np.memmap)
            self.assertFalse(mapped.wv.vectors.flags.writeable)
            self.assertEqual(mapped.wv.key_to_index, model.wv.key_to_index)
            self.assertGreater(len(model.wv.key_to_index), 30)
            np.testing.assert_array_equal(np.asarray(mapped.wv.vectors), model.wv.vectors)
            np.testing.assert_array_equal(full.wv.vectors, model.wv.vectors)
            np.testing.assert_array_equal(full.syn1neg, model.syn1neg)

    def test_resume_from_checkpoint_trains_exactly_like_an_uninterrupted_run(self):
        uninterrupted = self.train_windows(self.fresh_model(), self.WINDOWS)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "embedding.emb")
            self.train_windows(self.fresh_model(), self.WINDOWS[:2]).save(path)
            resumed = self.train_windows(EmbeddingModel.load(path), self.WINDOWS[2:])
        self.assertGreater(float(np.abs(uninterrupted.syn1neg).sum()), 0.0)  # training happened
        self.assertEqual(resumed.train_calls, uninterrupted.train_calls)
        self.assertEqual(resumed.wv.key_to_index, uninterrupted.wv.key_to_index)
        np.testing.assert_array_equal(resumed.wv.vectors, uninterrupted.wv.vectors)
        np.testing.assert_array_equal(resumed.syn1neg, uninterrupted.syn1neg)

    def test_overwrite_drops_arrays_of_the_previous_save(self):
        model = self.train_windows(self.fresh_model(), self.WINDOWS)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "embedding.emb")
            model.save(path)
            Path(f"{path}.old_attribute.npy").write_bytes(b"stale")
            model.save(path)
            self.assertFalse(os.path.exists(f"{path}.old_attribute.npy"))
            np.testing.assert_array_equal(EmbeddingModel.load(path).wv.vectors, model.wv.vectors)

    def test_single_file_checkpoints_still_load(self):
        model = self.train_windows(self.fresh_model(), self.WINDOWS)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "embedding.emb")
            with open(path, "wb") as stream:  # the former format: all arrays in the pickle
                model.model.save(stream)
            for mmap in (None, "r"):
                np.testing.assert_array_equal(
                    EmbeddingModel.load(path, mmap=mmap).wv.vectors, model.wv.vectors,
                )

    def test_memory_mapped_model_scores_like_a_full_load(self):
        from pipeline.calculating_similarity import score_mutual_topk_candidate_pairs

        model = self.train_windows(self.fresh_model(), self.WINDOWS)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "embedding.emb")
            model.save(path)
            pairs = [(["t1", "t2", "t3", "t4"], ["t21", "t22", "t25"]), ("t5", ["t30", "t31"])]
            for top_k in (1, 2):
                self.assertEqual(
                    score_mutual_topk_candidate_pairs(EmbeddingModel.load(path, mmap="r"), pairs, top_k=top_k),
                    score_mutual_topk_candidate_pairs(EmbeddingModel.load(path), pairs, top_k=top_k),
                )


if __name__ == "__main__":
    unittest.main()

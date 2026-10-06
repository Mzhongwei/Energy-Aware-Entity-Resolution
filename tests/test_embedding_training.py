import json
from pathlib import Path
import tempfile
import unittest

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


if __name__ == "__main__":
    unittest.main()

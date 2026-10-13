"""Gensim-backed, incrementally trainable word embeddings."""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import shutil
import tempfile

from gensim.models import Doc2Vec, FastText, Word2Vec


class EmbeddingModel:
    """Persistence-friendly wrapper around the supported Gensim models.

    The wrapper preserves the ``wv`` API used by downstream stages and the configuration
    needed to resume incremental training. Gensim owns the optimizer/training state.
    """

    framework = "gensim"
    MODEL_MAP = {
        "word2vec": Word2Vec,
        "doc2vec": Doc2Vec,
        "fasttext": FastText,
    }

    def __init__(
        self,
        dimensions=128,
        window_size=3,
        negative=5,
        epochs=5,
        min_count=0,
        training_algorithm="word2vec",
        learning_method="skipgram",
        workers=None,
        sampling_factor=0.001,
        seed=1729,
        alpha=0.025,
        min_alpha=0.0001,
        shrink_windows=True,
        train_calls=0,
        model=None,
    ):
        self.dimensions = int(dimensions)
        self.window_size = int(window_size)
        self.negative = int(negative)
        self.epochs = int(epochs)
        self.min_count = int(min_count)
        self.training_algorithm = str(training_algorithm).strip().lower()
        self.learning_method = str(learning_method).strip().lower()
        self.workers = int(mp.cpu_count() if workers is None else workers)
        self.sampling_factor = float(sampling_factor)
        self.seed = int(seed)
        self.alpha = float(alpha)
        self.min_alpha = float(min_alpha)
        self.shrink_windows = bool(shrink_windows)
        self.train_calls = int(train_calls)

        if self.training_algorithm not in self.MODEL_MAP:
            raise ValueError(
                "training_algorithm must be one of: "
                + ", ".join(sorted(self.MODEL_MAP))
            )
        if self.learning_method not in {"skipgram", "cbow"}:
            raise ValueError("learning_method must be skipgram or cbow")
        if self.workers < 1:
            raise ValueError("embeddings_training.workers must be at least 1")

        self.model = model if model is not None else self._build_model()

    def _common_arguments(self):
        return {
            "vector_size": self.dimensions,
            "window": self.window_size,
            "min_count": self.min_count,
            "sg": 1 if self.learning_method == "skipgram" else 0,
            "workers": self.workers,
            "sample": self.sampling_factor,
            "negative": self.negative,
            "epochs": self.epochs,
            "seed": self.seed,
            "alpha": self.alpha,
            "min_alpha": self.min_alpha,
            "shrink_windows": self.shrink_windows,
        }

    def _build_model(self):
        return self.MODEL_MAP[self.training_algorithm](**self._common_arguments())

    @property
    def wv(self):
        return self.model.wv

    @property
    def corpus_count(self):
        return self.model.corpus_count

    def build_vocab(self, sequences, update=False):
        return self.model.build_vocab(sequences, update=update)

    def train(self, sequences, total_examples=None, epochs=None):
        result = self.model.train(
            sequences,
            total_examples=total_examples,
            epochs=self.epochs if epochs is None else int(epochs),
        )
        self.train_calls += 1
        return result

    def get_model(self):
        return self.model

    def config_dict(self):
        return {
            "framework": self.framework,
            "dimensions": self.dimensions,
            "window_size": self.window_size,
            "negative": self.negative,
            "epochs": self.epochs,
            "min_count": self.min_count,
            "training_algorithm": self.training_algorithm,
            "learning_method": self.learning_method,
            "workers": self.workers,
            "sampling_factor": self.sampling_factor,
            "seed": self.seed,
            "alpha": self.alpha,
            "min_alpha": self.min_alpha,
            "shrink_windows": self.shrink_windows,
            "train_calls": self.train_calls,
        }

    def save(self, path):
        """Save the full, trainable model: a pickle at ``path`` and every NumPy array beside
        it as ``<path>.<attribute>.npy``, so a reader can memory-map the arrays
        (``load(path, mmap="r")``) instead of reading them whole.

        Files are staged in a temporary directory and renamed into place with the pickle
        last, so the pickle only ever names arrays of its own save. Pipeline checkpoints go
        to a fresh directory published by one rename; an in-place overwrite of ``path``
        assumes no concurrent reader, as before.
        """
        directory = os.path.dirname(path) or "."
        os.makedirs(directory, exist_ok=True)
        name = os.path.basename(path)
        staging = tempfile.mkdtemp(prefix=f".{name}.", suffix=".tmp", dir=directory)
        try:
            staged = os.path.join(staging, name)
            # sep_limit=0: every array goes to its own .npy, whatever its size.
            self.model.save(staged, sep_limit=0)
            with open(f"{staged}.meta.json", "w", encoding="utf-8") as stream:
                json.dump(self.config_dict(), stream, ensure_ascii=False, indent=2)
            arrays = sorted(entry for entry in os.listdir(staging) if entry.endswith(".npy"))
            stale = [
                entry for entry in os.listdir(directory)
                if entry.startswith(f"{name}.") and entry.endswith(".npy") and entry not in arrays
            ]
            for entry in arrays:
                os.replace(os.path.join(staging, entry), os.path.join(directory, entry))
            os.replace(f"{staged}.meta.json", f"{path}.meta.json")
            os.replace(staged, path)
            for entry in stale:
                os.remove(os.path.join(directory, entry))
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    @classmethod
    def load(cls, path, mmap=None):
        """Load a model saved by ``save``, or an older single-file one.

        ``mmap="r"`` memory-maps the saved arrays read-only: only the rows a reader touches
        are read from disk. Such a model can be scored but not trained; training needs the
        default full load.
        """
        metadata_path = f"{path}.meta.json"
        metadata = {}
        if os.path.isfile(metadata_path):
            with open(metadata_path, encoding="utf-8") as stream:
                metadata = json.load(stream)
            framework = metadata.get("framework")
            if framework not in {None, cls.framework}:
                raise ValueError(
                    f"{path} is a {framework!r} embedding checkpoint; "
                    "Gensim is now required, so retrain the embedding model"
                )

        requested_algorithm = str(metadata.get("training_algorithm", "")).lower()
        algorithms = (
            [requested_algorithm]
            if requested_algorithm in cls.MODEL_MAP
            else list(cls.MODEL_MAP)
        )
        model = None
        training_algorithm = None
        for algorithm in algorithms:
            try:
                model = cls.MODEL_MAP[algorithm].load(path, mmap=mmap)
                training_algorithm = algorithm
                break
            except Exception:
                continue
        if model is None:
            raise ValueError(f"Unable to load Gensim embedding model from {path}")

        return cls(
            dimensions=metadata.get("dimensions", getattr(model, "vector_size", 128)),
            window_size=metadata.get("window_size", getattr(model, "window", 3)),
            negative=metadata.get("negative", getattr(model, "negative", 5)),
            epochs=metadata.get("epochs", getattr(model, "epochs", 5)),
            min_count=metadata.get("min_count", getattr(model, "min_count", 0)),
            training_algorithm=training_algorithm,
            learning_method=metadata.get(
                "learning_method", "skipgram" if getattr(model, "sg", 1) else "cbow"
            ),
            workers=metadata.get("workers", getattr(model, "workers", mp.cpu_count())),
            sampling_factor=metadata.get("sampling_factor", getattr(model, "sample", 0.001)),
            seed=metadata.get("seed", getattr(model, "seed", 1729)),
            alpha=metadata.get("alpha", getattr(model, "alpha", 0.025)),
            min_alpha=metadata.get("min_alpha", getattr(model, "min_alpha", 0.0001)),
            shrink_windows=metadata.get(
                "shrink_windows", getattr(model, "shrink_windows", True)
            ),
            train_calls=metadata.get("train_calls", 0),
            model=model,
        )

    def __getattr__(self, item):
        return getattr(self.model, item)


__all__ = ["EmbeddingModel"]

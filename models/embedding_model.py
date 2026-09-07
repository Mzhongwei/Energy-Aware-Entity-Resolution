import multiprocessing as mp
import json
import os

from gensim.models import Doc2Vec, FastText, Word2Vec


class EmbeddingModel:
    MODEL_MAP = {
        "word2vec": Word2Vec,
        "doc2vec": Doc2Vec,
        "fasttext": FastText,
    }

    def __init__(
        self,
        dimensions,
        window_size,
        negative,
        epochs,
        min_count,
        training_algorithm="word2vec",
        learning_method="skipgram",
        workers=mp.cpu_count(),
        sampling_factor=0.001,
        model=None,
    ):
        self.dimensions = dimensions
        self.window_size = window_size
        self.negative = negative
        self.epochs = epochs
        self.min_count = min_count
        self.training_algorithm = training_algorithm.lower()
        self.learning_method = learning_method
        self.workers = workers
        self.sampling_factor = sampling_factor
        self.model = model or self._build_model()

    def _get_sg(self):
        if self.learning_method == "skipgram":
            return 1
        if self.learning_method == "CBOW":
            return 0
        raise ValueError("Unknown learning method {}".format(self.learning_method))

    def _build_model(self):
        sg = self._get_sg()

        if self.training_algorithm == "word2vec":
            return Word2Vec(
                vector_size=self.dimensions,
                window=self.window_size,
                min_count=self.min_count,
                sg=sg,
                workers=self.workers,
                sample=self.sampling_factor,
                negative=self.negative,
                epochs=self.epochs,
            )

        if self.training_algorithm == "doc2vec":
            return Doc2Vec(
                vector_size=self.dimensions,
                window=self.window_size,
                min_count=self.min_count,
                sg=sg,
                workers=self.workers,
                sample=self.sampling_factor,
                negative=self.negative,
                epochs=self.epochs,
            )

        if self.training_algorithm == "fasttext":
            print("Using Fasttext")
            return FastText(
                vector_size=self.dimensions,
                workers=self.workers,
                min_count=self.min_count,
                window=self.window_size,
                negative=self.negative,
                epochs=self.epochs,
            )

        raise ValueError("Unknown training algorithm {}".format(self.training_algorithm))

    def get_model(self):
        return self.model

    def save(self, path):
        model_dir = os.path.dirname(path)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)

        temp_path = f"{path}.{os.getpid()}.tmp"
        temp_meta_path = f"{temp_path}.meta.json"
        try:
            # A file handle keeps Gensim arrays in one file, so os.replace can
            # publish the complete model atomically for concurrent readers.
            with open(temp_path, "wb") as model_file:
                self.model.save(model_file)
            with open(temp_meta_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "dimensions": self.dimensions,
                        "window_size": self.window_size,
                        "negative": self.negative,
                        "epochs": self.epochs,
                        "min_count": self.min_count,
                        "training_algorithm": self.training_algorithm,
                        "learning_method": self.learning_method,
                        "workers": self.workers,
                        "sampling_factor": self.sampling_factor,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            os.replace(temp_meta_path, f"{path}.meta.json")
            os.replace(temp_path, path)
        except Exception:
            for temp_file in (temp_path, temp_meta_path):
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            raise

    @classmethod
    def load(cls, path):
        meta_path = f"{path}.meta.json"
        metadata = {}
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

        model = None
        training_algorithm = metadata.get("training_algorithm")

        if training_algorithm in cls.MODEL_MAP:
            model = cls.MODEL_MAP[training_algorithm].load(path)
        else:
            for algorithm, model_cls in cls.MODEL_MAP.items():
                try:
                    model = model_cls.load(path)
                    training_algorithm = algorithm
                    break
                except Exception:
                    continue

        if model is None:
            raise ValueError(f"Unable to load embedding model from {path}")

        return cls(
            dimensions=metadata.get("dimensions", getattr(model, "vector_size", 0)),
            window_size=metadata.get("window_size", getattr(model, "window", 0)),
            negative=metadata.get("negative", getattr(model, "negative", 0)),
            epochs=metadata.get("epochs", getattr(model, "epochs", 0)),
            min_count=metadata.get("min_count", getattr(model, "min_count", 0)),
            training_algorithm=training_algorithm or "word2vec",
            learning_method=metadata.get("learning_method", "skipgram"),
            workers=metadata.get("workers", mp.cpu_count()),
            sampling_factor=metadata.get("sampling_factor", 0.001),
            model=model,
        )

    def __getattr__(self, item):
        return getattr(self.model, item)

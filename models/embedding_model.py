"""PyTorch skip-gram negative-sampling (SGNS) embeddings with a ``wv`` keyed-vector API."""
from __future__ import annotations

from collections import Counter
import json
import os

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class KeyedVectors:
    def __init__(self, owner: "EmbeddingModel"):
        self._owner = owner

    @property
    def key_to_index(self):
        return self._owner.key_to_index

    @property
    def index_to_key(self):
        return self._owner.index_to_key

    @property
    def vector_size(self):
        return self._owner.dimensions

    @property
    def vectors(self):
        return self._owner.input_embeddings.weight.detach().cpu().numpy()

    def __len__(self):
        return len(self._owner.index_to_key)

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.vectors[self.key_to_index[key]]
        return np.asarray([self[item] for item in key], dtype=np.float32)


class EmbeddingModel:
    """Incrementally trainable skip-gram negative-sampling embedding model.

    ``device`` is "auto" (CUDA when available, else CPU), "cpu" or "cuda". The requested value
    is what gets saved, and it is resolved only when training, so a checkpoint trained on a GPU
    still loads in a CPU-only process for scoring.

    Randomness never touches the process-global RNGs. Every stream is derived from
    ``(seed, train_calls, purpose, epoch)``; ``train_calls`` is persisted with the model, so
    successive windows draw statistically independent streams while a rerun with the same
    seed and data is reproducible, including after a restart from a checkpoint.
    """

    framework = "pytorch"
    _INIT_STREAM, _PAIR_STREAM, _NEGATIVE_STREAM = 0, 1, 2

    def __init__(
        self,
        dimensions=128,
        window_size=3,
        negative=5,
        epochs=5,
        min_count=10,
        sampling_factor=0.001,
        batch_size=4096,
        learning_rate=0.01,
        min_learning_rate=0.0005,
        seed=1729,
        device="auto",
        dynamic_window=True,
        use_subsampling=True,
        optimizer="adam",
        train_calls=0,
    ):
        self.dimensions = int(dimensions)
        self.window_size = int(window_size)
        self.negative = int(negative)
        self.epochs = int(epochs)
        self.min_count = int(min_count)
        self.sampling_factor = float(sampling_factor)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.min_learning_rate = float(min_learning_rate)
        self.seed = int(seed)
        self.device = str(device or "auto").lower()
        if self.device not in {"auto", "cpu", "cuda"} and not self.device.startswith("cuda:"):
            raise ValueError("embeddings_training.device must be auto, cpu, cuda or cuda:<index>")
        self.dynamic_window = bool(dynamic_window)
        self.use_subsampling = bool(use_subsampling)
        self.optimizer_name = str(optimizer).lower()
        self.train_calls = int(train_calls)
        self.index_to_key: list[str] = []
        self.key_to_index: dict[str, int] = {}
        self.token_counts: Counter[str] = Counter()
        self.corpus_count = 0
        self.input_embeddings, self.output_embeddings = self._initialize_rows(0)
        self.wv = KeyedVectors(self)

    def _resolve_device(self):
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError(f"embeddings_training.device is {self.device}, but CUDA is unavailable")
        return torch.device(self.device)

    def _seed_sequence(self, purpose, epoch=0):
        return np.random.SeedSequence(self.seed, spawn_key=(self.train_calls, purpose, epoch))

    def _numpy_rng(self, purpose, epoch=0):
        return np.random.default_rng(self._seed_sequence(purpose, epoch))

    def _torch_generator(self, purpose, epoch=0):
        seed = int(self._seed_sequence(purpose, epoch).generate_state(1, dtype=np.uint64)[0])
        return torch.Generator(device="cpu").manual_seed(seed)

    def __bool__(self):
        return True

    def _initialize_rows(self, size):
        # Passing _weight skips nn.Embedding's default init, which would draw from the
        # global torch RNG.
        bound = 0.5 / max(1, self.dimensions)
        input_weight = torch.empty(size, self.dimensions).uniform_(
            -bound, bound, generator=self._torch_generator(self._INIT_STREAM),
        )
        output_weight = torch.zeros(size, self.dimensions)
        return (nn.Embedding(size, self.dimensions, _weight=input_weight),
                nn.Embedding(size, self.dimensions, _weight=output_weight))

    def _resize_embeddings(self, new_size):
        old_input = self.input_embeddings.weight.detach().cpu() if self.input_embeddings.num_embeddings else None
        old_output = self.output_embeddings.weight.detach().cpu() if self.output_embeddings.num_embeddings else None
        new_input, new_output = self._initialize_rows(new_size)
        if old_input is not None:
            with torch.no_grad():
                new_input.weight[: len(old_input)].copy_(old_input)
                new_output.weight[: len(old_output)].copy_(old_output)
        self.input_embeddings = new_input
        self.output_embeddings = new_output

    def build_vocab(self, sequences, update=False):
        batch_counts = Counter()
        corpus_count = 0
        for sequence in sequences:
            batch_counts.update(map(str, sequence))
            corpus_count += 1
        if not update:
            self.token_counts = Counter()
            self.index_to_key = []
            self.key_to_index = {}
        self.token_counts.update(batch_counts)
        new_tokens = sorted(
            token for token, count in self.token_counts.items()
            if count >= self.min_count and token not in self.key_to_index
        )
        if new_tokens:
            start = len(self.index_to_key)
            self.index_to_key.extend(new_tokens)
            self.key_to_index.update({token: start + offset for offset, token in enumerate(new_tokens)})
            self._resize_embeddings(len(self.index_to_key))
        self.corpus_count = corpus_count

    def _encode(self, sequences):
        """Flatten a corpus into int32 vocabulary ids plus per-token sequence ids; OOV is dropped."""
        lookup = self.key_to_index.get
        id_chunks, owner_chunks = [], []
        sequence_count = 0
        for sequence in sequences:
            encoded = np.fromiter((lookup(str(token), -1) for token in sequence), dtype=np.int32)
            encoded = encoded[encoded >= 0]
            id_chunks.append(encoded)
            owner_chunks.append(np.full(len(encoded), sequence_count, dtype=np.int32))
            sequence_count += 1
        if not id_chunks:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32), 0
        return np.concatenate(id_chunks), np.concatenate(owner_chunks), sequence_count

    def _keep_probabilities(self, counts):
        """Word2vec subsampling keep-probability per vocabulary id, computed once per train()."""
        if not self.use_subsampling or self.sampling_factor <= 0:
            return None
        frequency = counts / counts.sum()
        return np.minimum(1.0, (np.sqrt(frequency / self.sampling_factor) + 1.0)
                          * self.sampling_factor / frequency)

    def _training_pairs(self, ids, owners, keep_probabilities, epoch):
        rng = self._numpy_rng(self._PAIR_STREAM, epoch)
        if keep_probabilities is not None:
            kept = rng.random(len(ids)) < keep_probabilities[ids]
            ids, owners = ids[kept], owners[kept]
        # The window is applied after subsampling, over each sequence's kept tokens.
        if self.dynamic_window:
            radius = rng.integers(1, self.window_size + 1, size=len(ids))
        else:
            radius = np.full(len(ids), self.window_size)
        centers, contexts = [], []
        for offset in range(1, self.window_size + 1):
            left = np.arange(len(ids) - offset)
            left = left[owners[left] == owners[left + offset]]
            right = left + offset
            forward = left[radius[left] >= offset]      # context lies after the center
            backward = right[radius[right] >= offset]   # context lies before the center
            centers += [ids[forward], ids[backward]]
            contexts += [ids[forward + offset], ids[backward - offset]]
        centers = np.concatenate(centers) if centers else np.empty(0, dtype=np.int32)
        contexts = np.concatenate(contexts) if contexts else np.empty(0, dtype=np.int32)
        order = rng.permutation(len(centers))
        return centers[order], contexts[order]

    def _optimizer(self):
        parameters = list(self.input_embeddings.parameters()) + list(self.output_embeddings.parameters())
        if self.optimizer_name == "sgd":
            return torch.optim.SGD(parameters, lr=self.learning_rate)
        if self.optimizer_name == "adam":
            return torch.optim.Adam(parameters, lr=self.learning_rate)
        raise ValueError("pytorch.optimizer must be 'adam' or 'sgd'")

    def train(self, sequences, total_examples=None, epochs=None):
        if not self.index_to_key:
            raise ValueError("build_vocab must be called before train")
        train_epochs = int(self.epochs if epochs is None else epochs)
        device = self._resolve_device()
        self.input_embeddings.to(device)
        self.output_embeddings.to(device)
        optimizer = self._optimizer()
        counts = np.asarray([self.token_counts[token] for token in self.index_to_key], dtype=np.float64)
        negative_probabilities = torch.as_tensor(counts ** 0.75 / np.sum(counts ** 0.75), dtype=torch.float32)
        keep_probabilities = self._keep_probabilities(counts)
        ids, owners, sequence_count = self._encode(sequences)

        for epoch in range(train_epochs):
            negative_generator = self._torch_generator(self._NEGATIVE_STREAM, epoch)
            centers, contexts = self._training_pairs(ids, owners, keep_probabilities, epoch)
            if not len(centers):
                continue
            progress = epoch / max(1, train_epochs - 1)
            rate = self.learning_rate + progress * (self.min_learning_rate - self.learning_rate)
            for group in optimizer.param_groups:
                group["lr"] = rate
            for start in range(0, len(centers), self.batch_size):
                stop = min(len(centers), start + self.batch_size)
                center_ids = torch.as_tensor(centers[start:stop], device=device).long()
                context_ids = torch.as_tensor(contexts[start:stop], device=device).long()
                negative_ids = torch.multinomial(
                    negative_probabilities,
                    num_samples=(stop - start) * self.negative,
                    replacement=True,
                    generator=negative_generator,
                ).reshape(stop - start, self.negative).to(device)

                center_vectors = self.input_embeddings(center_ids)
                context_vectors = self.output_embeddings(context_ids)
                negative_vectors = self.output_embeddings(negative_ids)
                positive_score = torch.sum(center_vectors * context_vectors, dim=1)
                negative_score = torch.bmm(negative_vectors, center_vectors.unsqueeze(2)).squeeze(2)
                loss = -(F.logsigmoid(positive_score) + F.logsigmoid(-negative_score).sum(dim=1)).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        self.input_embeddings.cpu()
        self.output_embeddings.cpu()
        self.train_calls += 1
        return sequence_count

    def save(self, path):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        temporary = f"{path}.{os.getpid()}.tmp"
        metadata_temporary = f"{temporary}.meta.json"
        payload = {
            "input_weight": self.input_embeddings.weight.detach().cpu(),
            "output_weight": self.output_embeddings.weight.detach().cpu(),
            "index_to_key": self.index_to_key,
            "token_counts": dict(self.token_counts),
            "config": self.config_dict(),
        }
        try:
            torch.save(payload, temporary)
            with open(metadata_temporary, "w", encoding="utf-8") as stream:
                json.dump({"framework": self.framework, **self.config_dict()}, stream, indent=2)
            os.replace(temporary, path)
            os.replace(metadata_temporary, f"{path}.meta.json")
        except Exception:
            for candidate in (temporary, metadata_temporary):
                if os.path.exists(candidate):
                    os.remove(candidate)
            raise

    def config_dict(self):
        return {
            "dimensions": self.dimensions,
            "window_size": self.window_size,
            "negative": self.negative,
            "epochs": self.epochs,
            "min_count": self.min_count,
            "sampling_factor": self.sampling_factor,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "min_learning_rate": self.min_learning_rate,
            "seed": self.seed,
            "device": self.device,
            "dynamic_window": self.dynamic_window,
            "use_subsampling": self.use_subsampling,
            "optimizer": self.optimizer_name,
            "train_calls": self.train_calls,
        }

    @classmethod
    def load(cls, path, map_location="cpu"):
        metadata_path = f"{path}.meta.json"
        if os.path.isfile(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as stream:
                framework = json.load(stream).get("framework")
            if framework != cls.framework:
                raise ValueError(
                    f"{path} was not written by the PyTorch embedding model (framework={framework!r}); "
                    "Gensim checkpoints are no longer supported, retrain the embedding"
                )
        payload = torch.load(path, map_location=map_location, weights_only=True)
        model = cls(**payload["config"])
        model.index_to_key = [str(token) for token in payload["index_to_key"]]
        model.key_to_index = {token: index for index, token in enumerate(model.index_to_key)}
        model.token_counts = Counter({str(key): int(value) for key, value in payload["token_counts"].items()})
        model._resize_embeddings(len(model.index_to_key))
        with torch.no_grad():
            model.input_embeddings.weight.copy_(payload["input_weight"])
            model.output_embeddings.weight.copy_(payload["output_weight"])
        return model


__all__ = ["EmbeddingModel", "KeyedVectors"]

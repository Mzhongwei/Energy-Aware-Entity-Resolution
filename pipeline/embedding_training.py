from __future__ import annotations

import json
import os
from typing import Iterable, List, Optional, Sequence

import numpy as np

from models.embedding_model import EmbeddingModel

WALK_SHARD_FORMATS = ("walk-shards-v1", "walk-shards-v2")
_MATRIX_ROWS_PER_READ = 4096


def is_walk_shard_manifest(payload) -> bool:
    return isinstance(payload, dict) and payload.get("format") in WALK_SHARD_FORMATS


class WalkShardCorpus:
    """Re-iterable corpus backed by a completed random-walk shard manifest.

    v1 parts are JSONL walks. v2 parts are uint32 [walks, walk_length] .npy matrices of
    positions into the part's JSON vocabulary; every occurrence of a token yields the same
    str object, so Gensim's vocabulary lookups reuse its cached hash.
    """

    def __init__(self, manifest):
        if not is_walk_shard_manifest(manifest):
            raise ValueError("Invalid random-walk shard manifest")
        self.matrix = manifest["format"] == "walk-shards-v2"
        directory = manifest.get("directory")
        if not isinstance(directory, str) or not os.path.isabs(directory):
            raise ValueError("Random-walk shard directory must be an absolute path")
        self.directory = os.path.abspath(directory)
        self.parts = list(manifest.get("parts", []))
        self.walk_count = int(manifest.get("walk_count", 0))
        if self.walk_count < 0:
            raise ValueError("Invalid random-walk shard manifest metadata")
        counted_walks = 0
        for part in self.parts:
            for filename in _part_filenames(part, self.matrix):
                path = _shard_path(self.directory, filename)
                if not os.path.isfile(path):
                    raise FileNotFoundError(f"Random-walk shard does not exist: {path}")
            counted_walks += int(part.get("walk_count", 0))
        if counted_walks != self.walk_count:
            raise ValueError(
                f"Random-walk shard count mismatch: manifest={self.walk_count}, parts={counted_walks}"
            )

    def __len__(self):
        return self.walk_count

    def __iter__(self):
        emitted = 0
        for part in self.parts:
            path = os.path.join(self.directory, part["file"])
            walks = self._iter_matrix(part, path) if self.matrix else self._iter_jsonl(path)
            for walk in walks:
                emitted += 1
                yield walk
        if emitted != self.walk_count:
            raise ValueError(
                f"Random-walk shard contents changed: expected={self.walk_count}, actual={emitted}"
            )

    def _iter_jsonl(self, path):
        with open(path, encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                walk = json.loads(line)
                if not isinstance(walk, list):
                    raise ValueError(f"Invalid walk in {path}:{line_number}")
                yield walk

    def _iter_matrix(self, part, path):
        with open(os.path.join(self.directory, part["vocab"]), encoding="utf-8") as source:
            names = json.load(source)
        walks = np.load(path, mmap_mode="r", allow_pickle=False)
        if walks.ndim != 2 or walks.dtype != np.uint32 or walks.shape[0] != int(part["walk_count"]):
            raise ValueError(f"Random-walk shard matrix does not match its manifest: {path}")
        for offset in range(0, walks.shape[0], _MATRIX_ROWS_PER_READ):
            for row in walks[offset:offset + _MATRIX_ROWS_PER_READ].tolist():
                yield [names[position] for position in row]


def _shard_path(directory, filename):
    if not filename or os.path.basename(filename) != filename:
        raise ValueError(f"Invalid random-walk shard filename: {filename}")
    path = os.path.abspath(os.path.join(directory, filename))
    if os.path.commonpath((directory, path)) != directory:
        raise ValueError(f"Random-walk shard escapes its directory: {filename}")
    return path


def _part_filenames(part, matrix):
    if not isinstance(part, dict):
        raise ValueError(f"Invalid random-walk shard part: {part}")
    return (part.get("file"), part.get("vocab")) if matrix else (part.get("file"),)


def open_walk_sequences(payload):
    if is_walk_shard_manifest(payload):
        return WalkShardCorpus(payload)
    return payload


def cleanup_walk_shards(payload):
    """Remove only files named by an internal manifest after training is durable."""
    if not is_walk_shard_manifest(payload):
        return
    directory = payload.get("directory")
    if not isinstance(directory, str) or not os.path.isabs(directory):
        raise ValueError("Random-walk shard directory must be an absolute path")
    directory = os.path.abspath(directory)
    if not os.path.isdir(directory):
        return
    matrix = payload["format"] == "walk-shards-v2"
    for part in payload.get("parts", []):
        for filename in _part_filenames(part, matrix):
            path = _shard_path(directory, filename)
            if os.path.isfile(path):
                os.remove(path)
    manifest_path = os.path.join(directory, "manifest.json")
    if os.path.isfile(manifest_path):
        os.remove(manifest_path)
    try:
        os.rmdir(directory)
    except OSError:
        # A failed producer attempt may have left a uniquely named temporary file. It is
        # deliberately not removed here; the producer owns partial-directory recovery.
        pass
    parent = os.path.dirname(directory)
    if os.path.basename(parent) == ".walk_shards":
        try:
            os.rmdir(parent)
        except OSError:
            # Other windows may still own shard directories under the same parent.
            pass


def _embedding_config(config) -> dict:
    if not isinstance(config, dict):
        return {}
    emb_cfg = config.get("embeddings_training")
    if isinstance(emb_cfg, dict):
        return emb_cfg
    legacy_cfg = config.get("embeddings")
    if isinstance(legacy_cfg, dict):
        return legacy_cfg
    return {}


# embeddings_training key -> Gensim EmbeddingModel argument. Keys left out use defaults.
_MODEL_ARGUMENTS = {
    "n_dimensions": "dimensions",
    "window_size": "window_size",
    "negative": "negative",
    "epochs": "epochs",
    "min_count": "min_count",
    "training_algorithm": "training_algorithm",
    "learning_method": "learning_method",
    "workers": "workers",
    "sampling_factor": "sampling_factor",
    "seed": "seed",
    "alpha": "alpha",
    "min_alpha": "min_alpha",
    "shrink_windows": "shrink_windows",
}
_TRAINING_KEYS = {"inc_epochs"}
_PYTORCH_KEYS = {
    "batch_size", "device", "dynamic_window", "framework", "learning_rate",
    "min_learning_rate", "optimizer", "pair_chunk_tokens", "pair_generation",
    "pytorch", "use_subsampling",
}


def initialize_embeddings(config):
    emb_cfg = _embedding_config(config)
    pytorch_keys = sorted(_PYTORCH_KEYS & set(emb_cfg))
    if pytorch_keys:
        raise ValueError(
            f"embeddings_training keys {pytorch_keys} belong to the removed PyTorch "
            "implementation; configure Gensim with workers/training_algorithm/learning_method"
        )
    unknown = sorted(set(emb_cfg) - set(_MODEL_ARGUMENTS) - _TRAINING_KEYS)
    if unknown:
        raise ValueError(f"Unknown embeddings_training keys: {unknown}")
    return EmbeddingModel(**{
        argument: emb_cfg[key] for key, argument in _MODEL_ARGUMENTS.items() if emb_cfg.get(key) is not None
    })


def retrain_embeddings(config, model, sequences):
    if (model is None) or (not model):
        model = initialize_embeddings(config)

    emb_cfg = _embedding_config(config)
    train_epochs = int(emb_cfg.get("inc_epochs", emb_cfg.get("epochs", getattr(model, "epochs", 5))))

    if len(model.wv.key_to_index) == 0:
        model.build_vocab(sequences)
        total_examples = model.corpus_count
    else:
        model.build_vocab(sequences, update=True)
        total_examples = len(sequences)

    model.train(sequences, total_examples=total_examples, epochs=train_epochs)
    return model


train_embeddings = retrain_embeddings


def load_or_create_model(config, model_path: str):
    """Load a persisted embedding model if present, else initialize a fresh one."""
    if os.path.isfile(model_path):
        return EmbeddingModel.load(model_path)
    return initialize_embeddings(config)


__all__ = [
    "WalkShardCorpus",
    "cleanup_walk_shards",
    "is_walk_shard_manifest",
    "initialize_embeddings",
    "open_walk_sequences",
    "retrain_embeddings",
    "train_embeddings",
    "load_or_create_model",
]

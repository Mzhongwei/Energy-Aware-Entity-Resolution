from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

from models.embedding_model import EmbeddingModel


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


def initialize_embeddings(config) -> EmbeddingModel:
    emb_cfg = _embedding_config(config)
    return EmbeddingModel(
        dimensions=int(emb_cfg.get("n_dimensions", 300)),
        window_size=int(emb_cfg.get("window_size", 3)),
        negative=int(emb_cfg.get("negative", 5)),
        epochs=int(emb_cfg.get("epochs", 5)),
        min_count=int(emb_cfg.get("min_count", 0)),
        training_algorithm=str(emb_cfg.get("training_algorithm", "word2vec")),
        learning_method=str(emb_cfg.get("learning_method", "skipgram")),
        sampling_factor=float(emb_cfg.get("sampling_factor", 0.001)),
    )


def retrain_embeddings(config, model: Optional[EmbeddingModel], sequences) -> EmbeddingModel:
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


__all__ = [
    "initialize_embeddings",
    "retrain_embeddings",
    "train_embeddings",
]

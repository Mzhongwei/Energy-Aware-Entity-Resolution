from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

from models import EmbeddingModel


def initialize_embeddings(config) -> EmbeddingModel:
    emb_cfg = config.get("embeddings", {}) if isinstance(config, dict) else {}
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


def _normalize_walks(sequences: Optional[Iterable]) -> List[List[str]]:
    if sequences is None:
        return []

    walks: List[List[str]] = []
    for seq in sequences:
        if seq is None:
            continue
        if isinstance(seq, str):
            tokens = seq.split()
        elif isinstance(seq, Sequence):
            tokens = [str(token) for token in seq if token is not None and str(token) != ""]
        else:
            tokens = [str(seq)]
        if tokens:
            walks.append(tokens)
    return walks


def retrain_embeddings(config, model: Optional[EmbeddingModel], sequences) -> EmbeddingModel:
    walks = _normalize_walks(sequences)
    if model is None:
        model = initialize_embeddings(config)

    if not walks:
        return model

    emb_cfg = config.get("embeddings", {}) if isinstance(config, dict) else {}
    train_epochs = int(emb_cfg.get("inc_epochs", emb_cfg.get("epochs", getattr(model, "epochs", 5))))

    if len(model.wv.key_to_index) == 0:
        model.build_vocab(walks)
        total_examples = model.corpus_count
    else:
        model.build_vocab(walks, update=True)
        total_examples = len(walks)

    model.train(walks, total_examples=total_examples, epochs=train_epochs)
    return model


train_embeddings = retrain_embeddings


__all__ = [
    "initialize_embeddings",
    "retrain_embeddings",
    "train_embeddings",
]

import numpy as np
from typing import Dict, List, Tuple


def _resolve_keyed_vectors(model):
    base_model = getattr(model, "model", model)
    if hasattr(base_model, "wv"):
        return base_model.wv
    raise ValueError("embedding_model must expose keyed vectors via `.wv`.")


def _flatten_candidate_pairs(candidate_pairs: List[Tuple[str, List[str]]]) -> List[Tuple[str, str]]:
    if not isinstance(candidate_pairs, list):
        raise ValueError("candidate_pairs must be a list of tuples.")

    flattened_pairs: List[Tuple[str, str]] = []
    for pair in candidate_pairs:
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError("Each candidate_pairs item must be a tuple of (record_id, candidate_ids).")
        record_id, candidate_ids = pair
        if not isinstance(candidate_ids, list):
            raise ValueError("candidate_ids must be a list.")
        for candidate_id in candidate_ids:
            flattened_pairs.append((str(record_id), str(candidate_id)))
    return flattened_pairs


def _cosine_similarity(vec_a, vec_b) -> float:
    a = np.asarray(vec_a, dtype=np.float32)
    b = np.asarray(vec_b, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0.0:
        raise ValueError("Cannot compute cosine similarity for zero vectors.")
    return float(np.dot(a, b) / denom)


def _score_pairs_iterative(kv, flattened_pairs: List[Tuple[str, str]]) -> List[Tuple[str, str, float]]:
    matching_pairs: List[Tuple[str, str, float]] = []
    for record_id, candidate_id in flattened_pairs:
        if record_id not in kv.key_to_index:
            raise ValueError(f"Missing embedding for record id: {record_id}")
        if candidate_id not in kv.key_to_index:
            raise ValueError(f"Missing embedding for candidate id: {candidate_id}")
        similarity = _cosine_similarity(kv[record_id], kv[candidate_id])
        matching_pairs.append((record_id, candidate_id, similarity))
    return matching_pairs


def _score_pairs_batch(kv, flattened_pairs: List[Tuple[str, str]]) -> List[Tuple[str, str, float]]:
    unique_ids = sorted({record_id for record_id, _ in flattened_pairs} | {candidate_id for _, candidate_id in flattened_pairs})
    missing_ids = [node_id for node_id in unique_ids if node_id not in kv.key_to_index]
    if missing_ids:
        raise ValueError(f"Missing embeddings for ids: {missing_ids[:5]}")

    vectors = np.asarray([kv[node_id] for node_id in unique_ids], dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms <= 0.0):
        raise ValueError("Cannot compute cosine similarity for zero vectors.")
    vectors = vectors / norms

    id_to_pos: Dict[str, int] = {node_id: idx for idx, node_id in enumerate(unique_ids)}
    left_idx = np.asarray([id_to_pos[record_id] for record_id, _ in flattened_pairs], dtype=np.int64)
    right_idx = np.asarray([id_to_pos[candidate_id] for _, candidate_id in flattened_pairs], dtype=np.int64)
    similarities = np.sum(vectors[left_idx] * vectors[right_idx], axis=1)

    return [
        (record_id, candidate_id, float(score))
        for (record_id, candidate_id), score in zip(flattened_pairs, similarities.tolist())
    ]


def score_candidate_pairs(model, candidate_pairs: List[Tuple[str, List[str]]], batch_threshold: int = 2048) -> List[Tuple[str, str, float]]:
    kv = _resolve_keyed_vectors(model)
    flattened_pairs = _flatten_candidate_pairs(candidate_pairs)
    if not flattened_pairs:
        raise ValueError("candidate_pairs is empty; cannot calculate similarity.")

    if len(flattened_pairs) >= int(batch_threshold):
        return _score_pairs_batch(kv, flattened_pairs)
    return _score_pairs_iterative(kv, flattened_pairs)


__all__ = ["score_candidate_pairs"]

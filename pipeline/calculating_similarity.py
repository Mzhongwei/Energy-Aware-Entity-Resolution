import numpy as np
from typing import Dict, List, Tuple


ScoredPair = Tuple[str, str, float]


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
            raise ValueError("Each candidate_pairs item must be a tuple of (indexed_id, query_ids).")
        indexed_id, query_ids = pair
        if isinstance(query_ids, str):
            iterable_query_ids = [query_ids]
        elif isinstance(query_ids, (list, tuple, set)):
            iterable_query_ids = query_ids
        else:
            raise ValueError("query_ids must be a string or an iterable of query ids.")
        for query_id in iterable_query_ids:
            flattened_pairs.append((str(indexed_id), str(query_id)))
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
    for indexed_id, query_id in flattened_pairs:
        if indexed_id not in kv.key_to_index:
            raise ValueError(f"Missing embedding for indexed/training id: {indexed_id}")
        if query_id not in kv.key_to_index:
            raise ValueError(f"Missing embedding for query/incremental id: {query_id}")
        similarity = _cosine_similarity(kv[indexed_id], kv[query_id])
        matching_pairs.append((indexed_id, query_id, similarity))
    return matching_pairs


def _score_pairs_batch(kv, flattened_pairs: List[Tuple[str, str]]) -> List[Tuple[str, str, float]]:
    unique_ids = sorted({indexed_id for indexed_id, _ in flattened_pairs} | {query_id for _, query_id in flattened_pairs})
    missing_ids = [node_id for node_id in unique_ids if node_id not in kv.key_to_index]
    if missing_ids:
        raise ValueError(f"Missing embeddings for ids: {missing_ids[:5]}")

    vectors = np.asarray([kv[node_id] for node_id in unique_ids], dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms <= 0.0):
        raise ValueError("Cannot compute cosine similarity for zero vectors.")
    vectors = vectors / norms

    id_to_pos: Dict[str, int] = {node_id: idx for idx, node_id in enumerate(unique_ids)}
    left_idx = np.asarray([id_to_pos[indexed_id] for indexed_id, _ in flattened_pairs], dtype=np.int64)
    right_idx = np.asarray([id_to_pos[query_id] for _, query_id in flattened_pairs], dtype=np.int64)
    similarities = np.sum(vectors[left_idx] * vectors[right_idx], axis=1)

    return [
        (indexed_id, query_id, float(score))
        for (indexed_id, query_id), score in zip(flattened_pairs, similarities.tolist())
    ]


def score_candidate_pairs(model, candidate_pairs: List[Tuple[str, List[str]]], batch_threshold: int = 2048) -> List[Tuple[str, str, float]]:
    kv = _resolve_keyed_vectors(model)
    flattened_pairs = _flatten_candidate_pairs(candidate_pairs)
    if not flattened_pairs:
        raise ValueError("candidate_pairs is empty; cannot calculate similarity.")

    if len(flattened_pairs) >= int(batch_threshold):
        return _score_pairs_batch(kv, flattened_pairs)
    return _score_pairs_iterative(kv, flattened_pairs)


def _select_topk_pairs(matching_pairs: List[ScoredPair], top_k: int) -> Tuple[Dict[str, List[Tuple[str, float]]], Dict[str, set]]:
    candidates_for_indexed: Dict[str, List[Tuple[str, float]]] = {}
    candidates_for_query: Dict[str, List[Tuple[str, float]]] = {}

    for pair in matching_pairs:
        if not isinstance(pair, tuple) or len(pair) != 3:
            raise ValueError("Each matching_pairs item must be a tuple of (left_id/indexed_id, right_id/query_id, score).")
        indexed_id, query_id, score = str(pair[0]), str(pair[1]), float(pair[2])
        candidates_for_indexed.setdefault(indexed_id, []).append((query_id, score))
        candidates_for_query.setdefault(query_id, []).append((indexed_id, score))

    # stable sort: on equal scores the first seen candidate wins (same as the former strict `>` top1)
    topk_for_indexed = {
        indexed_id: sorted(candidates, key=lambda item: -item[1])[:top_k]
        for indexed_id, candidates in candidates_for_indexed.items()
    }
    topk_for_query = {
        query_id: {indexed_id for indexed_id, _ in sorted(candidates, key=lambda item: -item[1])[:top_k]}
        for query_id, candidates in candidates_for_query.items()
    }
    return topk_for_indexed, topk_for_query


def select_mutual_top1_pairs(matching_pairs: List[ScoredPair], top_k: int = 1) -> List[ScoredPair]:
    """Keep (indexed_id, query_id) when each one is within the other's top_k (top_k=1 -> mutual top1)."""
    if not isinstance(matching_pairs, list):
        raise ValueError("matching_pairs must be a list of scored tuples.")
    if not matching_pairs:
        raise ValueError("matching_pairs is empty; cannot perform mutual top-k selection.")
    if int(top_k) < 1:
        raise ValueError("top_k must be >= 1.")

    topk_for_indexed, topk_for_query = _select_topk_pairs(matching_pairs, int(top_k))
    final_pairs: List[ScoredPair] = []
    for indexed_id, ranked_queries in topk_for_indexed.items():
        for query_id, score in ranked_queries:
            if indexed_id in topk_for_query.get(query_id, ()):
                final_pairs.append((indexed_id, query_id, score))

    return final_pairs


def score_mutual_top1_candidate_pairs(model, candidate_pairs: List[Tuple[str, List[str]]], batch_threshold: int = 2048, top_k: int = 1) -> List[ScoredPair]:
    """
    Compute local mutual top-k pairs (top_k=1 -> mutual top1) with unified semantics:

    - left_id == indexed_id == training-side id
    - right_id == query_id == incremental-side id
    """
    matching_pairs = score_candidate_pairs(model, candidate_pairs, batch_threshold=batch_threshold)
    return select_mutual_top1_pairs(matching_pairs, top_k=top_k)


__all__ = ["score_candidate_pairs", "select_mutual_top1_pairs", "score_mutual_top1_candidate_pairs"]

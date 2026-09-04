import numpy as np
from typing import Dict, Iterable, Iterator, List, Tuple


ScoredPair = Tuple[str, str, float]
CandidatePair = Tuple[str, str]
TopPairs = Dict[str, Dict[str, float]]


def _resolve_keyed_vectors(model):
    base_model = getattr(model, "model", model)
    if hasattr(base_model, "wv"):
        return base_model.wv
    raise ValueError("embedding_model must expose keyed vectors via `.wv`.")


def _iter_candidate_pairs(candidate_pairs: List[Tuple[str, List[str]]]) -> Iterator[CandidatePair]:
    if not isinstance(candidate_pairs, list):
        raise ValueError("candidate_pairs must be a list of tuples.")

    for pair in candidate_pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("Each candidate_pairs item must be a 2-element list or tuple of (indexed_id, query_ids).")
        indexed_id, query_ids = pair
        if isinstance(query_ids, str):
            iterable_query_ids = [query_ids]
        elif isinstance(query_ids, (list, tuple, set)):
            iterable_query_ids = query_ids
        else:
            raise ValueError("query_ids must be a string or an iterable of query ids.")
        for query_id in iterable_query_ids:
            yield str(indexed_id), str(query_id)


def _flatten_candidate_pairs(candidate_pairs: List[Tuple[str, List[str]]]) -> List[CandidatePair]:
    return list(_iter_candidate_pairs(candidate_pairs))


def _chunk_pairs(pairs: Iterable[CandidatePair], chunk_size: int) -> Iterator[List[CandidatePair]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    chunk: List[CandidatePair] = []
    for pair in pairs:
        chunk.append(pair)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


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


def validate_top_k(top_k: int) -> int:
    top_k = int(top_k)
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")
    return top_k


def get_mutual_top_k(config: dict, default: int = 2) -> int:
    if not isinstance(config, dict):
        return validate_top_k(default)
    section = config.get("decision_making")
    if not isinstance(section, dict):
        section = config.get("similarity")
    if not isinstance(section, dict):
        section = {}
    return validate_top_k(section.get("top_k", default))


def normalize_scored_pair(pair) -> ScoredPair:
    if not isinstance(pair, (list, tuple)) or len(pair) != 3:
        raise ValueError(
            "Each matching_pairs item must be a list or tuple of "
            "(left_id/indexed_id, right_id/query_id, score)."
        )
    return str(pair[0]), str(pair[1]), float(pair[2])


def _update_top_pairs(
    matching_pairs: Iterable[ScoredPair],
    top_for_indexed: TopPairs,
    top_for_query: TopPairs,
    top_k: int,
) -> None:
    top_k = validate_top_k(top_k)

    def update(store: TopPairs, node_id: str, other_id: str, score: float) -> None:
        candidates = store.setdefault(node_id, {})
        current = candidates.get(other_id)
        if current is None or score > current:
            candidates[other_id] = score
        if len(candidates) > top_k:
            ranked = sorted(candidates.items(), key=lambda item: (-item[1], item[0]))
            store[node_id] = dict(ranked[:top_k])

    for pair in matching_pairs:
        indexed_id, query_id, score = normalize_scored_pair(pair)
        update(top_for_indexed, indexed_id, query_id, score)
        update(top_for_query, query_id, indexed_id, score)


def _select_top_pairs(matching_pairs: List[ScoredPair], top_k: int) -> Tuple[TopPairs, TopPairs]:
    top_for_indexed: TopPairs = {}
    top_for_query: TopPairs = {}
    _update_top_pairs(matching_pairs, top_for_indexed, top_for_query, top_k)

    return top_for_indexed, top_for_query


def _mutual_topk_from_top_pairs(
    top_for_indexed: TopPairs,
    top_for_query: TopPairs,
) -> List[ScoredPair]:
    final_pairs: List[ScoredPair] = []
    for indexed_id, query_candidates in top_for_indexed.items():
        for query_id, score in query_candidates.items():
            if indexed_id in top_for_query.get(query_id, {}):
                final_pairs.append((indexed_id, query_id, score))

    return sorted(final_pairs, key=lambda pair: (pair[0], -pair[2], pair[1]))


def select_mutual_topk_pairs(matching_pairs: List[ScoredPair], top_k: int = 2) -> List[ScoredPair]:
    if not isinstance(matching_pairs, list):
        raise ValueError("matching_pairs must be a list of scored tuples.")
    if not matching_pairs:
        raise ValueError("matching_pairs is empty; cannot perform mutual top-k selection.")

    top_k = validate_top_k(top_k)
    return _mutual_topk_from_top_pairs(*_select_top_pairs(matching_pairs, top_k))


def score_mutual_topk_candidate_pairs(
    model,
    candidate_pairs: List[Tuple[str, List[str]]],
    top_k: int = 2,
    batch_threshold: int = 2048,
    chunk_size: int = 4096,
) -> List[ScoredPair]:
    """
    Compute local mutual top-k pairs with unified semantics:

    - left_id == indexed_id == training-side id
    - right_id == query_id == incremental-side id
    """
    pair_count = sum(1 for _ in _iter_candidate_pairs(candidate_pairs))
    if pair_count == 0:
        raise ValueError("candidate_pairs is empty; cannot calculate similarity.")

    top_k = validate_top_k(top_k)
    kv = _resolve_keyed_vectors(model)
    use_batch = pair_count >= int(batch_threshold)
    top_for_indexed: TopPairs = {}
    top_for_query: TopPairs = {}

    for chunk in _chunk_pairs(_iter_candidate_pairs(candidate_pairs), int(chunk_size)):
        scored_chunk = _score_pairs_batch(kv, chunk) if use_batch else _score_pairs_iterative(kv, chunk)
        _update_top_pairs(scored_chunk, top_for_indexed, top_for_query, top_k)

    return _mutual_topk_from_top_pairs(top_for_indexed, top_for_query)


__all__ = [
    "score_candidate_pairs",
    "get_mutual_top_k",
    "normalize_scored_pair",
    "validate_top_k",
    "select_mutual_topk_pairs",
    "score_mutual_topk_candidate_pairs",
]

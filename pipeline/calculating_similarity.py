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


def get_similarity_top_k(config: dict, default: int = 1) -> int:
    """Mutual top-k used by the similarity stage, read from ``calculating_similarity.top_k``."""
    section = config.get("calculating_similarity") if isinstance(config, dict) else None
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


def _iter_candidate_groups(candidate_pairs: List[Tuple[str, List[str]]]) -> Iterator[Tuple[str, List[str]]]:
    if not isinstance(candidate_pairs, list):
        raise ValueError("candidate_pairs must be a list of tuples.")

    for pair in candidate_pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("Each candidate_pairs item must be a 2-element list or tuple of (indexed_id, query_ids).")
        indexed_id, query_ids = pair
        if isinstance(query_ids, str):
            query_ids = [query_ids]
        elif not isinstance(query_ids, (list, tuple, set)):
            raise ValueError("query_ids must be a string or an iterable of query ids.")
        yield indexed_id, query_ids


def _iter_rank_chunks(
    candidate_pairs: List[Tuple[str, List[str]]], rank: Dict[str, int], chunk_size: int,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    pending_left = pending_right = np.empty(0, dtype=np.int64)
    group_left: List[int] = []
    group_sizes: List[int] = []
    right: List[int] = []

    def flush(final: bool) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        nonlocal pending_left, pending_right
        left_ranks = np.concatenate([pending_left, np.repeat(np.asarray(group_left, dtype=np.int64), group_sizes)])
        right_ranks = np.concatenate([pending_right, np.asarray(right, dtype=np.int64)])
        group_left.clear(), group_sizes.clear(), right.clear()
        full = len(left_ranks) if final else len(left_ranks) - len(left_ranks) % chunk_size
        for start in range(0, full, chunk_size):
            yield left_ranks[start:start + chunk_size], right_ranks[start:start + chunk_size]
        pending_left, pending_right = left_ranks[full:], right_ranks[full:]

    for indexed_id, query_ids in _iter_candidate_groups(candidate_pairs):
        if not query_ids:
            continue
        group_left.append(rank[indexed_id])
        group_sizes.append(len(query_ids))
        right.extend(map(rank.__getitem__, query_ids))
        if len(right) >= chunk_size:
            yield from flush(final=False)
    yield from flush(final=True)


class _CosineScorer:
    """Cosine similarity over ranked ids, reading rows of the model's own vector table."""

    _NORM_BLOCK = 65536

    def __init__(self, kv, ids: List[str]):
        missing_ids = [node_id for node_id in ids if node_id not in kv.key_to_index]
        if missing_ids:
            raise ValueError(f"Missing embeddings for ids: {missing_ids[:5]}")

        table = getattr(kv, "vectors", None)
        if isinstance(table, np.ndarray):
            self.table = table
            self.rows = np.asarray([kv.key_to_index[node_id] for node_id in ids], dtype=np.int64)
        else:
            self.table = np.asarray([kv[node_id] for node_id in ids], dtype=np.float32)
            self.rows = np.arange(len(ids), dtype=np.int64)

        # Norms block by block, so no full normalized copy of the table is ever held.
        norms = np.empty(len(ids), dtype=np.float32)
        for start in range(0, len(ids), self._NORM_BLOCK):
            block = self.table[self.rows[start:start + self._NORM_BLOCK]].astype(np.float32, copy=False)
            norms[start:start + len(block)] = np.linalg.norm(block, axis=1)
        if np.any(norms <= 0.0):
            raise ValueError("Cannot compute cosine similarity for zero vectors.")
        self.inverse_norms = 1.0 / norms

    def __call__(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        dots = np.einsum(
            "ij,ij->i",
            self.table[self.rows[left]].astype(np.float32, copy=False),
            self.table[self.rows[right]].astype(np.float32, copy=False),
        )
        return dots * self.inverse_norms[left] * self.inverse_norms[right]


class _TopKState:
    """Running top-k partners per node; partners rank by (-score, partner id)."""

    def __init__(self, node_count: int, top_k: int):
        self.top_k = top_k
        self.scores = np.full((node_count, top_k), -np.inf, dtype=np.float32)
        self.partners = np.full((node_count, top_k), -1, dtype=np.int64)

    def update(self, nodes: np.ndarray, partners: np.ndarray, scores: np.ndarray) -> None:
        if self.top_k == 1:
            self._update_best(nodes, partners, scores)
            return
        touched = np.unique(nodes)
        kept = self.partners[touched] >= 0
        nodes = np.concatenate([np.repeat(touched, self.top_k)[kept.ravel()], nodes])
        partners = np.concatenate([self.partners[touched][kept], partners])
        scores = np.concatenate([self.scores[touched][kept], scores])

        order = np.lexsort((partners, -scores, nodes))
        nodes, partners, scores = nodes[order], partners[order], scores[order]
        # Scoring is deterministic, so a pair seen twice has equal scores and sorts adjacent.
        first = np.ones(len(nodes), dtype=bool)
        first[1:] = (nodes[1:] != nodes[:-1]) | (partners[1:] != partners[:-1])
        nodes, partners, scores = nodes[first], partners[first], scores[first]

        starts = np.flatnonzero(np.r_[True, nodes[1:] != nodes[:-1]])
        slots = np.arange(len(nodes)) - np.repeat(starts, np.diff(np.r_[starts, len(nodes)]))
        keep = slots < self.top_k

        self.scores[touched] = -np.inf
        self.partners[touched] = -1
        self.scores[nodes[keep], slots[keep]] = scores[keep]
        self.partners[nodes[keep], slots[keep]] = partners[keep]

    def _update_best(self, nodes: np.ndarray, partners: np.ndarray, scores: np.ndarray) -> None:
        order = np.lexsort((partners, -scores, nodes))
        nodes, partners, scores = nodes[order], partners[order], scores[order]
        first = np.r_[True, nodes[1:] != nodes[:-1]]
        nodes, partners, scores = nodes[first], partners[first], scores[first]

        best_scores, best_partners = self.scores[nodes, 0], self.partners[nodes, 0]
        better = (scores > best_scores) | ((scores == best_scores) & (partners < best_partners))
        self.scores[nodes[better], 0] = scores[better]
        self.partners[nodes[better], 0] = partners[better]


def score_mutual_topk_candidate_pairs(
    model,
    candidate_pairs: List[Tuple[str, List[str]]],
    top_k: int = 1,
    chunk_size: int = 2048,
) -> List[ScoredPair]:
    """
    Compute local mutual top-k pairs with unified semantics:

    - left_id == indexed_id == training-side id
    - right_id == query_id == incremental-side id

    Pairs are scored in chunks of ``chunk_size`` and folded into fixed-size top-k
    state per side, so memory depends on the chunk and on the number of distinct
    ids, not on the number of candidate pairs. Small chunks stay in CPU cache.
    """
    top_k = validate_top_k(top_k)
    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    raw_ids = set()
    for indexed_id, query_ids in _iter_candidate_groups(candidate_pairs):
        if query_ids:
            raw_ids.add(indexed_id)
            raw_ids.update(query_ids)
    if not raw_ids:
        raise ValueError("candidate_pairs is empty; cannot calculate similarity.")

    # Integer ids follow string order, so integer tie-breaking matches string tie-breaking.
    ids = sorted({str(node_id) for node_id in raw_ids})
    position = {node_id: index for index, node_id in enumerate(ids)}
    rank = {node_id: position[str(node_id)] for node_id in raw_ids}
    score = _CosineScorer(_resolve_keyed_vectors(model), ids)
    top_for_indexed = _TopKState(len(ids), top_k)
    top_for_query = _TopKState(len(ids), top_k)

    for left, right in _iter_rank_chunks(candidate_pairs, rank, chunk_size):
        scores = score(left, right)
        top_for_indexed.update(left, right, scores)
        top_for_query.update(right, left, scores)

    indexed, slot = np.nonzero(top_for_indexed.partners >= 0)
    query = top_for_indexed.partners[indexed, slot]
    scores = top_for_indexed.scores[indexed, slot]
    mutual = (top_for_query.partners[query] == indexed[:, None]).any(axis=1)
    indexed, query, scores = indexed[mutual], query[mutual], scores[mutual]
    order = np.lexsort((query, -scores, indexed))
    return [(ids[i], ids[q], float(s)) for i, q, s in zip(indexed[order], query[order], scores[order])]


__all__ = [
    "score_candidate_pairs",
    "get_mutual_top_k",
    "get_similarity_top_k",
    "normalize_scored_pair",
    "validate_top_k",
    "select_mutual_topk_pairs",
    "score_mutual_topk_candidate_pairs",
]

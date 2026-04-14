import numpy as np
from typing import Dict, List, Tuple

from tqdm import tqdm

from models.similarity_graph import SimilarityGraph


ScoredPair = Tuple[str, str, float]


def _select_best_pairs(matching_pairs: List[ScoredPair]) -> Tuple[Dict[str, Tuple[str, float]], Dict[str, Tuple[str, float]]]:
    best_for_indexed: Dict[str, Tuple[str, float]] = {}
    best_for_query: Dict[str, Tuple[str, float]] = {}

    for pair in matching_pairs:
        if not isinstance(pair, tuple) or len(pair) != 3:
            raise ValueError("Each matching_pairs item must be a tuple of (left_id/indexed_id, right_id/query_id, score).")
        indexed_id, query_id, score = pair
        score = float(score)

        current_indexed = best_for_indexed.get(indexed_id)
        if current_indexed is None or score > current_indexed[1]:
            best_for_indexed[indexed_id] = (query_id, score)

        current_query = best_for_query.get(query_id)
        if current_query is None or score > current_query[1]:
            best_for_query[query_id] = (indexed_id, score)

    return best_for_indexed, best_for_query


def select_mutual_top1_pairs(matching_pairs: List[ScoredPair]) -> List[ScoredPair]:
    if not isinstance(matching_pairs, list):
        raise ValueError("matching_pairs must be a list of scored tuples.")
    if not matching_pairs:
        raise ValueError("matching_pairs is empty; cannot perform decision making.")

    best_for_indexed, best_for_query = _select_best_pairs(matching_pairs)
    final_pairs: List[ScoredPair] = []
    for indexed_id, (query_id, score) in best_for_indexed.items():
        reverse_best = best_for_query.get(query_id)
        if reverse_best is None:
            continue
        if reverse_best[0] == indexed_id:
            final_pairs.append((indexed_id, query_id, score))

    return final_pairs


def _resolve_keyed_vectors(model):
    base_model = getattr(model, "model", model)
    if hasattr(base_model, "wv"):
        return base_model.wv
    raise ValueError("embedding_model must expose keyed vectors via `.wv`.")


def _cosine_similarity(vec_a, vec_b) -> float:
    a = np.asarray(vec_a, dtype=np.float32)
    b = np.asarray(vec_b, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0.0:
        raise ValueError("Cannot compute cosine similarity for zero vectors.")
    return float(np.dot(a, b) / denom)


def _rescore_pair(model, pair: ScoredPair) -> ScoredPair:
    indexed_id, query_id, _ = pair
    kv = _resolve_keyed_vectors(model)
    if indexed_id not in kv.key_to_index:
        raise ValueError(f"Missing embedding for indexed/training id: {indexed_id}")
    if query_id not in kv.key_to_index:
        raise ValueError(f"Missing embedding for query/incremental id: {query_id}")
    score = _cosine_similarity(kv[indexed_id], kv[query_id])
    return (str(indexed_id), str(query_id), float(score))


def _normalize_pairs(pairs: List[ScoredPair] | None) -> List[ScoredPair]:
    normalized_pairs: List[ScoredPair] = []
    if not pairs:
        return normalized_pairs

    best_scores: Dict[Tuple[str, str], float] = {}
    for pair in pairs:
        if not isinstance(pair, tuple) or len(pair) != 3:
            raise ValueError("Each pair must be a tuple of (left_id/indexed_id, right_id/query_id, score).")
        indexed_id, query_id, score = str(pair[0]), str(pair[1]), float(pair[2])
        pair_key = (indexed_id, query_id)
        current = best_scores.get(pair_key)
        if current is None or score > current:
            best_scores[pair_key] = score

    for (indexed_id, query_id), score in best_scores.items():
        normalized_pairs.append((indexed_id, query_id, score))
    return normalized_pairs


def _remove_pair(
    pair: ScoredPair | None,
    pair_store: Dict[Tuple[str, str], ScoredPair],
    left_map: Dict[str, ScoredPair],
    right_map: Dict[str, ScoredPair],
) -> None:
    if pair is None:
        return
    indexed_id, query_id, _ = pair
    pair_store.pop((indexed_id, query_id), None)
    left_map.pop(indexed_id, None)
    right_map.pop(query_id, None)


def _add_pair(
    pair: ScoredPair,
    pair_store: Dict[Tuple[str, str], ScoredPair],
    left_map: Dict[str, ScoredPair],
    right_map: Dict[str, ScoredPair],
) -> None:
    indexed_id, query_id, score = str(pair[0]), str(pair[1]), float(pair[2])
    normalized = (indexed_id, query_id, score)
    pair_store[(indexed_id, query_id)] = normalized
    left_map[indexed_id] = normalized
    right_map[query_id] = normalized


def _resolve_conflict_component(model, candidate_pairs: List[ScoredPair]) -> List[ScoredPair]:
    rescored_pairs = [_rescore_pair(model, pair) for pair in candidate_pairs]
    return select_mutual_top1_pairs(rescored_pairs)


def merge_mutual_top1_pairs(
    old_pairs: List[ScoredPair] | None,
    new_pairs: List[ScoredPair],
    model,
) -> List[ScoredPair]:
    resolved_pairs = _normalize_pairs(old_pairs)
    pair_store: Dict[Tuple[str, str], ScoredPair] = {(indexed_id, query_id): (indexed_id, query_id, score) for indexed_id, query_id, score in resolved_pairs}
    left_map: Dict[str, ScoredPair] = {indexed_id: (indexed_id, query_id, score) for indexed_id, query_id, score in resolved_pairs}
    right_map: Dict[str, ScoredPair] = {query_id: (indexed_id, query_id, score) for indexed_id, query_id, score in resolved_pairs}

    for new_pair in _normalize_pairs(new_pairs):
        indexed_id, query_id, new_score = new_pair
        exact_pair = pair_store.get((indexed_id, query_id))
        if exact_pair is not None:
            if new_score >= exact_pair[2]:
                _add_pair(new_pair, pair_store, left_map, right_map)
            else:
                _add_pair(_rescore_pair(model, exact_pair), pair_store, left_map, right_map)
            continue

        left_conflict = left_map.get(indexed_id)
        right_conflict = right_map.get(query_id)
        conflicting_pairs = []
        if left_conflict is not None:
            conflicting_pairs.append(left_conflict)
        if right_conflict is not None and right_conflict != left_conflict:
            conflicting_pairs.append(right_conflict)

        if not conflicting_pairs:
            _add_pair(new_pair, pair_store, left_map, right_map)
            continue

        if len(conflicting_pairs) == 1:
            old_pair = conflicting_pairs[0]
            if new_score > old_pair[2]:
                _remove_pair(old_pair, pair_store, left_map, right_map)
                _add_pair(new_pair, pair_store, left_map, right_map)
            else:
                rescored_old_pair = _rescore_pair(model, old_pair)
                winner = new_pair if new_score > rescored_old_pair[2] else rescored_old_pair
                _remove_pair(old_pair, pair_store, left_map, right_map)
                _add_pair(winner, pair_store, left_map, right_map)
            continue

        if new_score > conflicting_pairs[0][2] and new_score > conflicting_pairs[1][2]:
            for old_pair in conflicting_pairs:
                _remove_pair(old_pair, pair_store, left_map, right_map)
            _add_pair(new_pair, pair_store, left_map, right_map)
            continue

        for old_pair in conflicting_pairs:
            _remove_pair(old_pair, pair_store, left_map, right_map)

        for selected_pair in _resolve_conflict_component(model, conflicting_pairs + [new_pair]):
            _add_pair(selected_pair, pair_store, left_map, right_map)

    return list(pair_store.values())


def build_similarity_graph(final_pairs: List[ScoredPair], output_format: str = "graphml") -> SimilarityGraph:
    graph = SimilarityGraph(most_similar_num=1, output_format=output_format)
    for indexed_id, query_id, score in tqdm(final_pairs, total=len(final_pairs), desc="# build similarit graph..."):
        graph.add_similarity(str(indexed_id), [(str(query_id), float(score))])
    return graph


def decide_matches(
    mutualtop_pairs: List[ScoredPair],
    previous_pairs: List[ScoredPair] | None = None,
    model=None,
    output_format: str = "graphml",
) -> Tuple[List[ScoredPair], SimilarityGraph]:
    if model is None:
        raise ValueError("embedding model is required for decision making.")
    final_pairs = merge_mutual_top1_pairs(previous_pairs, mutualtop_pairs, model)
    return final_pairs, build_similarity_graph(final_pairs, output_format=output_format)


__all__ = ["select_mutual_top1_pairs", "merge_mutual_top1_pairs", "build_similarity_graph", "decide_matches"]

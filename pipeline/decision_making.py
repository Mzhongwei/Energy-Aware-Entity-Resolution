from typing import Dict, List, Tuple

from tqdm import tqdm

from models.similarity_graph import SimilarityGraph
from pipeline.calculating_similarity import (
    normalize_scored_pair,
    score_candidate_pairs,
    select_mutual_topk_pairs,
    validate_top_k,
)


ScoredPair = Tuple[str, str, float]


def _normalize_pairs(pairs: List[ScoredPair] | None) -> List[ScoredPair]:
    normalized_pairs: List[ScoredPair] = []
    if not pairs:
        return normalized_pairs

    best_scores: Dict[Tuple[str, str], float] = {}
    for pair in pairs:
        indexed_id, query_id, score = normalize_scored_pair(pair)
        pair_key = (indexed_id, query_id)
        current = best_scores.get(pair_key)
        if current is None or score > current:
            best_scores[pair_key] = score

    for (indexed_id, query_id), score in best_scores.items():
        normalized_pairs.append((indexed_id, query_id, score))
    return normalized_pairs


def merge_mutual_topk_pairs(
    old_pairs: List[ScoredPair] | None,
    new_pairs: List[ScoredPair],
    model,
    top_k: int = 2,
) -> List[ScoredPair]:
    top_k = validate_top_k(top_k)

    candidates = _normalize_pairs((old_pairs or []) + (new_pairs or []))
    if not candidates:
        return []
    candidate_pairs = [(indexed_id, [query_id]) for indexed_id, query_id, _ in candidates]
    rescored_pairs = score_candidate_pairs(model, candidate_pairs)
    return select_mutual_topk_pairs(rescored_pairs, top_k=top_k)


def build_similarity_graph(
    final_pairs: List[ScoredPair],
    output_format: str = "graphml",
    top_k: int = 2,
) -> SimilarityGraph:
    graph = SimilarityGraph(most_similar_num=validate_top_k(top_k), output_format=output_format)
    for indexed_id, query_id, score in tqdm(final_pairs, total=len(final_pairs), desc="# build similarit graph..."):
        graph.add_similarity(str(indexed_id), [(str(query_id), float(score))])
    return graph


def decide_matches(
    mutual_topk_pairs: List[ScoredPair],
    previous_pairs: List[ScoredPair] | None = None,
    model=None,
    output_format: str = "graphml",
    top_k: int = 2,
) -> Tuple[List[ScoredPair], SimilarityGraph]:
    if model is None:
        raise ValueError("embedding model is required for decision making.")
    final_pairs = merge_mutual_topk_pairs(previous_pairs, mutual_topk_pairs, model, top_k=top_k)
    return final_pairs, build_similarity_graph(final_pairs, output_format=output_format, top_k=top_k)


__all__ = [
    "merge_mutual_topk_pairs",
    "build_similarity_graph",
    "decide_matches",
]

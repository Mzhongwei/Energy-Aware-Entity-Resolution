from typing import Dict, List, Tuple

from models import SimilarityGraph


ScoredPair = Tuple[str, str, float]


def _select_best_pairs(matching_pairs: List[ScoredPair]) -> Tuple[Dict[str, Tuple[str, float]], Dict[str, Tuple[str, float]]]:
    best_for_left: Dict[str, Tuple[str, float]] = {}
    best_for_right: Dict[str, Tuple[str, float]] = {}

    for pair in matching_pairs:
        if not isinstance(pair, tuple) or len(pair) != 3:
            raise ValueError("Each matching_pairs item must be a tuple of (left_id, right_id, score).")
        left_id, right_id, score = pair
        score = float(score)

        current_left = best_for_left.get(left_id)
        if current_left is None or score > current_left[1]:
            best_for_left[left_id] = (right_id, score)

        current_right = best_for_right.get(right_id)
        if current_right is None or score > current_right[1]:
            best_for_right[right_id] = (left_id, score)

    return best_for_left, best_for_right


def select_mutual_top1_pairs(matching_pairs: List[ScoredPair]) -> List[ScoredPair]:
    if not isinstance(matching_pairs, list):
        raise ValueError("matching_pairs must be a list of scored tuples.")
    if not matching_pairs:
        raise ValueError("matching_pairs is empty; cannot perform decision making.")

    best_for_left, best_for_right = _select_best_pairs(matching_pairs)
    final_pairs: List[ScoredPair] = []

    for left_id, (right_id, score) in best_for_left.items():
        reverse_best = best_for_right.get(right_id)
        if reverse_best is None:
            continue
        if reverse_best[0] == left_id:
            final_pairs.append((left_id, right_id, score))

    return final_pairs


def build_similarity_graph(final_pairs: List[ScoredPair], output_format: str = "graphml") -> SimilarityGraph:
    graph = SimilarityGraph(most_similar_num=1, output_format=output_format)
    for left_id, right_id, score in final_pairs:
        graph.add_similarity(str(left_id), [(str(right_id), float(score))])
    return graph


def decide_matches(matching_pairs: List[ScoredPair], output_format: str = "graphml") -> SimilarityGraph:
    final_pairs = select_mutual_top1_pairs(matching_pairs)
    return build_similarity_graph(final_pairs, output_format=output_format)


__all__ = ["select_mutual_top1_pairs", "build_similarity_graph", "decide_matches"]

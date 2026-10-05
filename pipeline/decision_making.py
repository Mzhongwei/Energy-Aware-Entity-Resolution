from typing import Dict, List, Tuple

from pipeline.calculating_similarity import (
    normalize_scored_pair,
    score_mutual_topk_candidate_pairs,
    validate_top_k,
)
from utils.utils import orient_record_pair, record_id_side


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
    chunk_size: int = 2048,
) -> List[ScoredPair]:
    top_k = validate_top_k(top_k)

    candidates = _normalize_pairs((old_pairs or []) + (new_pairs or []))
    if not candidates:
        return []
    candidate_pairs = [(indexed_id, [query_id]) for indexed_id, query_id, _ in candidates]
    return score_mutual_topk_candidate_pairs(model, candidate_pairs, top_k=top_k, chunk_size=chunk_size)


class _MatchGraph:
    """
    The former igraph SimilarityGraph on plain dicts, so its output is unchanged.

    A score-1 pair merges both records' groups; any other pair is an edge between groups
    that keeps its highest weight. Each time a pair arrives, the groups it touched keep
    only their ``top_k`` highest edges (ties keep the earlier-created neighbour, as the
    igraph vertex order did), so the result depends on the order pairs are added in.
    """

    def __init__(self, top_k: int):
        self.top_k = validate_top_k(top_k)
        self.group_of: Dict[str, str] = {}
        self.members: Dict[str, List[str]] = {}
        self.created: Dict[str, int] = {}
        self.adjacent: Dict[str, Dict[str, float]] = {}

    def _group(self, record: str) -> str:
        if record not in self.group_of:
            self.group_of[record] = record
            self.members[record] = [record]
            self.created[record] = len(self.group_of)
            self.adjacent[record] = {}
        return self.group_of[record]

    def _add_edge(self, first: str, second: str, weight: float) -> None:
        current = self.adjacent[first].get(second)
        if current is None or current < weight:
            self.adjacent[first][second] = weight
            self.adjacent[second][first] = weight

    def _merge(self, kept: str, merged: str) -> None:
        if kept == merged:
            return
        for record in self.members.pop(merged):
            self.group_of[record] = kept
            self.members[kept].append(record)
        neighbours = self.adjacent.pop(merged)
        for neighbour in sorted(neighbours, key=self.created.__getitem__):
            if neighbour == merged:
                continue  # a self-loop goes away with its group
            self.adjacent[neighbour].pop(merged, None)
            if neighbour != kept:
                self._add_edge(kept, neighbour, neighbours[neighbour])
        del self.created[merged]

    def _limit(self, group: str) -> None:
        edges = self.adjacent.get(group)
        if edges is None:
            return
        # igraph lists a self-loop twice among a vertex's edges; deleting either copy drops it.
        ordered = [
            (neighbour, weight)
            for neighbour, weight in sorted(edges.items(), key=lambda edge: self.created[edge[0]])
            for _ in range(2 if neighbour == group else 1)
        ]
        if len(ordered) <= self.top_k:
            return
        for neighbour, _ in sorted(ordered, key=lambda edge: edge[1], reverse=True)[self.top_k:]:
            if edges.pop(neighbour, None) is not None and neighbour != group:
                self.adjacent[neighbour].pop(group, None)

    def add(self, record: str, other: str, score: float) -> None:
        group = self._group(record)
        other_group = self._group(other)
        if score == 1.0:
            self._merge(group, other_group)
            group = self.group_of[record]
        else:
            self._add_edge(group, other_group, score)
        self._limit(other_group)
        self._limit(group)

    def scored_pairs(self, config: dict) -> List[ScoredPair]:
        scores: Dict[Tuple[str, str], float] = {}
        for group, edges in self.adjacent.items():
            for neighbour, weight in edges.items():
                if self.created[group] <= self.created[neighbour]:
                    left, right = orient_record_pair(config, group, neighbour)
                    scores[(left, right)] = max(weight, scores.get((left, right), float("-inf")))
        for members in self.members.values():
            lefts = [record for record in members if record_id_side(config, record) == "left"]
            rights = [record for record in members if record_id_side(config, record) == "right"]
            for left in lefts:
                for right in rights:
                    scores[(left, right)] = 1.0
        return sorted((left, right, score) for (left, right), score in scores.items())


def expand_matches(config: dict, pairs: List[ScoredPair], top_k: int) -> List[ScoredPair]:
    """
    Canonical, sorted ``(left, right, score)`` matches, as exported by the former graph.

    Each record keeps at most ``top_k`` matches -- a record indexed in one window and queried
    in another can be in more mutual top-k pairs than that. Records joined by score-1 pairs
    form one group whose left x right members all match with score 1.
    """
    graph = _MatchGraph(top_k)
    for pair in pairs:
        graph.add(*normalize_scored_pair(pair))
    return graph.scored_pairs(config)


def decide_matches(
    mutual_topk_pairs: List[ScoredPair],
    previous_pairs: List[ScoredPair] | None = None,
    model=None,
    top_k: int = 2,
    min_similarity: float | None = None,
    chunk_size: int = 2048,
) -> List[ScoredPair]:
    """Merge this window's mutual top-k pairs into the previous decision, rescored by ``model``."""
    if model is None:
        raise ValueError("embedding model is required for decision making.")
    final_pairs = merge_mutual_topk_pairs(
        previous_pairs, mutual_topk_pairs, model, top_k=top_k, chunk_size=chunk_size,
    )
    if min_similarity is not None:
        threshold = float(min_similarity)
        if not -1.0 <= threshold <= 1.0:
            raise ValueError("decision_making.min_similarity must be between -1 and 1.")
        final_pairs = [pair for pair in final_pairs if pair[2] >= threshold]
    return final_pairs


__all__ = [
    "merge_mutual_topk_pairs",
    "expand_matches",
    "decide_matches",
]

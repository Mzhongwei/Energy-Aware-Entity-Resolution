"""BERT matching over the embedding decision's pairs.

The embedding pipeline acts as blocking: its mutual top-k decision proposes pairs, and this
stage keeps only the pairs a trained BERT cross-encoder classifies as matches.

Input:  scored pairs ``(left_id, right_id, embedding_score)`` from decision making.
Output: the subset BERT accepts, with the embedding score unchanged. The score stays the
        embedding similarity, not the BERT probability: matches with a score of exactly 1.0
        are grouped (see ``decision_making.expand_matches``), which a softmax probability
        can round to.
"""
from __future__ import annotations

import csv
import time
from io import StringIO
from typing import Dict, Iterable, List, Optional, Tuple

ScoredPair = Tuple[str, str, float]
# (left_id, right_id, embedding score, BERT probability)
JudgedPair = Tuple[str, str, float, float]
JUDGED_PAIRS_HEADER = ("left_id", "right_id", "score", "bert_probability")

DEFAULTS = {"threshold": 0.5, "batch_size": 64, "max_length": 128, "device": "auto"}


def bert_matching_config(config: dict) -> dict:
    section = config.get("bert_matching") if isinstance(config, dict) else None
    merged = dict(DEFAULTS, **(section if isinstance(section, dict) else {}))
    threshold = float(merged["threshold"])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("bert_matching.threshold must be between 0 and 1")
    if int(merged["batch_size"]) < 1 or int(merged["max_length"]) < 1:
        raise ValueError("bert_matching.batch_size and max_length must be positive")
    match_capacity(merged)
    return merged


def match_capacity(settings: dict) -> Tuple[Optional[int], Optional[int]]:
    """How many matches a left (A) and a right (B) record may keep; None is unlimited.

    (1, 1): clean-clean (one-to-one); (1, None): each A matches at most one B, a B may match
    several A (dirty A against a clean B, as retrieval-extrait); (None, None): every accepted
    pair is kept (dirty ER)."""
    capacity = []
    for key in ("max_matches_per_left", "max_matches_per_right"):
        value = settings.get(key)
        if value is not None and (isinstance(value, bool) or int(value) < 1):
            raise ValueError(f"bert_matching.{key} must be a positive integer or null")
        capacity.append(None if value is None else int(value))
    return capacity[0], capacity[1]


def select_matches(judged: Iterable[JudgedPair], max_left: Optional[int], max_right: Optional[int]) -> List[JudgedPair]:
    """Greedy capacity-constrained matching: pairs in decreasing BERT probability (then
    embedding score, then ids, for determinism), each kept while both of its records have
    capacity left. Without capacities every pair is kept."""
    pairs = sorted(judged, key=lambda pair: (-pair[3], -pair[2], pair[0], pair[1]))
    if max_left is None and max_right is None:
        return pairs
    left_used: Dict[str, int] = {}
    right_used: Dict[str, int] = {}
    selected = []
    for pair in pairs:
        left, right = pair[0], pair[1]
        if max_left is not None and left_used.get(left, 0) >= max_left:
            continue
        if max_right is not None and right_used.get(right, 0) >= max_right:
            continue
        left_used[left] = left_used.get(left, 0) + 1
        right_used[right] = right_used.get(right, 0) + 1
        selected.append(pair)
    return selected


def serialize_judged_pairs(pairs: Iterable[JudgedPair]) -> bytes:
    buffer = StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(JUDGED_PAIRS_HEADER)
    writer.writerows((left, right, repr(float(score)), repr(float(probability)))
                     for left, right, score, probability in pairs)
    return buffer.getvalue().encode("utf-8")


def load_judged_pairs(path: str) -> List[JudgedPair]:
    with open(path, newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        header = tuple(next(reader, ()))
        if header != JUDGED_PAIRS_HEADER:
            raise ValueError(f"{path} does not start with the header {','.join(JUDGED_PAIRS_HEADER)}")
        return [(left, right, float(score), float(probability)) for left, right, score, probability in reader]


class BertMatcher:
    """Judges each distinct pair once; later windows reuse the cached probability."""

    def __init__(self, config: dict, service=None, records=None):
        settings = bert_matching_config(config)
        self.threshold = float(settings["threshold"])
        self.batch_size = int(settings["batch_size"])
        if service is None:
            from pipeline.bert_inference import InferenceService
            from utils.pipeline_io import get_model_directory

            service = InferenceService(
                save_dir=get_model_directory(config, "bert"),
                max_length=int(settings["max_length"]),
                device=settings["device"],
            )
        if records is None:
            from pipeline.record_store import record_store_for

            records = record_store_for(config)
        self.service = service
        self.records = records
        self.probabilities: Dict[Tuple[str, str], float] = {}

    def filter_pairs(self, pairs: List[ScoredPair], remember: bool = True) -> List[ScoredPair]:
        """Keep the pairs BERT classifies as matches. ``remember`` caches each judgement for
        pairs that come back in later windows (the cumulative decision); pairs judged once
        and never again (window candidates) need no cache."""
        return [(left, right, score) for left, right, score, _ in self.judge(pairs, remember)]

    def judge(self, pairs: List[ScoredPair], remember: bool = True) -> List[JudgedPair]:
        """Like ``filter_pairs``, with each accepted pair's BERT probability."""
        started = time.perf_counter()
        probabilities = self.probabilities if remember else {}
        unseen = list(dict.fromkeys((str(left), str(right)) for left, right, _ in pairs
                                    if (str(left), str(right)) not in probabilities))
        if unseen:
            texts = self.records.get_many({rid for pair in unseen for rid in pair})
            judged = self.service.predict_batch(
                [texts[left] for left, _ in unseen],
                [texts[right] for _, right in unseen],
                batch_size=self.batch_size,
            )
            probabilities.update(zip(unseen, judged))
        accepted = [(str(left), str(right), float(score), float(probabilities[(str(left), str(right))]))
                    for left, right, score in pairs
                    if probabilities[(str(left), str(right))] >= self.threshold]
        print(f"[bert_matching] pairs={len(pairs)} judged={len(unseen)} accepted={len(accepted)} "
              f"seconds={time.perf_counter() - started:.6f}", flush=True)
        return accepted


__all__ = [
    "BertMatcher", "bert_matching_config", "load_judged_pairs", "match_capacity", "select_matches",
    "serialize_judged_pairs",
]

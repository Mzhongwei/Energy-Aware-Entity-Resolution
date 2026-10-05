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

import time
from typing import Dict, List, Tuple

ScoredPair = Tuple[str, str, float]

DEFAULTS = {"threshold": 0.5, "batch_size": 64, "max_length": 128, "device": "auto"}


def bert_matching_config(config: dict) -> dict:
    section = config.get("bert_matching") if isinstance(config, dict) else None
    merged = dict(DEFAULTS, **(section if isinstance(section, dict) else {}))
    threshold = float(merged["threshold"])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("bert_matching.threshold must be between 0 and 1")
    if int(merged["batch_size"]) < 1 or int(merged["max_length"]) < 1:
        raise ValueError("bert_matching.batch_size and max_length must be positive")
    return merged


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

    def filter_pairs(self, pairs: List[ScoredPair]) -> List[ScoredPair]:
        started = time.perf_counter()
        unseen = list(dict.fromkeys((str(left), str(right)) for left, right, _ in pairs
                                    if (str(left), str(right)) not in self.probabilities))
        if unseen:
            texts = self.records.get_many({rid for pair in unseen for rid in pair})
            probabilities = self.service.predict_batch(
                [texts[left] for left, _ in unseen],
                [texts[right] for _, right in unseen],
                batch_size=self.batch_size,
            )
            self.probabilities.update(zip(unseen, probabilities))
        accepted = [(str(left), str(right), float(score)) for left, right, score in pairs
                    if self.probabilities[(str(left), str(right))] >= self.threshold]
        print(f"[bert_matching] pairs={len(pairs)} judged={len(unseen)} accepted={len(accepted)} "
              f"seconds={time.perf_counter() - started:.6f}", flush=True)
        return accepted


__all__ = ["BertMatcher", "bert_matching_config"]

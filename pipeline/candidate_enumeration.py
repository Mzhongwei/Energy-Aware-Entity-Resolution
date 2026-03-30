from __future__ import annotations

from typing import Any

from models import CGIndex

def query_candidates(feature: Any, right_index: CGIndex) -> List[str]:
    if right_index is None:
        raise ValueError("right_index is required for candidate querying.")
    return right_index.query(feature)

def enumerate_candidates(cg_feature, index):
    """
    Docstring for enumerate_candidates
    
    :param cg_feature: Description
    :param index: Description

    :return candidate_pairs: list of tuple, tuple: (rid, [candidate_id1, candidate_id2, ...])
    """
    candidate_pairs = []
    for record_id, feature in cg_feature:
        candidate_ids = [cid for cid in query_candidates(feature, index) if cid != str(record_id)]
        candidate_pairs.append((str(record_id), candidate_ids))
    return candidate_pairs


__all__ = [
    "enumerate_candidates",
]

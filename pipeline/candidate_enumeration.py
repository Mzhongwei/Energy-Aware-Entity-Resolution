from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

from models import CGIndex

def query_indexed_ids(feature: Any, indexed_record_index: CGIndex) -> List[str]:
    if indexed_record_index is None:
        raise ValueError("indexed_record_index is required for candidate querying.")
    return indexed_record_index.query(feature)

def enumerate_candidates(cg_feature, index):
    """
    Build candidate pairs with a unified semantic direction:

    - left_id: indexed/training-side id
    - right_id: query/incremental-side id

    :return candidate_pairs: list of tuple, tuple: (indexed_id, [query_id1, query_id2, ...])
    """
    indexed_to_query_ids: Dict[str, List[str]] = defaultdict(list)
    for query_id, feature in cg_feature:
        query_id = str(query_id)
        indexed_ids = [indexed_id for indexed_id in query_indexed_ids(feature, index) if indexed_id != query_id]
        for indexed_id in indexed_ids:
            indexed_to_query_ids[str(indexed_id)].append(query_id)

    candidate_pairs = []
    for indexed_id, query_ids in indexed_to_query_ids.items():
        candidate_pairs.append((str(indexed_id), query_ids))
    return candidate_pairs

def fetch_candidates(data_pairs_file):
    """
    Get candidate pairs from a txt file with unified semantic direction:

    left_id,indexed_id
    right_id,query_id

    example:
    idx__1, idx__2
    idx__3, idx__4
    ...

    : return: list of tuple
    """
    candidate_pairs = []
    with open(data_pairs_file, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            left, right = [part.strip() for part in line.split(",", 1)]
            candidate_pairs.append((str(left), str(right)))
    if not candidate_pairs:
        raise IOError("Matches file is empty.")
    return candidate_pairs

__all__ = [
    "enumerate_candidates", "fetch_candidates", "query_indexed_ids"
]

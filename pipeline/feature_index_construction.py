from __future__ import annotations

import tempfile
from typing import Any, Dict, Iterable, List, Optional, Tuple

from models import CGIndex

from .cg_feature_extraction import _as_field_list, _cg_block


def build_index(features: Iterable[Tuple[int, Any]], right_index: CGIndex) -> CGIndex:
    if right_index is None:
        raise ValueError("right_index is required.")
    batch = list(features)
    right_index.upsert(batch)
    if right_index.method == "fullindexing":
        staged = getattr(right_index, "_cg_fullindexing_staged_ids", [])
        seen = set(staged)
        for rid, _ in batch:
            rid_int = int(rid)
            if rid_int not in seen:
                staged.append(rid_int)
                seen.add(rid_int)
        setattr(right_index, "_cg_fullindexing_staged_ids", staged)
    return right_index


def _commit_fullindexing_ids(right_index: CGIndex) -> None:
    if right_index is None or right_index.method != "fullindexing":
        return
    staged = list(getattr(right_index, "_cg_fullindexing_staged_ids", []))
    if not staged:
        return
    committed = list(getattr(right_index, "_cg_fullindexing_committed_ids", []))
    seen = set(committed)
    for rid in staged:
        if rid not in seen:
            committed.append(int(rid))
            seen.add(rid)
    setattr(right_index, "_cg_fullindexing_committed_ids", committed)
    setattr(right_index, "_cg_fullindexing_staged_ids", [])


def commit_cg_index(right_index: CGIndex) -> CGIndex:
    if right_index is None:
        raise ValueError("right_index is required.")
    right_index.commit()
    _commit_fullindexing_ids(right_index)
    return right_index


def _blocking_spec_for_method(method: str, config: Optional[Dict[str, Any]]) -> List[str]:
    cg_cfg = _cg_block(config)
    if method == "key-blocking":
        return _as_field_list(cg_cfg.get("key_blocking", {}).get("keys"), "candidate_generation.key_blocking.keys")
    if method == "token-blocking":
        return _as_field_list(cg_cfg.get("token_blocking", {}).get("field"), "candidate_generation.token_blocking.field")
    if method == "minhash-lsh":
        return _as_field_list(cg_cfg.get("minhash_lsh", {}).get("field"), "candidate_generation.minhash_lsh.field")
    return []


def _create_cg_index(method: str, config: Optional[Dict[str, Any]]) -> CGIndex:
    cg_cfg = _cg_block(config)
    index_dir = cg_cfg.get("index_dir") or tempfile.mkdtemp(prefix="cg_index_")
    build_id = str(cg_cfg.get("build_id", f"cg_{method}"))
    dataset_fp = str(cg_cfg.get("dataset_fp", ""))
    blocking_spec = _blocking_spec_for_method(method, config)
    return CGIndex(
        build_id=build_id,
        dataset_fp=dataset_fp,
        method=method,
        blocking_spec=blocking_spec,
        index_dir=index_dir,
    )


def create_cg_index(method: str, config: Optional[Dict[str, Any]]) -> CGIndex:
    return _create_cg_index(method, config)


__all__ = [
    "build_index",
    "commit_cg_index",
    "create_cg_index",
]

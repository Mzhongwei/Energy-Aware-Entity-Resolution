from __future__ import annotations

import tempfile
from typing import Any, Dict, Iterable, List, Optional, Tuple

from models.cg_index import CGIndex

from .cg_feature_extraction import _as_field_list, _cg_block


def build_index(features: Iterable[Tuple[str, Any]], right_index: CGIndex) -> CGIndex:
    if right_index is None:
        raise ValueError("right_index is required.")
    right_index.build(list(features))
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


def create_cg_index(
    method: str,
    config: Optional[Dict[str, Any]],
    index_dir: Optional[str] = None,
) -> CGIndex:
    cg_cfg = _cg_block(config)
    resolved_index_dir = index_dir or cg_cfg.get("index_dir") or tempfile.mkdtemp(prefix="cg_index_")
    build_id = str(cg_cfg.get("build_id", f"cg_{method}"))
    dataset_fp = str(cg_cfg.get("dataset_fp", ""))
    blocking_spec = _blocking_spec_for_method(method, config)
    return CGIndex.from_config(
        config or {},
        index_dir=resolved_index_dir,
        build_id=build_id,
        dataset_fp=dataset_fp,
        blocking_spec=blocking_spec,
    )


__all__ = [
    "build_index",
    "create_cg_index",
]

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from models import CGIndex

from .cg_feature_extraction import _ensure_rid, _rid_to_int, compute_features
from .feature_index_construction import build_index, commit_cg_index, create_cg_index


def _empty_candidates_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["left_id", "right_id"])


def _enforce_output_contract(candidates_df: pd.DataFrame, top_k: Optional[int]) -> pd.DataFrame:
    if candidates_df is None or candidates_df.empty:
        return _empty_candidates_df()

    required = {"left_id", "right_id"}
    missing = required.difference(candidates_df.columns)
    if missing:
        raise ValueError(f"candidates_df missing required columns: {sorted(missing)}")

    work = candidates_df.copy()
    work["left_id"] = work["left_id"].map(_rid_to_int)
    work["right_id"] = work["right_id"].map(_rid_to_int)

    sort_cols = ["left_id"]
    ascending = [True]
    if "score" in work.columns:
        work["score"] = pd.to_numeric(work["score"], errors="coerce")
        sort_cols += ["score", "right_id"]
        ascending += [False, True]
    else:
        sort_cols += ["right_id"]
        ascending += [True]

    work = work.sort_values(sort_cols, ascending=ascending, kind="mergesort")
    work = work.drop_duplicates(subset=["left_id", "right_id"], keep="first")

    if top_k is not None:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer.")
        work = work.groupby("left_id", sort=False, group_keys=False).head(top_k)

    cols = ["left_id", "right_id"] + (["score"] if "score" in work.columns else [])
    return work.loc[:, cols].reset_index(drop=True)


def generate_candidates_from_index(
    left_features: Iterable[Tuple[int, Any]],
    right_index: CGIndex,
    top_k: Optional[int] = None,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    if right_index is None:
        raise ValueError("right_index is required for candidate querying.")

    rows: List[Dict[str, int]] = []
    for left_id, left_feat in left_features:
        if right_index.method == "fullindexing" and hasattr(right_index, "_cg_fullindexing_committed_ids"):
            right_ids = set(int(r) for r in getattr(right_index, "_cg_fullindexing_committed_ids", []))
        else:
            right_ids = right_index.query(left_feat)
        if not right_ids:
            continue
        for right_id in sorted(int(r) for r in right_ids):
            rows.append({"left_id": int(left_id), "right_id": right_id})

    return _enforce_output_contract(pd.DataFrame(rows), top_k=top_k)


def _resolved_top_k(config: Optional[Dict[str, Any]]) -> Optional[int]:
    cg_cfg = config.get("candidate_generation", {}) if isinstance(config, dict) else {}
    top_k = cg_cfg.get("top_k")
    if top_k is None:
        return None
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("candidate_generation.top_k must be a positive integer.")
    return top_k


def _resolved_seed(method: str, config: Optional[Dict[str, Any]]) -> Optional[int]:
    cg_cfg = config.get("candidate_generation", {}) if isinstance(config, dict) else {}
    if method == "minhash-lsh":
        mh_cfg = cg_cfg.get("minhash_lsh", {}) if isinstance(cg_cfg.get("minhash_lsh", {}), dict) else {}
        if mh_cfg.get("seed") is not None:
            return int(mh_cfg["seed"])
    if cg_cfg.get("random_seed") is not None:
        return int(cg_cfg["random_seed"])
    return None


def generate_candidates(
    *,
    mode: str,
    method: Optional[str] = None,
    left_df: Optional[pd.DataFrame] = None,
    right_df: Optional[pd.DataFrame] = None,
    right_index: Optional[CGIndex] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, CGIndex]:
    mode_normalized = str(mode).strip().lower()
    if mode_normalized not in {"batch", "incremental"}:
        raise ValueError("mode must be 'batch' or 'incremental'.")

    method = config["candidate_generation"]["method"]
    top_k = _resolved_top_k(config)
    seed = _resolved_seed(method, config)

    if left_df is None:
        raise ValueError("left_df is required.")
    if not isinstance(left_df, pd.DataFrame):
        raise ValueError("left_df must be a pandas.DataFrame.")
    _ensure_rid(left_df)

    if mode_normalized == "batch":
        if right_df is None:
            raise ValueError("right_df is required in batch mode.")
        if not isinstance(right_df, pd.DataFrame):
            raise ValueError("right_df must be a pandas.DataFrame.")
        _ensure_rid(right_df)

        index = create_cg_index(method, config)
        right_features = compute_features(right_df, method, config)
        build_index(right_features, index)
        commit_cg_index(index)

        left_features = compute_features(left_df, method, config)
        candidates_df = generate_candidates_from_index(left_features, index, top_k=top_k, seed=seed)
        return candidates_df, index

    index = right_index
    if index is None:
        if right_df is None:
            raise ValueError("Incremental mode requires an existing right_index or right_df to build one.")
        if not isinstance(right_df, pd.DataFrame):
            raise ValueError("right_df must be a pandas.DataFrame when provided.")
        _ensure_rid(right_df)
        index = create_cg_index(method, config)
    else:
        if not isinstance(index, CGIndex):
            raise ValueError("right_index must be a CGIndex instance.")
        if index.method != method:
            raise ValueError(f"right_index.method ({index.method}) does not match requested method ({method}).")

    if right_df is not None:
        if not isinstance(right_df, pd.DataFrame):
            raise ValueError("right_df must be a pandas.DataFrame.")
        _ensure_rid(right_df)
        right_features = compute_features(right_df, method, config)
        build_index(right_features, index)
        commit_cg_index(index)

    left_features = compute_features(left_df, method, config)
    candidates_df = generate_candidates_from_index(left_features, index, top_k=top_k, seed=seed)
    return candidates_df, index


__all__ = [
    "generate_candidates_from_index",
    "generate_candidates",
]

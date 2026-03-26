import numpy as np
from typing import Tuple

from .calculating_similarity import IdxMatrix


def ratio_rnn_edges(
    D: np.ndarray,
    I: np.ndarray,
    ratio: float = None,
    delta: float = 0.1,
    enforce_rnn: bool = True,
    single_threshold: float = 0.95,
    max_degree: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    N, K = D.shape
    if K < 1:
        return np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.float32)

    all_neighbors = I.flatten()
    degrees = np.bincount(all_neighbors, minlength=N)
    bad_nodes = set(np.where(degrees > max_degree)[0])

    s1 = D[:, 0]
    j1 = I[:, 0]

    force_mask = (s1 >= single_threshold) & (~np.isin(j1, list(bad_nodes))) & (~np.isin(np.arange(N), list(bad_nodes)))
    force_rows = np.where(force_mask)[0]
    force_cols = j1[force_mask]
    force_scores = s1[force_mask]

    if force_mask.all():
        return force_rows.astype(np.int32), force_cols.astype(np.int32), force_scores.astype(np.float32)

    remain_mask = ~force_mask
    if K < 2:
        rows = np.empty((0,), dtype=np.int32)
        cols = np.empty((0,), dtype=np.int32)
        scores = np.empty((0,), dtype=np.float32)
    else:
        s1_remain = s1[remain_mask]
        s2_remain = D[remain_mask, 1]
        j1_remain = j1[remain_mask]

        mask = np.ones_like(s1_remain, dtype=bool)
        if ratio is not None:
            mask &= (s1_remain / (s2_remain + 1e-12)) >= ratio
        if delta is not None:
            mask &= (s1_remain - s2_remain) >= float(delta)

        rows = np.where(remain_mask)[0][mask]
        cols = j1_remain[mask]
        scores = s1_remain[mask]

    rows = np.concatenate([force_rows, rows])
    cols = np.concatenate([force_cols, cols])
    scores = np.concatenate([force_scores, scores])

    if enforce_rnn and rows.size > 0:
        best_row_for_col = np.full(N, -1, dtype=np.int32)
        best_score_for_col = np.full(N, -np.inf, dtype=np.float32)
        for r in range(N):
            for k in range(min(K, I.shape[1])):
                c = I[r, k]
                sc = D[r, k]
                if sc > best_score_for_col[c]:
                    best_score_for_col[c] = sc
                    best_row_for_col[c] = r
        keep = best_row_for_col[cols] == rows
        rows = rows[keep]
        cols = cols[keep]
        scores = scores[keep]

    return rows.astype(np.int32), cols.astype(np.int32), scores.astype(np.float32)


def r1nn_only(I: np.ndarray, D: np.ndarray, allow_self: bool = False):
    N, K = D.shape
    if K < 1:
        return np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.float32)

    rows = np.arange(N, dtype=np.int32)
    j1 = I[:, 0].astype(np.int32)

    valid = (j1 >= 0) & (j1 < N)
    if not np.all(valid):
        rows_v = rows[valid]
        j1_v = j1[valid]
        partner = I[j1_v, 0]
        mask = np.zeros(N, dtype=bool)
        mask[valid] = partner == rows_v
        if not allow_self:
            mask[valid] &= j1_v != rows_v
    else:
        partner = I[j1, 0]
        mask = partner == rows
        if not allow_self:
            mask &= j1 != rows

    out_rows = rows[mask].astype(np.int32)
    out_cols = j1[mask].astype(np.int32)
    out_scores = D[mask, 0].astype(np.float32)
    return out_rows, out_cols, out_scores


def pipeline_ratio_rnn_triangle_one2one(
    E: np.ndarray,
    ratio: float = 1.2,
    delta: float = None,
    enforce_rnn: bool = True,
    single_threshold: float = 0.95,
    triangle_alpha: float = 0.9,
    triangle_undirected: bool = True,
    triangle_max_deg: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    D, I = IdxMatrix.topk_all_cosine(E, k=10, budget_mb=512)
    rows, cols, scores = ratio_rnn_edges(
        D,
        I,
        ratio=ratio,
        delta=delta,
        enforce_rnn=enforce_rnn,
        single_threshold=single_threshold,
    )
    return rows, cols, scores


__all__ = [
    "ratio_rnn_edges",
    "r1nn_only",
    "pipeline_ratio_rnn_triangle_one2one",
]

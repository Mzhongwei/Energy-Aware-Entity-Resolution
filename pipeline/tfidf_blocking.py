"""TF-IDF candidate generation: an alternative to the embedding similarity for the running-best rule.

Records are bags of lowercase alphanumeric tokens (``record_terms``, written by normalization
next to each record's BERT text). The index holds the indexed (left, A) records: its vocabulary
and inverse document frequencies come from them alone, since a query token that no indexed
record has cannot make a pair similar. Tokens in more than ``max_df`` of the indexed records
(stop words, attribute boilerplate) are dropped. Vectors are binary term weights times IDF,
L2-normalized, so a pair's score is the cosine of its TF-IDF vectors; pairs sharing no kept
token score 0 and are never candidates.

``window_top_k`` returns the same top-k states as the embedding path
(``pipeline.calculating_similarity._window_top_k``), so the running-best selection is shared.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

TOKEN = re.compile(r"[a-z0-9]+")
DEFAULT_MAX_DF = 0.1
# Query rows scored per sparse product; each holds a dense rows x indexed-records float32 block.
DEFAULT_CHUNK_ROWS = 512


def record_terms(values: Iterable[object]) -> str:
    """Space-separated tokens of a record's attribute values (names excluded), in order."""
    tokens = []
    for value in values:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        tokens.extend(TOKEN.findall(str(value).lower()))
    return " ".join(tokens)


def tfidf_config(config: dict) -> dict:
    """``calculating_similarity.tfidf``: fields (record columns used, default all), max_df,
    chunk_rows."""
    section = ((config.get("calculating_similarity") or {}).get("tfidf") or {}) if isinstance(config, dict) else {}
    max_df = float(section.get("max_df", DEFAULT_MAX_DF))
    if not 0.0 < max_df <= 1.0:
        raise ValueError("calculating_similarity.tfidf.max_df must be in (0, 1]")
    chunk_rows = int(section.get("chunk_rows", DEFAULT_CHUNK_ROWS))
    if chunk_rows < 1:
        raise ValueError("calculating_similarity.tfidf.chunk_rows must be positive")
    fields = section.get("fields")
    if fields is not None:
        fields = [str(field).strip().lower() for field in fields]
    return {"fields": fields, "max_df": max_df, "chunk_rows": chunk_rows}


class TfidfIndex:
    """TF-IDF vectors of the indexed records; queries are vectorized with the same vocabulary."""

    def __init__(self, ids: Sequence[str], terms: Sequence[str], max_df: float = DEFAULT_MAX_DF):
        if len(ids) != len(terms):
            raise ValueError("ids and terms must have the same length")
        if not ids:
            raise ValueError("TfidfIndex needs at least one indexed record")
        self.ids = [str(node_id) for node_id in ids]
        token_sets = [set(text.split()) for text in terms]
        df = Counter()
        for tokens in token_sets:
            df.update(tokens)
        limit = max(1, int(max_df * len(self.ids)))
        kept = sorted(token for token, count in df.items() if count <= limit)
        self.vocabulary = {token: column for column, token in enumerate(kept)}
        n = len(self.ids)
        # Smoothed IDF: every kept token has a positive weight.
        self.idf = np.asarray([math.log((1 + n) / (1 + df[token])) + 1.0 for token in kept], dtype=np.float32)
        self.matrix_t = self._vectorize(token_sets).T.tocsr()  # vocabulary x indexed records

    def _vectorize(self, token_sets: Sequence[set]) -> sp.csr_matrix:
        rows, columns = [], []
        for row, tokens in enumerate(token_sets):
            for token in tokens:
                column = self.vocabulary.get(token)
                if column is not None:
                    rows.append(row)
                    columns.append(column)
        columns = np.asarray(columns, dtype=np.int64)
        matrix = sp.csr_matrix(
            (self.idf[columns], (np.asarray(rows, dtype=np.int64), columns)),
            shape=(len(token_sets), len(self.vocabulary)), dtype=np.float32,
        )
        norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
        norms[norms == 0] = 1.0
        return sp.csr_matrix(sp.diags(1.0 / norms, dtype=np.float32) @ matrix)

    def vectorize(self, terms: Sequence[str]) -> sp.csr_matrix:
        return self._vectorize([set(text.split()) for text in terms])

    @classmethod
    def from_records(cls, records: Mapping[str, str], max_df: float = DEFAULT_MAX_DF) -> "TfidfIndex":
        ids = sorted(records)
        return cls(ids, [records[node_id] for node_id in ids], max_df=max_df)


def _top_per_axis(scores: np.ndarray, depth: int, axis: int) -> Tuple[np.ndarray, np.ndarray]:
    """(row, column) positions of the positive scores at least the ``depth``-th largest along
    ``axis``. Ties with the ``depth``-th score are all kept: the top-k state breaks them by id."""
    from pipeline.calculating_similarity import _kth_largest

    kth = _kth_largest(scores, depth, axis)
    keep = scores > 0
    if kth is not None:
        keep &= scores >= np.expand_dims(kth, axis)
    return np.nonzero(keep)


def window_top_k(index: TfidfIndex, query_ids: Sequence[str], query_terms: Sequence[str], top_k: int,
                 chunk_rows: int = DEFAULT_CHUNK_ROWS):
    """Exact top-k partners of the indexed records and of one window's query records.

    Returns ``(ids, indexed_ranks, top_for_indexed, top_for_query)`` as
    ``pipeline.calculating_similarity._window_top_k``: ids sorted, ranks into them, and the
    top-k states of each side (partners rank by -score, then partner id). Only pairs with a
    positive score enter the states.
    """
    from pipeline.calculating_similarity import _TopKState, validate_top_k

    top_k = validate_top_k(top_k)
    if len(query_ids) != len(query_terms):
        raise ValueError("query_ids and query_terms must have the same length")
    query_ids = [str(node_id) for node_id in query_ids]
    ids = sorted(set(index.ids) | set(query_ids))
    position = {node_id: rank for rank, node_id in enumerate(ids)}
    indexed = np.fromiter((position[node_id] for node_id in index.ids), dtype=np.int64, count=len(index.ids))
    queries = np.fromiter((position[node_id] for node_id in query_ids), dtype=np.int64, count=len(query_ids))
    top_for_indexed = _TopKState(len(ids), top_k)
    top_for_query = _TopKState(len(ids), top_k)

    vectors = index.vectorize(query_terms)
    for start in range(0, len(query_ids), int(chunk_rows)):
        scores = (vectors[start:start + int(chunk_rows)] @ index.matrix_t).toarray()
        # A pair in the final top-k of a query (row) or of an indexed record (column) is in
        # this chunk's top-k of that row or column.
        row_r, row_c = _top_per_axis(scores, top_k, axis=1)
        col_r, col_c = _top_per_axis(scores, top_k, axis=0)
        rows = np.concatenate([row_r, col_r])
        columns = np.concatenate([row_c, col_c])
        if len(rows) == 0:
            continue
        pair_scores = scores[rows, columns].astype(np.float32)
        left, right = indexed[columns], queries[start + rows]
        top_for_indexed.update(left, right, pair_scores)
        top_for_query.update(right, left, pair_scores)
        del scores
    return ids, np.sort(indexed), top_for_indexed, top_for_query


__all__ = ["TfidfIndex", "record_terms", "tfidf_config", "window_top_k"]

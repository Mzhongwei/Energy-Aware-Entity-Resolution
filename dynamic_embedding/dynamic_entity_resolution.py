# -*- coding: utf-8 -*-
import pandas as pd
import faiss
import numpy as np
from typing import List, Tuple, Optional, Dict

from utils.utils import parse_idx_suffix


class FaissIndex:
    """
    Use FAISS to build an index for words that start with 'idx__' and perform Top-K (cosine similarity).
    Similarity = inner product (all vectors have been L2 normalized).

    usage:
    fx = FaissIndex(model, prefix="idx__")
    sims = fx.get_similar_words(model, query_word="idx__123", top_k=topk_remain)
    # 若本轮训练所有向量都变了：
    fx.rebuild_index(model)   # 每轮重建（你的场景更合适）

    """
    def __init__(self, model, prefix: str = "idx__"):
        self.prefix = prefix
        self.filtered_word_to_idx: Dict[str, int] = {}
        self.idx_to_filtered_word: Dict[int, str] = {}
        self.dimension = int(model.vector_size)
        self.index = faiss.IndexFlatIP(self.dimension)  # inner product(Combined with L2 normalization = cosine)

        # collect vectors starting with "idx"
        words = [w for w in model.wv.index_to_key if w.startswith(self.prefix)]
        if not words:
            return

        mat = np.asarray([model.wv[w] for w in words], dtype=np.float32)
        faiss.normalize_L2(mat)           # normalization,  inner product = cosine
        self.index.add(mat)               # add into index
        for i, w in enumerate(words):
            self.filtered_word_to_idx[w] = i
            self.idx_to_filtered_word[i] = w

    def _word_exists(self, word: str) -> bool:
        return word in self.filtered_word_to_idx

    def get_similar_words(
        self,
        model,
        query_word: str,
        top_k: int = 5
    ) -> Optional[List[Tuple[str, float]]]:
        """
        Perform Top-K on the “same prefix subset” for a given query_word (must start with prefix).
        Return [(neighbor_word, cosine_sim), ...], sorted in descending order by similarity.
        """
        if not self._word_exists(query_word):
            return None

        # get the queried vector and normalization (1, d)
        q = np.asarray(model.wv[query_word], dtype=np.float32)[None, :]
        faiss.normalize_L2(q)

        # search topk + 1 (remove the word itself)
        k = min(top_k + 1, self.index.ntotal)
        D, I = self.index.search(q, k)  # D: inner product(=cosine), I: index
        sims = []
        for dist, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            neighbor = self.idx_to_filtered_word.get(int(idx))
            if neighbor is None or neighbor == query_word:
                continue
            sims.append((neighbor, float(dist)))

        # get top_k and sort
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:top_k]

    def update_index(self, model) -> None:
        """
        Only add indexes for “newly appearing prefix words.”
        Note: If the vectors of old words have also changed (which is common in your scenario), you should use rebuild_index.
        """
        new_words = [
            w for w in model.wv.index_to_key
            if w.startswith(self.prefix) and w not in self.filtered_word_to_idx
        ]
        if not new_words:
            return

        new_vecs = np.asarray([model.wv[w] for w in new_words], dtype=np.float32)
        faiss.normalize_L2(new_vecs)
        base = self.index.ntotal
        self.index.add(new_vecs)
        for i, w in enumerate(new_words):
            idx = base + i
            self.filtered_word_to_idx[w] = idx
            self.idx_to_filtered_word[idx] = w

    def rebuild_index(self, model, id_num: Optional[int] = None) -> None:
        """
        Full rebuild: Use this when “all vectors change” or you want to force an overwrite in the order idx__0..idx__N.
        If id_num is provided, keys are generated in the order 0..id_num; otherwise, the dictionary is scanned.
        """
        self.index.reset()
        self.filtered_word_to_idx.clear()
        self.idx_to_filtered_word.clear()

        if id_num is not None:
            words = [f"{self.prefix}{i}" for i in range(id_num + 1)
                     if f"{self.prefix}{i}" in model.wv.key_to_index]
        else:
            words = [w for w in model.wv.index_to_key if w.startswith(self.prefix)]

        if not words:
            return

        mat = np.asarray([model.wv[w] for w in words], dtype=np.float32)
        faiss.normalize_L2(mat)
        self.index.add(mat)
        for i, w in enumerate(words):
            self.filtered_word_to_idx[w] = i
            self.idx_to_filtered_word[i] = w


class IdxMatrix:
    """
    pure CPU / Numpy, with embedding model only
    get all top_k at once
    """
    @staticmethod
    def build_idx_matrix(kv, prefix: str = "idx__") -> Tuple[List[str], np.ndarray]:
        """
        get matrix of 'idx__' and corresponding vectors (L2-normalized)
        """
        keys = [w for w in kv.key_to_index if w.startswith(prefix)]
        if not keys:
            return [], np.zeros((0, kv.vector_size), dtype=np.float32)
        # Map the keys (list of "idx__") to their row numbers (indexes) in kv.vectors.
        idxs = np.fromiter((kv.key_to_index[w] for w in keys), dtype=np.int64, count=len(keys))
        # E: vector matrix containing only the words “idx__”
        E = kv.vectors[idxs].astype(np.float32, copy=True)
        # Perform L2 normalization on each row vector: divide each vector by its norm to make its length equal to 1
        E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
        return keys, E
    
    @staticmethod
    def pick_block(N: int, dtype=np.float32, budget_mb: int = 512) -> int:
        """
        Choose a block size B so that the temp similarity matrix S=(B,N) roughly fits the memory budget.
        S bytes ~= B * N * itemsize
        """
        if N <= 0:
            return 0
        bytes_per = np.dtype(dtype).itemsize
        B = int(budget_mb * 1024 * 1024 / (N * bytes_per))
        B = max(1, min(N, B))
        # Align to 128 for better BLAS/cache behavior
        if B >= 128:
            B = (B // 128) * 128
        return B

    @staticmethod
    def topk_all_cosine(E: np.ndarray, k: int, budget_mb: int = 512) -> Tuple[np.ndarray, np.ndarray]:
        """
        param: E (N, d) normalized
        return: D, I like (N, k)
        """
        # N: number of vectors (number of rows)
        N = E.shape[0]
        if N == 0:
            return np.zeros((0, 0), dtype=np.float32), np.zeros((0, 0), dtype=np.int32)

        # take K = k+1 candidates to exclude self-matching 
        K = min(k + 1, N)

        # block: block size, each block generates a similarity submatrix of shape (block, N)
        # When N is small, process the entire block directly (block=N). If N is large, reduce the block size 2048 to control memory usage 
        # block = pick_block(N, dtype=np.float32, budget_mb=512)
        block = IdxMatrix.pick_block(N, dtype=E.dtype, budget_mb=budget_mb)
        if block <= 0:
            block = min(N, 2048)

        k_exe = min(k, N - 1)  # cannot return more than N-1 neighbors per row
        all_I = np.empty((N, k_exe), dtype=np.int32)
        all_D = np.empty((N, k_exe), dtype=np.float32)

        for i in range(0, N, block):
            # get current block, size (block, N)
            E_block = E[i:i + block]
            # Calculate the similarity matrix S between the current block and the entire vector
            # size: (block, N), value:  inner product after normalization = cosine 
            S = E_block @ E.T
            rows = S.shape[0]
            # r = [0, 1, ..., block-1]
            r = np.arange(rows)
            # Set the diagonal position to -inf to ensure that it will not be selected as a match
            S[r, i + r] = -np.inf

            # idx_part: matrix (block, K), the set of column indexes for each row of Top-K candidates (not sorted)
            K_exe = min(K, N)
            idx_part = np.argpartition(S, -K_exe, axis=1)[:, -K_exe:]
            # vals_part: matrix (block, K), the corresonding cosine sim value for each index above
            vals_part = S[np.arange(rows)[:, None], idx_part]
            
            # order: “local sort index” for each row in descending order
            order = np.argsort(-vals_part, axis=1)
            I = idx_part[np.arange(rows)[:, None], order][:, :k]
            D = vals_part[np.arange(rows)[:, None], order][:, :k]

            all_I[i:i + rows] = I
            all_D[i:i + rows] = D

            # free large temporaries early
            del S, idx_part, vals_part, order

        return all_D, all_I

    @staticmethod
    def to_records(
        keys: List[str],
        D: np.ndarray,
        I: np.ndarray,
        round_cnt: int,
        topk_remain: int,
        filter_fn=None
    ) -> List[dict]:
        records: List[dict] = []
        N = len(keys)
        if N == 0 or D.size == 0 or I.size == 0:
            return records
        
        k = D.shape[1]
        for t_idx in range(N):
            # desc_sort by similarity 
            neighs = [(keys[j], float(D[t_idx, r])) for r, j in enumerate(I[t_idx, :k])]
            if filter_fn is not None:
                neighs = filter_fn(keys[t_idx], neighs)
            for rank, (nid, sim) in enumerate(neighs[:topk_remain], start=1):
                records.append({
                    "round": round_cnt,
                    "target_id": keys[t_idx],
                    "neighbor_rank": rank,
                    "neighbor_id": nid,
                    "similarity": sim
                })
        return records

    @staticmethod
    def ratio_rnn_edges(
        D: np.ndarray,
        I: np.ndarray,
        ratio: float = None,
        delta: float = 0.1,
        enforce_rnn: bool = True,
        single_threshold: float = 0.95,
        max_degree: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        param D: A similarity matrix for the top-K items in each row, shape (N, K) 
        param I: A neighbor index matrix for the top-K neighbors in each row, shape (N, K)
        param ratio: (Ratio Test) Take the top two elements from each row: s1=D[:,0], s2=D[:,1]. Requires s1/s2 ≥ ratio (default 1.2).
        param delta: (Difference Test) Optional. Requires s1 - s2 ≥ delta.
        param enforce_rnn: (RNN Mutual Nearest Neighbors) Candidate (row, col) pairs are retained only if the other's top1 is also itself.
        param single_threshold: Direct pass threshold when K < 2
        param max_degree: If a node appears in more than max_degree candidates, it is considered a hub and discarded immediately.

        method:
        基于 Top-K 结果做：
        1) 比值检验（默认 s1/s2 >= ratio）
        2) （可选）差值检验（s1 - s2 >= delta）
        3) （可选）互为最近邻 RNN 过滤
        
        (by default, up to one edge per row, i.e., the top-ranked candidate)
        return rows: Candidate edges consisting of rows
        return cols: Candidate edges consisting of cols
        return scores: Candidate edges consisting of scores 
        """

        N, K = D.shape
        if K < 1:
            return np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.float32)
        
        # -------------------------
        # Step 1: Hubness filtering
        # -------------------------
        all_neighbors = I.flatten()
        degrees = np.bincount(all_neighbors, minlength=N)
        bad_nodes = set(np.where(degrees > max_degree)[0])

        s1 = D[:, 0]
        j1 = I[:, 0]

        # ----------------------------
        # (1) 强制直通：s1 >= 阈值的直接保留
        # ----------------------------
        force_mask = (s1 >= single_threshold) & (~np.isin(j1, list(bad_nodes))) & (~np.isin(np.arange(N), list(bad_nodes)))
        force_rows = np.where(force_mask)[0]
        force_cols = j1[force_mask]
        force_scores = s1[force_mask]

        # 如果所有都满足阈值，直接返回
        if force_mask.all():
            return force_rows.astype(np.int32), force_cols.astype(np.int32), force_scores.astype(np.float32)

        # ----------------------------
        # (2) 其他 row 再做 ratio/delta
        # ----------------------------
        remain_mask = ~force_mask
        if K < 2:
            # K=1 的情况，除了阈值直通的，其余直接丢掉
            rows, cols, scores = np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.float32)
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

        # ----------------------------
        # (3) 合并结果
        # ----------------------------
        rows = np.concatenate([force_rows, rows])
        cols = np.concatenate([force_cols, cols])
        scores = np.concatenate([force_scores, scores])

        # ----------------------------
        # (4) enforce RNN
        # ----------------------------
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
            keep = (best_row_for_col[cols] == rows)
            rows = rows[keep]
            cols = cols[keep]
            scores = scores[keep]

        return rows.astype(np.int32), cols.astype(np.int32), scores.astype(np.float32)

    def r1nn_only(I: np.ndarray, D: np.ndarray, allow_self: bool = False):
        N, K = D.shape
        if K < 1:
            return (np.empty((0,), dtype=np.int32),
                    np.empty((0,), dtype=np.int32),
                    np.empty((0,), dtype=np.float32))

        rows = np.arange(N, dtype=np.int32)
        j1 = I[:, 0].astype(np.int32)

        # 处理可能的无效索引（例如 -1 填充或越界）
        valid = (j1 >= 0) & (j1 < N)
        if not np.all(valid):
            rows_v = rows[valid]
            j1_v = j1[valid]
            partner = I[j1_v, 0]
            mask = np.zeros(N, dtype=bool)
            mask[valid] = (partner == rows_v)
            if not allow_self:
                mask[valid] &= (j1_v != rows_v)
        else:
            partner = I[j1, 0]
            mask = (partner == rows)
            if not allow_self:
                mask &= (j1 != rows)

        out_rows = rows[mask].astype(np.int32)
        out_cols = j1[mask].astype(np.int32)
        out_scores = D[mask, 0].astype(np.float32)
        return out_rows, out_cols, out_scores



    # ============ 一个把三步串起来的便捷方法（可选） ============

    @staticmethod
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
        """
        端到端：
          TopK -> 比值检验(+RNN) -> 三角一致性清洗 -> 贪心一对一
        返回 (match_rows, match_cols, match_scores)
        """
        D, I = IdxMatrix.topk_all_cosine(E, k=10, budget_mb=512)
        rows, cols, scores = IdxMatrix.ratio_rnn_edges(D, I, ratio=ratio, delta=delta, enforce_rnn=enforce_rnn, single_threshold=single_threshold)
        # rows, cols, scores = IdxMatrix.r1nn_only(I, D)
        # if rows.size == 0:
        #     return rows, cols, scores
        # N = E.shape[0]
        # r2, c2, s2 = IdxMatrix.triangle_consistency_clean(
        #     N, rows, cols, scores,
        #     alpha=triangle_alpha,
        #     undirected=triangle_undirected,
        #     max_deg=triangle_max_deg
        # )
        # if r2.size == 0:
        #     return r2, c2, s2
        # r3, c3, s3 = IdxMatrix.greedy_one_to_one(N, r2, c2, s2)
        return rows, cols, scores 
    

def filter_result(target: str, similarity_list: List[Tuple[str, float]], source_num: int, prefix: str = "idx__") -> List[Tuple[str, float]]:
    """
    Port of your _filter_result, made independent of self:
    - if target_id <= source_num: keep neighbors with id >  source_num
    - else                      : keep neighbors with id <= source_num
    """
    result: List[Tuple[str, float]] = []
    if not similarity_list:
        return result

    t_id = parse_idx_suffix(target, prefix=prefix)
    if t_id is None:
        return result

    src = int(source_num)
    for nid, score in similarity_list:
        n_id = parse_idx_suffix(nid, prefix=prefix)
        if n_id is None:
            continue
        if t_id <= src:
            if n_id > src:
                result.append((nid, score))
        else:
            if n_id <= src:
                result.append((nid, score))
    return result

def dynentity_resolution(model, target, n):
    filtered_keys = [word for word in model.wv.index_to_key if word.startswith("idx__")] # only search words beginning with "idx__"
    sims = []
    # sims = [(word, score) for word, score in model.wv.most_similar(target, topn=n*10) if word in filtered_keys and score >0.5][:n]
    sims = [(word, score) for word, score in model.wv.most_similar(target, topn=n*10) if word in filtered_keys][:n]
    # sims = model.wv.most_similar(target, topn=10, restrict_vocab=len(filtered_keys))  # get other similar words
    return sims

# def er_caise(model, pairs_info, n_graph):
#     df = pd.read_csv(pairs_info)
#     for index, row in df.iterrows():
#         table2_id = row['table2.id']
#         if table2_id <= n_graph:
#             model.similarity(table2_id, row['table1.id'])
#         print(row['c1'], row['c2'])
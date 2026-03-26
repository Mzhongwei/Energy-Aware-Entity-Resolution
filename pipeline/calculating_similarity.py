import faiss
import numpy as np
from typing import Dict, List, Optional, Tuple

from utils.utils import parse_idx_suffix


class FaissIndex:
    def __init__(self, model, prefix: str = "idx__"):
        self.prefix = prefix
        self.filtered_word_to_idx: Dict[str, int] = {}
        self.idx_to_filtered_word: Dict[int, str] = {}
        self.dimension = int(model.vector_size)
        self.index = faiss.IndexFlatIP(self.dimension)

        words = [w for w in model.wv.index_to_key if w.startswith(self.prefix)]
        if not words:
            return

        mat = np.asarray([model.wv[w] for w in words], dtype=np.float32)
        faiss.normalize_L2(mat)
        self.index.add(mat)
        for i, w in enumerate(words):
            self.filtered_word_to_idx[w] = i
            self.idx_to_filtered_word[i] = w

    def _word_exists(self, word: str) -> bool:
        return word in self.filtered_word_to_idx

    def get_similar_words(
        self,
        model,
        query_word: str,
        top_k: int = 5,
    ) -> Optional[List[Tuple[str, float]]]:
        if not self._word_exists(query_word):
            return None

        q = np.asarray(model.wv[query_word], dtype=np.float32)[None, :]
        faiss.normalize_L2(q)

        k = min(top_k + 1, self.index.ntotal)
        D, I = self.index.search(q, k)
        sims = []
        for dist, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            neighbor = self.idx_to_filtered_word.get(int(idx))
            if neighbor is None or neighbor == query_word:
                continue
            sims.append((neighbor, float(dist)))

        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:top_k]

    def update_index(self, model) -> None:
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
        self.index.reset()
        self.filtered_word_to_idx.clear()
        self.idx_to_filtered_word.clear()

        if id_num is not None:
            words = [f"{self.prefix}{i}" for i in range(id_num + 1) if f"{self.prefix}{i}" in model.wv.key_to_index]
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
    @staticmethod
    def build_idx_matrix(kv, prefix: str = "idx__") -> Tuple[List[str], np.ndarray]:
        keys = [w for w in kv.key_to_index if w.startswith(prefix)]
        if not keys:
            return [], np.zeros((0, kv.vector_size), dtype=np.float32)
        idxs = np.fromiter((kv.key_to_index[w] for w in keys), dtype=np.int64, count=len(keys))
        E = kv.vectors[idxs].astype(np.float32, copy=True)
        E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
        return keys, E

    @staticmethod
    def pick_block(N: int, dtype=np.float32, budget_mb: int = 512) -> int:
        if N <= 0:
            return 0
        bytes_per = np.dtype(dtype).itemsize
        B = int(budget_mb * 1024 * 1024 / (N * bytes_per))
        B = max(1, min(N, B))
        if B >= 128:
            B = (B // 128) * 128
        return B

    @staticmethod
    def topk_all_cosine(E: np.ndarray, k: int, budget_mb: int = 512) -> Tuple[np.ndarray, np.ndarray]:
        N = E.shape[0]
        if N == 0:
            return np.zeros((0, 0), dtype=np.float32), np.zeros((0, 0), dtype=np.int32)

        K = min(k + 1, N)
        block = IdxMatrix.pick_block(N, dtype=E.dtype, budget_mb=budget_mb)
        if block <= 0:
            block = min(N, 2048)

        k_exe = min(k, N - 1)
        all_I = np.empty((N, k_exe), dtype=np.int32)
        all_D = np.empty((N, k_exe), dtype=np.float32)

        for i in range(0, N, block):
            E_block = E[i:i + block]
            S = E_block @ E.T
            rows = S.shape[0]
            r = np.arange(rows)
            S[r, i + r] = -np.inf

            K_exe = min(K, N)
            idx_part = np.argpartition(S, -K_exe, axis=1)[:, -K_exe:]
            vals_part = S[np.arange(rows)[:, None], idx_part]
            order = np.argsort(-vals_part, axis=1)
            I = idx_part[np.arange(rows)[:, None], order][:, :k]
            D = vals_part[np.arange(rows)[:, None], order][:, :k]

            all_I[i:i + rows] = I
            all_D[i:i + rows] = D
            del S, idx_part, vals_part, order

        return all_D, all_I

    @staticmethod
    def to_records(
        keys: List[str],
        D: np.ndarray,
        I: np.ndarray,
        round_cnt: int,
        topk_remain: int,
        filter_fn=None,
    ) -> List[dict]:
        records: List[dict] = []
        N = len(keys)
        if N == 0 or D.size == 0 or I.size == 0:
            return records

        k = D.shape[1]
        for t_idx in range(N):
            neighs = [(keys[j], float(D[t_idx, r])) for r, j in enumerate(I[t_idx, :k])]
            if filter_fn is not None:
                neighs = filter_fn(keys[t_idx], neighs)
            for rank, (nid, sim) in enumerate(neighs[:topk_remain], start=1):
                records.append(
                    {
                        "round": round_cnt,
                        "target_id": keys[t_idx],
                        "neighbor_rank": rank,
                        "neighbor_id": nid,
                        "similarity": sim,
                    }
                )
        return records


def filter_result(target: str, similarity_list: List[Tuple[str, float]], source_num: int, prefix: str = "idx__") -> List[Tuple[str, float]]:
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
    filtered_keys = [word for word in model.wv.index_to_key if word.startswith("idx__")]
    return [(word, score) for word, score in model.wv.most_similar(target, topn=n * 10) if word in filtered_keys][:n]


__all__ = ["FaissIndex", "IdxMatrix", "dynentity_resolution", "filter_result"]

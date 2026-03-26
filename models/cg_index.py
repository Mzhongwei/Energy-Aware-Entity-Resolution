from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from abc import ABC, abstractmethod
import json
import os


# ----------------------------
# Driver Interface
# ----------------------------

class IndexDriver(ABC):
    @abstractmethod
    def upsert(self, batch: Iterable[Tuple[int, Any]]) -> None:
        raise NotImplementedError

    @abstractmethod
    def query(self, computed_keys_or_struct: Any) -> Set[int]:
        raise NotImplementedError

    def commit(self) -> None:
        """
        Optional hook: merge delta into base.
        Default no-op for drivers without delta.
        """
        return

    @abstractmethod
    def persist(self, index_dir: str) -> None:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, index_dir: str) -> "IndexDriver":
        raise NotImplementedError


# ----------------------------
# Key/token BlockingDriver (base + delta)
# ----------------------------

class BlockingDriver(IndexDriver):
    FILE = "blocking.json"

    def __init__(self) -> None:
        self.base: Dict[str, Set[int]] = {}
        self.delta: Dict[str, Set[int]] = {}

    def upsert(self, batch: Iterable[Tuple[int, List[str]]]) -> None:
        # write into delta (not visible until commit)
        for eid, keys in batch:
            eid = int(eid)
            for k in keys:
                self.delta.setdefault(k, set()).add(eid)

    def query(self, computed_keys: List[str]) -> Set[int]:
        # query base only (committed / visible)
        out: Set[int] = set()
        for k in computed_keys:
            out |= self.base.get(k, set())
        return out

    def commit(self) -> None:
        # merge delta into base
        if not self.delta:
            return
        for k, ids in self.delta.items():
            self.base.setdefault(k, set()).update(ids)
        self.delta.clear()

    def persist(self, index_dir: str) -> None:
        os.makedirs(index_dir, exist_ok=True)
        path = os.path.join(index_dir, self.FILE)
        tmp = path + ".tmp"
        # persist base only (delta is uncommitted)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({k: sorted(v) for k, v in self.base.items()}, f, ensure_ascii=False)
        os.replace(tmp, path)

    @classmethod
    def load(cls, index_dir: str) -> "BlockingDriver":
        inst = cls()
        path = os.path.join(index_dir, cls.FILE)
        if not os.path.exists(path):
            return inst
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        inst.base = {k: set(map(int, ids)) for k, ids in data.items()}
        inst.delta = {}  # always empty after load
        return inst


# ----------------------------
# minhash lsh Driver (base + delta)
# ----------------------------

class MinhashlshDriver(IndexDriver):
    """
    bucket_id (int) -> set[rid]
    """
    FILE = "minhash_lsh_buckets.json"

    def __init__(self) -> None:
        self.base: Dict[int, Set[int]] = {}
        self.delta: Dict[int, Set[int]] = {}

    def upsert(self, batch: Iterable[Tuple[int, List[int]]]) -> None:
        # write into delta (not visible until commit)
        for eid, bucket_ids in batch:
            eid = int(eid)
            for bid in bucket_ids:
                bid = int(bid)
                self.delta.setdefault(bid, set()).add(eid)

    def query(self, bucket_ids: List[int]) -> Set[int]:
        # query base only (committed / visible)
        out: Set[int] = set()
        for bid in bucket_ids:
            out |= self.base.get(int(bid), set())
        return out

    def commit(self) -> None:
        if not self.delta:
            return
        for bid, ids in self.delta.items():
            self.base.setdefault(bid, set()).update(ids)
        self.delta.clear()

    def persist(self, index_dir: str) -> None:
        os.makedirs(index_dir, exist_ok=True)
        path = os.path.join(index_dir, self.FILE)
        tmp = path + ".tmp"
        payload = {str(k): sorted(v) for k, v in self.base.items()}  # JSON key must be str
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp, path)

    @classmethod
    def load(cls, index_dir: str) -> "MinhashlshDriver":
        inst = cls()
        path = os.path.join(index_dir, cls.FILE)
        if not os.path.exists(path):
            return inst
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        inst.base = {int(k): set(map(int, ids)) for k, ids in data.items()}
        inst.delta = {}
        return inst


# ----------------------------
# CGIndex shell (with commit boundary)
# ----------------------------

@dataclass
class CGIndex:
    """
    Storage for candidate generation index.

    Visibility rule:
      - upsert() writes into delta (not queryable)
      - commit_incremental() makes delta visible (merge into base)
      - query() returns only committed (base) results
    """
    build_id: str
    dataset_fp: str
    method: str
    blocking_spec: List[str]
    index_dir: str
    key_schema_id: str = "v1"

    # next_id tracks max_rid_exclusive seen so far (may include uncommitted)
    next_id: int = 0
    # committed_next_id tracks visibility boundary for fullindexing
    committed_next_id: int = 0

    _driver: Optional[IndexDriver] = None

    DRIVER_MAP = {
        "key-blocking": BlockingDriver,
        "token-blocking": BlockingDriver,
        "minhash-lsh": MinhashlshDriver,
    }

    def driver(self) -> Optional[IndexDriver]:
        if self.method == "fullindexing":
            return None

        if self._driver is None:
            cls = self.DRIVER_MAP.get(self.method)
            if cls is None:
                raise NotImplementedError(f"driver not implemented for method={self.method}")
            if os.path.exists(os.path.join(self.index_dir, "manifest.json")):
                self._driver = cls.load(self.index_dir)
            else:
                self._driver = cls()
        return self._driver

    def upsert(self, batch: Iterable[Tuple[int, Any]]) -> None:
        '''
        upsert values into a cg_index, make them queriable in order to facilitate the query
        
        :param batch: (rid, features), 
        :type batch: Iterable[Tuple[int, Any]]
        '''
        batch = list(batch)

        # update next_id (max_rid_exclusive)
        max_eid = None
        for eid, _ in batch:
            eid = int(eid)
            max_eid = eid if max_eid is None else max(max_eid, eid)
        if max_eid is not None:
            self.next_id = max(self.next_id, max_eid + 1)

        if self.method == "fullindexing":
            return

        drv = self.driver()
        if drv is None:
            return
        drv.upsert(batch)

    def commit(self) -> None:
        """
        Make all staged (delta) updates visible for query.
        """
        if self.method != "fullindexing":
            drv = self.driver()
            if drv is not None:
                drv.commit()

        # visibility boundary for fullindexing and metadata
        self.committed_next_id = self.next_id

    def query(self, computed_keys_or_struct: Any) -> Set[int]:
        """
        query: features -> rid

        :param computed_keys_or_struct: features
        :type computed_keys_or_struct: Any
        :return: set of rid
        :rtype: Set[int]
        """
        if self.method == "fullindexing":
            # only committed ids are visible
            return set(range(self.committed_next_id))

        drv = self.driver()
        if drv is None:
            return set()
        return drv.query(computed_keys_or_struct)

    def persist(self) -> None:
        """
        write this CGIndex into local file
        """
        os.makedirs(self.index_dir, exist_ok=True)

        # Persist only committed base state.
        # (If caller wants durability, call commit() before persist().)
        if self.method != "fullindexing":
            drv = self.driver()
            if drv is not None:
                drv.persist(self.index_dir)

        tmp = os.path.join(self.index_dir, "manifest.json.tmp")
        path = os.path.join(self.index_dir, "manifest.json")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "build_id": self.build_id,
                    "dataset_fp": self.dataset_fp,
                    "method": self.method,
                    "blocking_spec": self.blocking_spec,
                    "key_schema_id": self.key_schema_id,
                    "next_id": self.next_id,
                    "committed_next_id": self.committed_next_id,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        os.replace(tmp, path)

    @classmethod
    def load(cls, index_dir: str) -> "CGIndex":
        """
        get a CGIndex from local file
        """
        path = os.path.join(index_dir, "manifest.json")
        with open(path, "r", encoding="utf-8") as f:
            m = json.load(f)
        return cls(
            build_id=m["build_id"],
            dataset_fp=m["dataset_fp"],
            method=m["method"],
            blocking_spec=list(m.get("blocking_spec", [])),
            index_dir=index_dir,
            key_schema_id=m.get("key_schema_id", "v1"),
            next_id=int(m.get("next_id", 0)),
            committed_next_id=int(m.get("committed_next_id", m.get("next_id", 0))),
        )


# ----------------------------
# Minimal demo
# ----------------------------
if __name__ == "__main__":
    idx = CGIndex(
        build_id="build_001",
        dataset_fp="customers_prod@watermark:2026-02-24T10:00:00Z",
        method="key-blocking",
        blocking_spec=["name", "address"],
        index_dir="./index_root/build_001",
        key_schema_id="normalize_simple|sep=|",
        next_id=0,
        committed_next_id=0,
    )

    # upsert goes to delta => NOT visible yet
    idx.upsert([
        (1, ["acme|1 main st sf"]),
        (2, ["acme|1 main st sf"]),
        (3, ["globex|99 market st"]),
    ])
    print(idx.query(["acme|1 main st sf"]))  # expected: set()

    # commit makes it visible
    idx.commit()
    print(idx.query(["acme|1 main st sf"]))  # expected: {1,2}

    idx.persist()

    idx2 = CGIndex.load("./index_root/build_001")
    print(idx2.query(["globex|99 market st"]))  # expected: {3}
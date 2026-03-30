from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from abc import ABC, abstractmethod
import json
import os


def _normalize_record_id(record_id: Any) -> str:
    if record_id is None:
        raise ValueError("record_id cannot be None")
    record_id = str(record_id).strip()
    if not record_id:
        raise ValueError("record_id cannot be empty")
    return record_id


class IndexDriver(ABC):
    @abstractmethod
    def build(self, batch: Iterable[Tuple[str, Any]]) -> None:
        raise NotImplementedError

    @abstractmethod
    def query(self, computed_keys_or_struct: Any) -> Set[str]:
        raise NotImplementedError

    @abstractmethod
    def persist(self, index_dir: str) -> None:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, index_dir: str) -> "IndexDriver":
        raise NotImplementedError


class FullIndexingDriver(IndexDriver):
    FILE = "fullindexing.json"

    def __init__(self) -> None:
        self.record_ids: Set[str] = set()

    def build(self, batch: Iterable[Tuple[str, Any]]) -> None:
        for record_id, _ in batch:
            self.record_ids.add(_normalize_record_id(record_id))

    def query(self, computed_keys_or_struct: Any) -> Set[str]:
        return set(self.record_ids)

    def persist(self, index_dir: str) -> None:
        os.makedirs(index_dir, exist_ok=True)
        path = os.path.join(index_dir, self.FILE)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(sorted(self.record_ids), f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

    @classmethod
    def load(cls, index_dir: str) -> "FullIndexingDriver":
        inst = cls()
        path = os.path.join(index_dir, cls.FILE)
        if not os.path.exists(path):
            return inst
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        inst.record_ids = {str(v) for v in data}
        return inst


class BlockingDriver(IndexDriver):
    FILE = "blocking.json"

    def __init__(self) -> None:
        self.base: Dict[str, Set[str]] = {}

    def build(self, batch: Iterable[Tuple[str, List[str]]]) -> None:
        for record_id, keys in batch:
            record_id = _normalize_record_id(record_id)
            for key in keys or []:
                self.base.setdefault(str(key), set()).add(record_id)

    def query(self, computed_keys: List[str]) -> Set[str]:
        out: Set[str] = set()
        for key in computed_keys or []:
            out |= self.base.get(str(key), set())
        return out

    def persist(self, index_dir: str) -> None:
        os.makedirs(index_dir, exist_ok=True)
        path = os.path.join(index_dir, self.FILE)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({k: sorted(v) for k, v in self.base.items()}, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

    @classmethod
    def load(cls, index_dir: str) -> "BlockingDriver":
        inst = cls()
        path = os.path.join(index_dir, cls.FILE)
        if not os.path.exists(path):
            return inst
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        inst.base = {str(k): {str(v) for v in ids} for k, ids in data.items()}
        return inst


class MinhashlshDriver(IndexDriver):
    FILE = "minhash_lsh_buckets.json"

    def __init__(self) -> None:
        self.base: Dict[int, Set[str]] = {}

    def build(self, batch: Iterable[Tuple[str, List[int]]]) -> None:
        for record_id, bucket_ids in batch:
            record_id = _normalize_record_id(record_id)
            for bucket_id in bucket_ids or []:
                self.base.setdefault(int(bucket_id), set()).add(record_id)

    def query(self, bucket_ids: List[int]) -> Set[str]:
        out: Set[str] = set()
        for bucket_id in bucket_ids or []:
            out |= self.base.get(int(bucket_id), set())
        return out

    def persist(self, index_dir: str) -> None:
        os.makedirs(index_dir, exist_ok=True)
        path = os.path.join(index_dir, self.FILE)
        tmp = path + ".tmp"
        payload = {str(k): sorted(v) for k, v in self.base.items()}
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

    @classmethod
    def load(cls, index_dir: str) -> "MinhashlshDriver":
        inst = cls()
        path = os.path.join(index_dir, cls.FILE)
        if not os.path.exists(path):
            return inst
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        inst.base = {int(k): {str(v) for v in ids} for k, ids in data.items()}
        return inst


@dataclass
class CGIndex:
    build_id: str
    dataset_fp: str
    method: str
    blocking_spec: List[str]
    index_dir: str
    key_schema_id: str = "v1"
    _driver: Optional[IndexDriver] = None

    DRIVER_MAP = {
        "fullindexing": FullIndexingDriver,
        "key-blocking": BlockingDriver,
        "token-blocking": BlockingDriver,
        "minhash-lsh": MinhashlshDriver,
    }

    @classmethod
    def from_config(cls, config: Dict[str, Any], *, index_dir: str, build_id: Optional[str] = None, dataset_fp: Optional[str] = None, blocking_spec: Optional[List[str]] = None) -> "CGIndex":
        cg_cfg = config.get("candidate_generation", {}) if isinstance(config, dict) else {}
        method = str(cg_cfg.get("method", "fullindexing")).strip()
        driver_cls = cls.DRIVER_MAP.get(method)
        if driver_cls is None:
            raise ValueError(f"Unsupported candidate_generation.method '{method}'")
        return cls(
            build_id=str(build_id if build_id is not None else cg_cfg.get("build_id", f"cg_{method}")),
            dataset_fp=str(dataset_fp if dataset_fp is not None else cg_cfg.get("dataset_fp", "")),
            method=method,
            blocking_spec=list(blocking_spec or []),
            index_dir=index_dir,
            key_schema_id=str(cg_cfg.get("key_schema_id", "v1")),
            _driver=driver_cls(),
        )

    def build(self, features: Iterable[Tuple[str, Any]]) -> None:
        if self._driver is None:
            raise ValueError("CGIndex driver is not initialized")
        self._driver.build(features)

    def query(self, computed_keys_or_struct: Any) -> List[str]:
        if self._driver is None:
            raise ValueError("CGIndex driver is not initialized")
        return sorted(self._driver.query(computed_keys_or_struct))

    def persist(self) -> None:
        os.makedirs(self.index_dir, exist_ok=True)
        if self._driver is None:
            raise ValueError("CGIndex driver is not initialized")
        self._driver.persist(self.index_dir)

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
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        os.replace(tmp, path)

    @classmethod
    def from_disk(cls, config: Dict[str, Any], index_dir: str) -> "CGIndex":
        path = os.path.join(index_dir, "manifest.json")
        with open(path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        driver_cls = cls.DRIVER_MAP.get(manifest["method"])
        if driver_cls is None:
            raise ValueError(f"Unsupported stored CGIndex method '{manifest['method']}'")
        index = cls(
            build_id=manifest["build_id"],
            dataset_fp=manifest["dataset_fp"],
            method=manifest["method"],
            blocking_spec=list(manifest.get("blocking_spec", [])),
            index_dir=index_dir,
            key_schema_id=manifest.get("key_schema_id", "v1"),
            _driver=driver_cls.load(index_dir),
        )
        cg_cfg = config.get("candidate_generation", {}) if isinstance(config, dict) else {}
        expected_method = str(cg_cfg.get("method", index.method)).strip()
        if expected_method and index.method != expected_method:
            raise ValueError(
                f"CGIndex method mismatch: config expects '{expected_method}', stored index is '{index.method}'."
            )
        return index

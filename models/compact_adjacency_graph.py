"""Compact, unweighted graph builder and mmap-backed CSR reader."""
from __future__ import annotations

from array import array
import ast
import json
import os
import shutil
import tempfile
import time
from functools import lru_cache

import numpy as np
import pandas as pd

from models.graph_backend import is_appear, is_first, is_root, legacy_class_to_flags
from utils.utils import convert_token_value


def _missing(value) -> bool:
    if isinstance(value, dict):
        return not value
    if isinstance(value, (list, tuple, set)):
        return not value or all(_missing(item) for item in value)
    result = pd.isna(value)
    return bool(np.asarray(result).all()) if isinstance(result, (np.ndarray, pd.Series)) else bool(result)


class CompactAdjacencyGraph:
    """Append-friendly graph whose node IDs remain stable for its lifetime."""

    def __init__(self, node_types=(), flatten=(), directed=False, ngram_config=None):
        self.directed = bool(directed)
        self.name_to_id: dict[str, int] = {}
        self.id_to_name: list[str] = []
        self.node_class_flags = bytearray()
        self._node_types: list[str] = []
        self._adjacency: list[array] = []
        self._dirty = True
        self.dyn_roots: set[int] = set()
        self.node_classes = {}
        self.to_flatten = flatten if flatten != "no" else []
        self.ngram_config = ngram_config if isinstance(ngram_config, dict) else None
        for spec in node_types or ():
            class_info, name = spec.split("__", 1)
            self.node_classes[name] = legacy_class_to_flags(int(class_info[0]))

    @classmethod
    def load_snapshot(cls, path, node_types=(), flatten=(), ngram_config=None):
        reader = CSRGraphReader(path, mmap=True)
        graph = cls(node_types=node_types, flatten=flatten, directed=reader.directed,
                    ngram_config=ngram_config)
        graph.id_to_name = [reader.get_node_name(i) for i in range(reader.vertex_count())]
        graph.name_to_id = {name: i for i, name in enumerate(graph.id_to_name)}
        graph.node_class_flags = bytearray(reader.node_class_flags)
        graph._node_types = [""] * reader.vertex_count()
        typecode = "I" if reader.indices.dtype == np.uint32 else "Q"
        graph._adjacency = [array(typecode, reader.neighbors(i)) for i in range(reader.vertex_count())]
        graph.dyn_roots = set(reader.dyn_roots)
        graph._dirty = False
        return graph

    def vertex_count(self):
        return len(self.id_to_name)

    def get_vertex_index(self, name):
        return self.name_to_id.get(name)

    def get_node_name(self, node_id):
        return self.id_to_name[int(node_id)]

    def get_node_class_flags(self, node_id):
        return int(self.node_class_flags[int(node_id)])

    def is_first(self, node_id):
        return is_first(self.get_node_class_flags(node_id))

    def is_root(self, node_id):
        return is_root(self.get_node_class_flags(node_id))

    def is_appear(self, node_id):
        return is_appear(self.get_node_class_flags(node_id))

    def get_or_create_node(self, name, node_type, node_class_flags=None):
        node_id = self.name_to_id.get(name)
        if node_id is not None:
            return node_id
        node_id = len(self.id_to_name)
        flags = self.node_classes.get(node_type, 0) if node_class_flags is None else int(node_class_flags)
        self.name_to_id[name] = node_id
        self.id_to_name.append(str(name))
        self._node_types.append(str(node_type))
        self.node_class_flags.append(flags)
        self._adjacency.append(array("I"))
        self._dirty = True
        return node_id

    def _update_node(self, name, node_type):
        node_id = self.get_or_create_node(name, node_type)
        if self.is_root(node_id):
            self.dyn_roots.add(node_id)
        return node_id

    _update_token = _update_node

    def add_edge(self, src_id, dst_id):
        src_id, dst_id = int(src_id), int(dst_id)
        self._adjacency[src_id].append(dst_id)
        if not self.directed and src_id != dst_id:
            self._adjacency[dst_id].append(src_id)
        self._dirty = True

    def _finalize(self):
        if not self._dirty:
            return
        started = time.perf_counter()
        typecode = "I" if self.vertex_count() <= np.iinfo(np.uint32).max else "Q"
        for index, row in enumerate(self._adjacency):
            if len(row) > 1:
                self._adjacency[index] = array(typecode, sorted(set(row)))
            elif row.typecode != typecode:
                self._adjacency[index] = array(typecode, row)
        self._dirty = False
        print(f"[compact-graph] dedup_seconds={time.perf_counter() - started:.6f} "
              f"num_nodes={self.vertex_count()} num_edges={self.edge_count()}")

    def neighbors(self, node_id):
        self._finalize()
        return self._adjacency[int(node_id)]

    def edge_count(self):
        self._finalize()
        total = sum(map(len, self._adjacency))
        return total if self.directed else total // 2

    @lru_cache(maxsize=200_000)
    def _tokenize_cached(self, value):
        return tuple(token for token in str(value).strip().split("_") if token)

    def _instance_nodes(self, value, prefix):
        nodes = {self._update_node(f"tt__{value}", prefix)}
        if self.to_flatten == "all" or prefix in self.to_flatten:
            nodes.update(self._update_token(f"tt__{token}", "st") for token in self._tokenize_cached(value))
        return nodes

    @staticmethod
    def _normalize_rid(value):
        if isinstance(value, (list, tuple)):
            return [str(item) for item in value if item not in ("", None)]
        if isinstance(value, str):
            try:
                parsed = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                parsed = value
            if isinstance(parsed, (list, tuple)):
                return [str(item) for item in parsed if item not in ("", None)]
        return [str(value)]

    @staticmethod
    def _ngrams(tokens, sizes, skip=0):
        if skip:
            for size in sizes:
                for start in range(len(tokens)):
                    window = tokens[start:min(len(tokens), start + size + skip)]
                    if len(window) >= size:
                        step = max(1, (len(window) - 1) // (size - 1))
                        candidate = window[::step][:size]
                        if len(candidate) == size:
                            yield tuple(candidate)
        else:
            for size in sizes:
                for start in range(len(tokens) - size + 1):
                    yield tuple(tokens[start:start + size])

    def build_relation(self, df):
        started = time.perf_counter()
        if "rid" not in df.columns:
            raise ValueError("Missing 'rid' column during graph construction.")
        columns = [(name, pos) for pos, name in enumerate(df.columns) if name != "rid"]
        rid_pos = list(df.columns).index("rid")
        for values in df.itertuples(index=False, name=None):
            rid_tokens = self._normalize_rid(values[rid_pos])
            if not rid_tokens:
                raise ValueError("Missing rid token during graph construction.")
            rid_id = self._update_node(rid_tokens[0], "idx")
            for column, pos in columns:
                value = values[pos]
                if _missing(value):
                    continue
                cid_id = self._update_node(f"cid__{column}", "cid")
                tokens, numeric = convert_token_value(value)
                if tokens is None:
                    continue
                prefix = "tn" if numeric else "tt"
                for token in tokens:
                    for token_id in self._instance_nodes(token, prefix):
                        self.add_edge(token_id, cid_id)
                        self.add_edge(token_id, rid_id)
                cfg = self.ngram_config
                if cfg and not numeric:
                    for ngram in self._ngrams(tuple(tokens), cfg.get("token_n", []), cfg.get("skip", 0)):
                        ng_id = self._update_token(f"ng::{len(ngram)}::" + "␟".join(ngram), "ng")
                        self.add_edge(ng_id, cid_id)
                        self.add_edge(ng_id, rid_id)
        print(f"[compact-graph] construction_seconds={time.perf_counter() - started:.6f} "
              f"num_nodes={self.vertex_count()}")

    def export_snapshot(self, path, version=0):
        started = time.perf_counter()
        self._finalize()
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        temp_path = tempfile.mkdtemp(prefix=f".{os.path.basename(path)}.", dir=parent)
        try:
            counts = np.fromiter((len(row) for row in self._adjacency), dtype=np.uint64,
                                 count=self.vertex_count())
            indptr = np.empty(self.vertex_count() + 1, dtype=np.uint64)
            indptr[0] = 0
            np.cumsum(counts, out=indptr[1:])
            index_dtype = np.uint32 if self.vertex_count() <= np.iinfo(np.uint32).max else np.uint64
            indices = np.fromiter((item for row in self._adjacency for item in row),
                                  dtype=index_dtype, count=int(indptr[-1]))
            encoded = [name.encode("utf-8") for name in self.id_to_name]
            offsets = np.empty(self.vertex_count() + 1, dtype=np.uint64)
            offsets[0] = 0
            np.cumsum(np.fromiter(map(len, encoded), dtype=np.uint64), out=offsets[1:])
            np.save(os.path.join(temp_path, "indptr.npy"), indptr, allow_pickle=False)
            np.save(os.path.join(temp_path, "indices.npy"), indices, allow_pickle=False)
            np.save(os.path.join(temp_path, "node_class.npy"), np.frombuffer(self.node_class_flags, dtype=np.uint8), allow_pickle=False)
            np.save(os.path.join(temp_path, "node_names_offsets.npy"), offsets, allow_pickle=False)
            with open(os.path.join(temp_path, "node_names_utf8.bin"), "wb") as names_file:
                names_file.write(b"".join(encoded))
            np.save(os.path.join(temp_path, "dyn_roots.npy"), np.asarray(sorted(self.dyn_roots), dtype=index_dtype), allow_pickle=False)
            manifest = {"version": int(version), "format": "csr-v1", "directed": self.directed,
                        "num_nodes": self.vertex_count(), "num_edges": self.edge_count(),
                        "node_id_dtype": np.dtype(index_dtype).name, "offset_dtype": "uint64"}
            with open(os.path.join(temp_path, "manifest.json"), "w", encoding="utf-8") as output:
                json.dump(manifest, output, sort_keys=True, indent=2)
            if os.path.exists(path):
                shutil.rmtree(path)
            os.replace(temp_path, path)
        except Exception:
            shutil.rmtree(temp_path, ignore_errors=True)
            raise
        size = sum(os.path.getsize(os.path.join(path, name)) for name in os.listdir(path))
        print(f"[compact-graph] csr_export_seconds={time.perf_counter() - started:.6f} snapshot_bytes={size}")
        return path


class CSRGraphReader:
    def __init__(self, path, mmap=True):
        started = time.perf_counter()
        mode = "r" if mmap else None
        with open(os.path.join(path, "manifest.json"), encoding="utf-8") as source:
            self.manifest = json.load(source)
        if self.manifest.get("format") != "csr-v1":
            raise ValueError(f"Unsupported graph snapshot format: {self.manifest.get('format')}")
        self.directed = bool(self.manifest["directed"])
        self.indptr = np.load(os.path.join(path, "indptr.npy"), mmap_mode=mode, allow_pickle=False)
        self.indices = np.load(os.path.join(path, "indices.npy"), mmap_mode=mode, allow_pickle=False)
        self.node_class_flags = np.load(os.path.join(path, "node_class.npy"), mmap_mode=mode, allow_pickle=False)
        self._name_offsets = np.load(os.path.join(path, "node_names_offsets.npy"), mmap_mode=mode, allow_pickle=False)
        names_path = os.path.join(path, "node_names_utf8.bin")
        self._name_bytes = (np.memmap(names_path, dtype=np.uint8, mode="r")
                            if os.path.getsize(names_path) else np.empty(0, dtype=np.uint8))
        self.dyn_roots = set(map(int, np.load(os.path.join(path, "dyn_roots.npy"), allow_pickle=False)))
        self._name_to_id = None
        print(f"[compact-graph] csr_mmap_open_seconds={time.perf_counter() - started:.6f} "
              f"num_nodes={self.vertex_count()} num_edges={self.manifest['num_edges']}")

    def vertex_count(self):
        return int(self.manifest["num_nodes"])

    def get_node_name(self, node_id):
        start, end = map(int, self._name_offsets[int(node_id):int(node_id) + 2])
        return bytes(self._name_bytes[start:end]).decode("utf-8")

    def get_vertex_index(self, name):
        if self._name_to_id is None:
            self._name_to_id = {self.get_node_name(node_id): node_id for node_id in range(self.vertex_count())}
        return self._name_to_id.get(name)

    def get_node_class_flags(self, node_id):
        return int(self.node_class_flags[int(node_id)])

    def is_first(self, node_id):
        return is_first(self.get_node_class_flags(node_id))

    def is_root(self, node_id):
        return is_root(self.get_node_class_flags(node_id))

    def is_appear(self, node_id):
        return is_appear(self.get_node_class_flags(node_id))

    def neighbors(self, node_id):
        start, end = map(int, self.indptr[int(node_id):int(node_id) + 2])
        return self.indices[start:end]

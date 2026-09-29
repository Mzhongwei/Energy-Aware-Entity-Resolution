"""Compact, unweighted graph builder and mmap-backed CSR reader."""
from __future__ import annotations

from array import array
import ast
import json
import mmap as mmap_module
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
        # Rows appended to since the last dedup; only these are re-sorted by _finalize().
        self._dirty_rows: set[int] = set()
        self._typecode = "I"
        # UTF-8 names kept in snapshot layout so an export does not re-encode every name.
        self._name_blob = bytearray()
        self._name_ends = array("Q")
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
        graph._typecode = "I" if reader.indices.dtype == np.uint32 else "Q"
        graph._adjacency = [array(graph._typecode, reader.neighbors(i).tobytes())
                            for i in range(reader.vertex_count())]
        graph._name_blob = bytearray(reader._name_bytes[:])
        graph._name_ends = array("Q", reader._name_offsets[1:].astype(np.uint64).tobytes())
        graph.dyn_roots = set(reader.dyn_roots)
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
        self._name_blob += str(name).encode("utf-8")
        self._name_ends.append(len(self._name_blob))
        self._node_types.append(str(node_type))
        self.node_class_flags.append(flags)
        self._adjacency.append(array(self._typecode))
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
        self._dirty_rows.add(src_id)
        if not self.directed and src_id != dst_id:
            self._adjacency[dst_id].append(src_id)
            self._dirty_rows.add(dst_id)

    def _finalize(self):
        typecode = "I" if self.vertex_count() <= np.iinfo(np.uint32).max else "Q"
        if typecode != self._typecode:
            # Crossing the uint32 node-ID limit widens every row, not just the dirty ones.
            self._dirty_rows.update(range(self.vertex_count()))
            self._typecode = typecode
        if not self._dirty_rows:
            return
        started = time.perf_counter()
        dirty_count = len(self._dirty_rows)
        for index in self._dirty_rows:
            row = self._adjacency[index]
            if len(row) > 1:
                self._adjacency[index] = array(typecode, sorted(set(row)))
            elif row.typecode != typecode:
                self._adjacency[index] = array(typecode, row)
        self._dirty_rows.clear()
        print(f"[compact-graph] dedup_seconds={time.perf_counter() - started:.6f} "
              f"dirty_rows={dirty_count} num_nodes={self.vertex_count()} num_edges={self.edge_count()}")

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
            counts = np.fromiter(map(len, self._adjacency), dtype=np.uint64,
                                 count=self.vertex_count())
            indptr = np.empty(self.vertex_count() + 1, dtype=np.uint64)
            indptr[0] = 0
            np.cumsum(counts, out=indptr[1:])
            index_dtype = np.uint32 if self._typecode == "I" else np.uint64
            # Every row shares the index typecode after _finalize(), so one C-level join
            # concatenates their buffers in CSR order.
            indices = np.frombuffer(b"".join(self._adjacency), dtype=index_dtype)
            offsets = np.empty(self.vertex_count() + 1, dtype=np.uint64)
            offsets[0] = 0
            offsets[1:] = np.frombuffer(self._name_ends, dtype=np.uint64)
            np.save(os.path.join(temp_path, "indptr.npy"), indptr, allow_pickle=False)
            np.save(os.path.join(temp_path, "indices.npy"), indices, allow_pickle=False)
            np.save(os.path.join(temp_path, "node_class.npy"), np.frombuffer(self.node_class_flags, dtype=np.uint8), allow_pickle=False)
            np.save(os.path.join(temp_path, "node_names_offsets.npy"), offsets, allow_pickle=False)
            with open(os.path.join(temp_path, "node_names_utf8.bin"), "wb") as names_file:
                names_file.write(self._name_blob)
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
    """Read-only CSR snapshot view.

    Only ``indices`` (O(E)) and the UTF-8 name blob stay mmap-backed. The O(V) per-node
    arrays are loaded into RAM because the walk touches them on every step, and indexing a
    numpy memmap scalar costs a Python-level ``memmap.__getitem__`` call each time.
    """

    def __init__(self, path, mmap=True):
        started = time.perf_counter()
        mode = "r" if mmap else None
        with open(os.path.join(path, "manifest.json"), encoding="utf-8") as source:
            self.manifest = json.load(source)
        if self.manifest.get("format") != "csr-v1":
            raise ValueError(f"Unsupported graph snapshot format: {self.manifest.get('format')}")
        self.directed = bool(self.manifest["directed"])
        self.indptr = np.load(os.path.join(path, "indptr.npy"), allow_pickle=False)
        self.indices = np.load(os.path.join(path, "indices.npy"), mmap_mode=mode, allow_pickle=False)
        # Plain ndarray view of the same pages: slicing skips memmap's Python overrides.
        self._indices = self.indices.view(np.ndarray)
        self.node_class_flags = np.load(os.path.join(path, "node_class.npy"), allow_pickle=False).tobytes()
        self._name_offsets = np.load(os.path.join(path, "node_names_offsets.npy"), allow_pickle=False)
        with open(os.path.join(path, "node_names_utf8.bin"), "rb") as names_file:
            if mmap and os.path.getsize(names_file.name):
                self._name_bytes = mmap_module.mmap(names_file.fileno(), 0, access=mmap_module.ACCESS_READ)
            else:
                self._name_bytes = names_file.read()
        self.dyn_roots = set(map(int, np.load(os.path.join(path, "dyn_roots.npy"), allow_pickle=False)))
        self._names: dict[int, str] = {}
        self._name_to_id = None
        print(f"[compact-graph] csr_mmap_open_seconds={time.perf_counter() - started:.6f} "
              f"num_nodes={self.vertex_count()} num_edges={self.manifest['num_edges']}")

    def vertex_count(self):
        return int(self.manifest["num_nodes"])

    def get_node_name(self, node_id):
        # Decoded once per reader; repeated tokens also share one str object.
        node_id = int(node_id)
        name = self._names.get(node_id)
        if name is None:
            start = int(self._name_offsets[node_id])
            end = int(self._name_offsets[node_id + 1])
            name = self._names[node_id] = self._name_bytes[start:end].decode("utf-8")
        return name

    def get_node_names(self, node_ids):
        """Decode many IDs at once; ``node_ids`` is any integer array."""
        unique, inverse = np.unique(np.asarray(node_ids), return_inverse=True)
        names = [self.get_node_name(node_id) for node_id in unique.tolist()]
        return [names[position] for position in inverse.ravel().tolist()]

    def get_vertex_index(self, name):
        if self._name_to_id is None:
            self._name_to_id = {self.get_node_name(node_id): node_id for node_id in range(self.vertex_count())}
        return self._name_to_id.get(name)

    def get_node_class_flags(self, node_id):
        return self.node_class_flags[int(node_id)]

    def is_first(self, node_id):
        return is_first(self.get_node_class_flags(node_id))

    def is_root(self, node_id):
        return is_root(self.get_node_class_flags(node_id))

    def is_appear(self, node_id):
        return is_appear(self.get_node_class_flags(node_id))

    def neighbors(self, node_id):
        node_id = int(node_id)
        return self._indices[self.indptr[node_id]:self.indptr[node_id + 1]]

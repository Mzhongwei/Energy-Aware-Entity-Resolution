"""Backend-neutral graph contracts and compact node-class encoding."""
from __future__ import annotations

from typing import Optional, Protocol, Sequence

ISFIRST = 1 << 0
ISROOT = 1 << 1
ISAPPEAR = 1 << 2


def encode_node_class(isfirst: bool = False, isroot: bool = False, isappear: bool = False) -> int:
    return ((ISFIRST if isfirst else 0) | (ISROOT if isroot else 0) |
            (ISAPPEAR if isappear else 0))


def is_first(flags: int) -> bool:
    return bool(int(flags) & ISFIRST)


def is_root(flags: int) -> bool:
    return bool(int(flags) & ISROOT)


def is_appear(flags: int) -> bool:
    return bool(int(flags) & ISAPPEAR)


def legacy_class_to_flags(value: int) -> int:
    """Translate the existing three-bit ``isfirst/isroot/isappear`` integer."""
    value = int(value)
    return encode_node_class(bool(value & 4), bool(value & 2), bool(value & 1))


class GraphReader(Protocol):
    directed: bool
    dyn_roots: set[int]

    def vertex_count(self) -> int: ...
    def get_vertex_index(self, name: str) -> Optional[int]: ...
    def get_node_name(self, node_id: int) -> str: ...
    def get_node_class_flags(self, node_id: int) -> int: ...
    def is_first(self, node_id: int) -> bool: ...
    def is_root(self, node_id: int) -> bool: ...
    def is_appear(self, node_id: int) -> bool: ...
    def neighbors(self, node_id: int) -> Sequence[int]: ...


class GraphBuilder(Protocol):
    directed: bool
    dyn_roots: set[int]

    def get_or_create_node(self, name: str, node_type: str,
                           node_class_flags: int) -> int: ...
    def add_edge(self, src_id: int, dst_id: int) -> None: ...
    def build_relation(self, df) -> None: ...
    def export_snapshot(self, path: str, version: int) -> str: ...

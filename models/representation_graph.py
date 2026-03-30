from __future__ import annotations

from typing import Dict, List, Optional

import igraph as ig
from igraph import Graph


class RepresentationGraph:
    def __init__(self, directed: bool = False) -> None:
        self.graph: Graph = ig.Graph(directed=directed)
        self.name2idx: Dict[str, int] = {}
        self.graph["weighted"] = directed

    def rebuild_index_map(self) -> None:
        self.name2idx.clear()
        for v in self.graph.vs:
            self.name2idx[v["name"]] = int(v.index)

    def clean_attributes(self) -> Graph:
        allowed_types = (str, int, float, bool)
        for v in self.graph.vs:
            for attr in list(v.attributes()):
                if not isinstance(v[attr], allowed_types):
                    del v[attr]
        for e in self.graph.es:
            for attr in list(e.attributes()):
                if not isinstance(e[attr], allowed_types):
                    del e[attr]
        return self.graph

    def load_graph(self, graph_file: str) -> None:
        self.graph = Graph.Read_GraphML(graph_file)
        if "weighted" not in self.graph.attributes():
            self.graph["weighted"] = self.graph.is_directed()

        self.rebuild_index_map()
        self._after_graph_loaded()

    def _after_graph_loaded(self) -> None:
        return

    def _after_topology_changed(self) -> None:
        return

    def get_directed_info(self) -> bool:
        return self.graph.is_directed()

    def get_graph(self) -> Graph:
        return self.graph

    def get_neighbors(self, node_name):
        if node_name not in self.name2idx:
            return []
        idx = self.name2idx[node_name]
        neighbors = self.graph.neighbors(idx, mode="ALL")
        return [self.graph.vs[n]["name"] for n in neighbors]

    def show_summary(self) -> None:
        print(self.graph.summary())

    def get_node_attribute(self, node_name, attr):
        return self.graph.vs.find(name=node_name)[attr]

    def set_node_attribute(self, node_name, attr, value) -> None:
        self.graph.vs.find(name=node_name)[attr] = value

    def get_record(self, id_index):
        neighbors = self.graph.neighbors(id_index, mode="OUT")
        return [self.graph.vs[n]["name"] for n in neighbors if self.graph.vs[n]["type"] != "st"]

    def _add_vertex(self, node_name: str, node_type: Optional[str] = None, **attrs):
        if node_name in self.name2idx:
            return self.graph.vs[self.name2idx[node_name]]
        node_attrs = {
            "name": node_name,
            "appearing_frequency": 0,
            "frequency_in_graph": 1,
            "test_pretraining": False,
            "test_neighbors_freq": {},
        }
        if node_type is not None:
            node_attrs["type"] = node_type
        node_attrs.update(attrs)
        node = self.graph.add_vertex(**node_attrs)
        self.name2idx[node_name] = int(node.index)
        return node

    def _update_node(self, node_name: str, node_type: Optional[str] = None, **attrs) -> int:
        if self.graph.vcount() == 0 or node_name not in self.name2idx:
            node = self._add_vertex(node_name, node_type, **attrs)
            node["frequency_in_graph"] = 1
        else:
            node = self.graph.vs[self.name2idx[node_name]]
            node["frequency_in_graph"] = int(node["frequency_in_graph"]) + 1
        return int(node.index)

    def _update_token(self, ins_value: str, node_type: Optional[str] = None, **attrs) -> int:
        return self._update_node(ins_value, node_type, **attrs)

    def _add_edge(self, node1_index: int, node2_index: int) -> None:
        if not self.graph.are_connected(node1_index, node2_index):
            self.graph.add_edge(node1_index, node2_index)

    def get_vertex_index(self, name: str) -> Optional[int]:
        return self.name2idx.get(name)

    def delete_vertices(self, names: List[str]) -> None:
        idxs = [self.name2idx[n] for n in names if n in self.name2idx]
        if not idxs:
            return
        self.graph.delete_vertices(idxs)
        self.rebuild_index_map()
        self._after_topology_changed()

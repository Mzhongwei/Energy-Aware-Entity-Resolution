from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import igraph as ig
from igraph import Graph


class RepresentationGraph:
    def __init__(self, directed: bool = False) -> None:
        self.graph: Graph = ig.Graph(directed=directed)
        self.name2idx: Dict[str, int] = {}
        self.edge2eid: Dict[Tuple[int, int], int] = {}
        self.graph["weighted"] = directed

    def rebuild_index_map(self) -> None:
        self.name2idx.clear()
        for v in self.graph.vs:
            self.name2idx[v["name"]] = int(v.index)

    def _edge_key(self, node1_index: int, node2_index: int) -> Tuple[int, int]:
        node1_index = int(node1_index)
        node2_index = int(node2_index)
        if self.graph.is_directed():
            return (node1_index, node2_index)
        if node1_index <= node2_index:
            return (node1_index, node2_index)
        return (node2_index, node1_index)

    def rebuild_edge_index(self) -> None:
        self.edge2eid.clear()
        for edge in self.graph.es:
            source, target = edge.tuple
            self.edge2eid[self._edge_key(source, target)] = int(edge.index)

    def get_edge_index(self, node1_index: int, node2_index: int) -> Optional[int]:
        return self.edge2eid.get(self._edge_key(node1_index, node2_index))

    @staticmethod
    def _strip_non_primitive_attrs(g: Graph) -> Graph:
        allowed_types = (str, int, float, bool)
        for v in g.vs:
            for attr in list(v.attributes()):
                if not isinstance(v[attr], allowed_types):
                    del v[attr]
        for e in g.es:
            for attr in list(e.attributes()):
                if not isinstance(e[attr], allowed_types):
                    del e[attr]
        return g

    def clean_attributes(self) -> Graph:
        return self._strip_non_primitive_attrs(self.graph)

    def clean_attributes_copy(self) -> Graph:
        """Strip a copy, leaving self.graph's attributes intact for continued use."""
        return self._strip_non_primitive_attrs(self.graph.copy())

    def load_graph(self, graph_file: str) -> None:
        self.graph = Graph.Read_GraphML(graph_file)
        if "weighted" not in self.graph.attributes():
            self.graph["weighted"] = self.graph.is_directed()

        self.rebuild_index_map()
        self.rebuild_edge_index()
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
        if self.get_edge_index(node1_index, node2_index) is None:
            self._add_edges_batch([(int(node1_index), int(node2_index))])

    def _add_edges_batch(self, edges: List[Tuple[int, int]], weights: Optional[List[float]] = None) -> List[int]:
        if not edges:
            return []

        start_eid = self.graph.ecount()
        if weights is None:
            self.graph.add_edges(edges)
        else:
            self.graph.add_edges(edges, attributes={"weight": weights})

        new_eids = list(range(start_eid, self.graph.ecount()))
        for eid, edge in zip(new_eids, edges):
            self.edge2eid[self._edge_key(edge[0], edge[1])] = eid
        return new_eids

    def get_vertex_index(self, name: str) -> Optional[int]:
        return self.name2idx.get(name)

    def delete_vertices(self, names: List[str]) -> None:
        idxs = [self.name2idx[n] for n in names if n in self.name2idx]
        if not idxs:
            return
        self.graph.delete_vertices(idxs)
        self.rebuild_index_map()
        self.rebuild_edge_index()
        self._after_topology_changed()

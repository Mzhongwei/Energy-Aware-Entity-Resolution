from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import igraph as ig
from igraph import Graph


class RepresentationGraph:
    def __init__(
        self,
        node_types: Optional[Iterable[str]] = None,
        flatten: Iterable[str] | str = [],
        directed: bool = False,
        smooth: Optional[str] = None,
        meta_path: Optional[Iterable] = None,
    ) -> None:
        self.graph: Graph = ig.Graph(directed=directed)
        self.node_classes: Dict[str, int] = {}
        self.node_is_numeric: Dict[str, bool] = {}
        self.to_flatten = flatten if flatten != "no" else []
        self.meta_path = []
        self.dyn_roots = None
        self.name2idx: Dict[str, int] = {}

        self.graph["num_ids"] = -1
        self.graph["smooth"] = smooth
        self.graph["weighted"] = directed
        self.graph["words_count"] = 0

        if (not meta_path) and node_types:
            self._extract_node_types(node_types)
            self._check_flatten()

        if meta_path:
            self.meta_path = meta_path
            self.meta_link = []
            self.meta_node = set()
            paths = meta_path if isinstance(meta_path[0], list) else [meta_path]
            for path in paths:
                for a, b in zip(path[:-1], path[1:]):
                    link = tuple(sorted([a, b]))
                    if link not in self.meta_link:
                        self.meta_link.append(link)
                for node_type in path:
                    self.meta_node.add(node_type)

        if meta_path:
            self.dyn_roots = {}
            if isinstance(meta_path[0], list):
                for path in meta_path:
                    self.dyn_roots[path[0]] = set()
            else:
                self.dyn_roots[meta_path[0]] = set()
        else:
            self.dyn_roots = set()

    def _extract_node_types(self, node_types: Iterable[str]) -> None:
        for node_type in node_types:
            class_info, name = node_type.split("__")
            rwclass = int(class_info[0])
            num_flag = class_info[1]
            self.node_classes[name] = rwclass
            self.node_is_numeric[name] = (num_flag == "#")

    def _check_flatten(self) -> None:
        if self.to_flatten and self.to_flatten != "no":
            for prefix in self.to_flatten:
                if prefix not in self.node_classes:
                    raise ValueError(f"Unknown flatten type: {prefix}")

    def _get_node_type(self, node_name: str) -> str:
        for pre in self.node_classes:
            if node_name.startswith(pre + "__"):
                return pre
        raise ValueError(f"Node {node_name} does not match any known prefix.")

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

        try:
            self.graph["num_ids"] = int(float(self.graph["num_ids"]))
        except Exception:
            self.graph["num_ids"] = -1
        try:
            self.graph["words_count"] = int(float(self.graph["words_count"]))
        except Exception:
            self.graph["words_count"] = 0
        if "smooth" not in self.graph.attributes():
            self.graph["smooth"] = None
        if "weighted" not in self.graph.attributes():
            self.graph["weighted"] = self.graph.is_directed()

        self.rebuild_index_map()
        if not self.meta_path:
            for v in self.graph.vs:
                v["node_class"] = self._update_node_class(v["type"]) if "type" in v.attributes() else {
                    "isfirst": False,
                    "isroot": False,
                    "isappear": False,
                }
        self._after_graph_loaded()

    def _after_graph_loaded(self) -> None:
        return

    def _after_topology_changed(self) -> None:
        return

    def get_smooth_method(self):
        return self.graph["smooth"]

    def get_directed_info(self) -> bool:
        return self.graph.is_directed()

    def set_id_nums(self, id_num) -> None:
        self.graph["num_ids"] = id_num

    def accum_id_nums(self) -> None:
        self.graph["num_ids"] += 1

    def get_id_nums(self):
        return self.graph["num_ids"]

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

    def _add_vertex(self, node_name: str, node_prefix: str):
        if node_name in self.name2idx:
            return self.graph.vs[self.name2idx[node_name]]

        if self.meta_path:
            node = self.graph.add_vertex(
                name=node_name,
                type=node_prefix,
                appearing_frequency=0,
                frequency_in_graph=1,
                test_pretraining=False,
                test_neighbors_freq={},
            )
        else:
            node = self.graph.add_vertex(
                name=node_name,
                type=node_prefix,
                numeric=self.node_is_numeric.get(node_prefix, False),
                node_class=self._update_node_class(node_prefix),
                appearing_frequency=0,
                frequency_in_graph=1,
                test_pretraining=False,
                test_neighbors_freq={},
            )
        self.name2idx[node_name] = int(node.index)
        return node

    def _update_node(self, node_name: str, node_prefix: str) -> int:
        if self.graph.vcount() == 0 or node_name not in self.name2idx:
            node = self._add_vertex(node_name, node_prefix)
            node["frequency_in_graph"] = 1
        else:
            node = self.graph.vs[self.name2idx[node_name]]
            node["frequency_in_graph"] = int(node["frequency_in_graph"]) + 1
        self.graph["words_count"] = int(self.graph["words_count"]) + 1

        if self.meta_path:
            if node_prefix in getattr(self, "dyn_roots", {}).keys():
                self.dyn_roots[node_prefix].add(int(node.index))
        else:
            if node["node_class"].get("isroot", False):
                self.dyn_roots.add(int(node.index))
        return int(node.index)

    def _update_token(self, ins_value: str, prefix: str) -> int:
        if self.graph.vcount() == 0 or ins_value not in self.name2idx:
            node = self._add_vertex(ins_value, prefix)
        else:
            node = self.graph.vs[self.name2idx[ins_value]]
            node["frequency_in_graph"] = int(node["frequency_in_graph"]) + 1
        if self.meta_path:
            if prefix in getattr(self, "dyn_roots", {}).keys():
                self.dyn_roots[prefix].add(int(node.index))
        else:
            if node["node_class"]["isroot"]:
                self.dyn_roots.add(int(node.index))
        return int(node.index)

    def _update_node_class(self, prefix: str) -> Dict[str, bool]:
        node_class_bin = "{:03b}".format(self.node_classes[prefix])
        return {
            "isfirst": bool(int(node_class_bin[0])),
            "isroot": bool(int(node_class_bin[1])),
            "isappear": bool(int(node_class_bin[2])),
        }

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

from __future__ import annotations
from collections import defaultdict
from datetime import datetime
from functools import lru_cache
from typing import Dict, List, Tuple, Iterable, Optional, Set

import math
import pandas as pd
from tqdm import tqdm

from models.representation_graph import RepresentationGraph
from pipeline.sampler import NodeSampler
from utils.write_log import write_log
from utils.utils import OUTPUT_FORMAT, TIME_FORMAT, convert_token_value

app_debug = write_log("logs", "debug", "dynamic_graph")


def _graph_config(configuration):
    if not isinstance(configuration, dict):
        return {}
    graph_cfg = configuration.get("graph_construction")
    if isinstance(graph_cfg, dict):
        return graph_cfg
    legacy_cfg = configuration.get("graph")
    if isinstance(legacy_cfg, dict):
        return legacy_cfg
    return {}


class DynGraphIgraph(RepresentationGraph):
    def __init__(
        self,
        node_types: Optional[Iterable[str]] = None,
        flatten: Iterable[str] | str = [],
        directed: bool = False,
        smooth: Optional[str] = None,
        meta_path: Optional[Iterable] = None,
        ngram_config: Optional[Dict] = None,
        rare_bias: Optional[str] = "idf",
        rare_alpha: float = 1.0,
        undirected_weighted: bool = True,
        cache_ngram_size: int = 200_000,
    ) -> None:
        super().__init__(directed=directed)

        self.node_classes: Dict[str, int] = {}
        self.node_is_numeric: Dict[str, bool] = {}
        self.to_flatten = self._resolve_flatten(flatten)
        self.meta_path = meta_path or []
        self.meta_link: List[Tuple[str, str]] = []
        self.meta_node: Set[str] = set()
        self.dyn_roots = None

        if node_types:
            self._extract_node_types(node_types)
        self._init_meta_path(meta_path)
        self._check_flatten()

        self.samplers = []
        self.ngram_config = ngram_config if isinstance(ngram_config, dict) else None
        self._cache_ngram_size = cache_ngram_size
        self.rare_bias = rare_bias
        self.rare_alpha = float(rare_alpha)
        self.undirected_weighted = undirected_weighted
        self.graph["smooth"] = smooth
        self.graph["words_count"] = 0
        self.graph["weighted"] = bool(smooth) or bool(undirected_weighted) or directed

    def _resolve_flatten(self, flatten: Iterable[str] | str):
        if flatten == "no":
            return []
        return flatten

    def _extract_node_types(self, node_types: Iterable[str]) -> None:
        for node_type in node_types:
            class_info, name = node_type.split("__")
            rwclass = int(class_info[0])
            num_flag = class_info[1]
            self.node_classes[name] = rwclass
            self.node_is_numeric[name] = (num_flag == "#")

    def _init_meta_path(self, meta_path: Optional[Iterable]) -> None:
        if meta_path:
            paths = meta_path if isinstance(meta_path[0], list) else [meta_path]
            for path in paths:
                for a, b in zip(path[:-1], path[1:]):
                    link = tuple(sorted([a, b]))
                    if link not in self.meta_link:
                        self.meta_link.append(link)
                for node_type in path:
                    self.meta_node.add(node_type)

            self.dyn_roots = {}
            if isinstance(meta_path[0], list):
                for path in meta_path:
                    self.dyn_roots[path[0]] = set()
            else:
                self.dyn_roots[meta_path[0]] = set()
        else:
            self.dyn_roots = set()

    def _check_flatten(self) -> None:
        if self.to_flatten and self.to_flatten not in ("no", "all"):
            for prefix in self.to_flatten:
                if prefix not in self.node_classes:
                    raise ValueError(f"Unknown flatten type: {prefix}")

    def _update_node_class(self, prefix: str) -> Dict[str, bool]:
        node_class_bin = "{:03b}".format(self.node_classes[prefix])
        return {
            "isfirst": bool(int(node_class_bin[0])),
            "isroot": bool(int(node_class_bin[1])),
            "isappear": bool(int(node_class_bin[2])),
        }

    def _after_graph_loaded(self) -> None:
        if "words_count" not in self.graph.attributes():
            self.graph["words_count"] = 0
        if "smooth" not in self.graph.attributes():
            self.graph["smooth"] = None
        if "weighted" not in self.graph.attributes():
            self.graph["weighted"] = bool(self.graph["smooth"]) or bool(self.undirected_weighted) or self.graph.is_directed()
        if not self.meta_path:
            for v in self.graph.vs:
                if "type" not in v.attributes():
                    continue
                if "numeric" not in v.attributes():
                    v["numeric"] = self.node_is_numeric.get(v["type"], False)
                node_class = v["node_class"] if "node_class" in v.attributes() else None
                if (not isinstance(node_class, dict)) and v["type"] in self.node_classes:
                    v["node_class"] = self._update_node_class(v["type"])
        self.samplers = [None] * self.graph.vcount()
        for v in self.graph.vs:
            self._update_neighbors(int(v.index))
            v["test_pretraining"] = True
            v["test_neighbors_freq"] = {}

    def _after_topology_changed(self) -> None:
        self.samplers = [None] * self.graph.vcount()
        for v in self.graph.vs:
            self._update_neighbors(int(v.index))

    def _add_vertex(self, node_name: str, node_prefix: str):
        attrs = {}
        if not self.meta_path:
            attrs["numeric"] = self.node_is_numeric.get(node_prefix, False)
            if node_prefix in self.node_classes:
                attrs["node_class"] = self._update_node_class(node_prefix)
        return super()._add_vertex(node_name, node_prefix, **attrs)

    def _update_node(self, node_name: str, node_prefix: str) -> int:
        node_index = super()._update_node(node_name, node_prefix)
        self.graph["words_count"] = int(self.graph["words_count"]) + 1
        node = self.graph.vs[node_index]
        node_class = node["node_class"] if "node_class" in node.attributes() else None

        if self.meta_path:
            if node_prefix in getattr(self, "dyn_roots", {}).keys():
                self.dyn_roots[node_prefix].add(int(node.index))
        else:
            if isinstance(node_class, dict) and node_class.get("isroot", False):
                self.dyn_roots.add(int(node.index))
        return int(node.index)

    def _update_token(self, ins_value: str, prefix: str) -> int:
        node_index = super()._update_token(ins_value, prefix)
        node = self.graph.vs[node_index]
        node_class = node["node_class"] if "node_class" in node.attributes() else None
        if self.meta_path:
            if prefix in getattr(self, "dyn_roots", {}).keys():
                self.dyn_roots[prefix].add(int(node.index))
        else:
            if isinstance(node_class, dict) and node_class.get("isroot", False):
                self.dyn_roots.add(int(node.index))
        return int(node.index)

    # ------------------------
    # n-gram helpers
    # ------------------------
    @lru_cache(maxsize=200_000)
    def _tokenize_cached(self, s: str) -> Tuple[str, ...]:
        return tuple([t for t in str(s).strip().split('_') if t])

    def _gen_ngrams(self, tokens: Tuple[str, ...], token_ns: List[int], skip: int = 0) -> List[Tuple[str, ...]]:
        out: List[Tuple[str, ...]] = []
        L = len(tokens)
        if L == 0:
            return out
        if skip <= 0:
            for n in token_ns:
                if L < n:
                    continue
                for i in range(L - n + 1):
                    out.append(tuple(tokens[i:i + n]))
        else:
            # k-skip-n-gram generation (simple version)
            for n in token_ns:
                if L < n:
                    continue
                for i in range(L):
                    jmax = min(L, i + n + skip)
                    window = tokens[i:jmax]
                    if len(window) >= n:
                        step = max(1, (len(window) - 1) // (n - 1))
                        cand = window[::step][:n]
                        if len(cand) == n:
                            out.append(tuple(cand))
        return out

    def _emit_tokens(self, og_value: str, prefix_for_token: str) -> List[Tuple[str, str]]:
        """Return flattened tokens for a single textual value."""
        res: List[Tuple[str, str]] = []
        toks = self._tokenize_cached(og_value)
        if len(toks) == 0:
            return res

        for t in toks:
            if self.meta_path:
                res.append((t, prefix_for_token))
            else:
                res.append((t, 'st'))
        return res

    def _normalize_cell_tokens(self, cell_value, column_name: str) -> Tuple[List[str], bool]:
        if isinstance(cell_value, list):
            tokens = [str(el) for el in cell_value if el not in ("", None)]
            return tokens, False
        if column_name == "rid":
            return [str(cell_value)], False
        return convert_token_value(cell_value)

    # ------------------------
    # Sampling Weights and Rare-Item Bias
    # ------------------------
    def _rare_weight_of_vertex(self, vidx: int) -> float:
        if self.rare_bias is None:
            return self._add_edge_weight(vidx)
        if self.rare_bias == 'idf':
            freq = int(self.graph.vs[vidx]["frequency_in_graph"])
            return (1.0 / (freq + 1.0)) ** self.rare_alpha
        if self.rare_bias == 'degree':
            deg = int(self.graph.degree(vidx, mode='ALL'))
            return (1.0 / (deg + 1.0)) ** self.rare_alpha
        return 1.0

    def _edge_weight_for_sampling(self, v_from: int, v_to: int) -> float:
        try:
            eid = self.graph.get_eid(v_from, v_to)
            base = float(self.graph.es[eid]['weight']) if 'weight' in self.graph.es[eid].attributes() else 1.0
        except Exception:
            base = 1.0
        rare = self._rare_weight_of_vertex(v_to)
        return max(1e-12, base * rare)

    def _add_edge_weight(self, to_index: int):
        # Calculate the weights using the smooth method 
        degree = 1
        if self.graph['smooth'] == 'ICF':
            degree = self.graph.degree(to_index, mode='OUT')
            weight = float(math.log(max(1, int(self.graph.vcount())) / (degree + 1.0)))
        elif self.graph['smooth'] == 'IDF':
            degree = int(self.graph.vs[to_index].get("frequency_in_graph", 0))
            weight = float(math.log(max(1, int(self.graph['words_count'])) / (degree + 1.0)))
        elif self.graph['smooth'] == 'log':
            degree = max(1, int(self.graph.degree(to_index, mode='ALL')))
            weight = 1.0 / (math.log(degree + 1.0) + 1.0)
        else:
            return 1
        return weight    

    def _add_edge(self, node1_index: int, node2_index: int) -> None:
        use_weight = bool(self.graph["weighted"])
        edge_weight = float(self._add_edge_weight(node2_index)) if use_weight else 1.0

        eid = self.graph.get_eid(node1_index, node2_index, error=False)
        if eid != -1:
            if use_weight:
                current = float(self.graph.es[eid]["weight"]) if "weight" in self.graph.es[eid].attributes() else 0.0
                self.graph.es[eid]["weight"] = current + edge_weight
            return

        if use_weight:
            self.graph.add_edge(node1_index, node2_index, weight=edge_weight)
        else:
            self.graph.add_edge(node1_index, node2_index)

    # ------------------------
    # Neighbor Sampler Cache
    # ------------------------
    def _extend_sampler(self, size: int):
        if len(self.samplers) < size:
            self.samplers.extend([None] * (size - len(self.samplers)))

    def _update_neighbors(self, index: int):
        v = self.graph.vs[index]
        neighbors = self.graph.neighbors(index, mode='OUT')  # same thing for directed and undirected graph
        graph = self.graph
        vs = graph.vs

        if self.meta_path:
            graph = self.graph
            vs = graph.vs
            use_w = bool(graph["weighted"])

            # one type -> one bucket
            buckets = defaultdict(lambda: {"neighbors": [], "weights": []})

            for nb in neighbors:
                t = vs[nb]["type"]
                w = self._edge_weight_for_sampling(index, nb) if use_w else 1.0

                buckets[t]["neighbors"].append(nb)
                buckets[t]["weights"].append(w)

            # construct NodeSampler for each type
            sampler_for_index = {}
            for t, data in buckets.items():
                if not data["neighbors"]:
                    continue
                if use_w:
                    sampler_for_index[t] = NodeSampler(
                        neighbors=data["neighbors"],
                        weighted=True,
                        threshold=50,
                        weights=data["weights"]
                    )
                else:
                    sampler_for_index[t] = NodeSampler(
                        neighbors=data["neighbors"],
                        weighted=False,
                        threshold=1000
                    )

            # Make sure the samplers are long enough, assign the values
            self.samplers[index] = sampler_for_index

        else:
            # Weighted paths in undirected graphs
            use_w = bool(graph["weighted"])
            if not use_w:
                self.samplers[index] = NodeSampler(neighbors=neighbors, weighted=False, threshold=1000)
            else:
                weights = [self._edge_weight_for_sampling(index, el) for el in neighbors]
                self.samplers[index] = NodeSampler(neighbors=neighbors, weighted=True, threshold=50, weights=weights)

            if 'node_class' in v.attributes() and not v["node_class"].get("isfirst", True):
                self.samplers[index].update_firstnode_list([vs[idx]["node_class"].get("isfirst", False) for idx in neighbors])

    def get_sampler(self, index: int):
        if index < len(self.samplers):
            return self.samplers[index]
        return None

    # ------------------------
    # Update a cell value in the graph.
    # ------------------------
    def _update_instance_vertex_edge(self, ins_value: str, prefix: str):
        instances_index = set()
        instance_index = self._update_node(f'tt__{ins_value}', prefix)
        instances_index.add(instance_index)

        if self.to_flatten == "all" or prefix in self.to_flatten:
            for val, pfx in self._emit_tokens(ins_value, prefix_for_token=prefix):
                if not val:
                    continue
                val_index = self._update_token(f'tt__{val}', pfx)
                instances_index.add(val_index)

        return instances_index

    # ------------------------
    # Construct graph (meta-path + simple mode + optional n-gram)
    # ------------------------
    def build_relation(self, df):
        affected_nodes = set()
        columns = list(df.columns)
        col_positions = {col: idx for idx, col in enumerate(columns)}
        row_iter = df.itertuples(index=False, name=None)

        if self.meta_path:
            update_instance_vertex_edge = self._update_instance_vertex_edge
            update_node = self._update_node
            add_edge = self._add_edge

            for row_values in tqdm(row_iter, total=len(df), desc="# Building/Updating graph"):
                values = {}
                for col, pos in col_positions.items():
                    if col in self.meta_node:
                        values.setdefault(col, [])
                        token_list, _ = self._normalize_cell_tokens(row_values[pos], col)
                        if token_list is not None and len(token_list) > 0:
                            for el in token_list:
                                index_set = update_instance_vertex_edge(el, col)
                                for index in index_set:
                                    if index not in values[col]:
                                        values[col].append(index)
                                    affected_nodes.add(index)
                        else:
                            index = update_node("nan", col)
                            if index not in values[col]:
                                values[col].append(index)
                            affected_nodes.add(index)
                # app_debug.info(values)
                for a, b in getattr(self, 'meta_link', []):
                    if a in values and b in values:
                        for v1 in values[a]:
                            for v2 in values[b]:
                                add_edge(v1, v2)
        else:
            rid_pos = col_positions.get("rid")
            if rid_pos is None:
                raise ValueError("Missing 'rid' column during graph construction.")

            data_columns = [(col, pos) for col, pos in col_positions.items() if col != "rid"]
            update_node = self._update_node
            update_instance_vertex_edge = self._update_instance_vertex_edge
            add_edge = self._add_edge
            add_ngrams_for_token_list = self._add_ngrams_for_token_list

            for row_values in tqdm(row_iter, total=len(df), desc="# Building/Updating graph"):
                rid_tokens, _ = self._normalize_cell_tokens(row_values[rid_pos], "rid")
                if not rid_tokens:
                    raise ValueError("Missing rid token during graph construction.")
                rid_node = rid_tokens[0]
                rid_index = update_node(rid_node, "idx")
                affected_nodes.add(rid_index)

                for cid_node, pos in data_columns:
                    og_value = row_values[pos]
                    if pd.isna(og_value):
                        continue

                    cid_index = update_node(f'cid__{cid_node}', "cid")
                    affected_nodes.add(cid_index)

                    token_list, is_numeric = convert_token_value(og_value)
                    if token_list is not None:
                        node_prefix = "tn" if is_numeric else "tt"
                        for el in token_list:
                            instance_index = update_instance_vertex_edge(el, node_prefix)
                            for index in instance_index:
                                add_edge(index, cid_index)
                                add_edge(index, rid_index)
                            affected_nodes.update(instance_index)

                        if not is_numeric:
                            index_list = add_ngrams_for_token_list(token_list, cid_index, rid_index)
                            affected_nodes.update(index_list)

        # extend & update samplers
        self._extend_sampler(self.graph.vcount())
        neighbor_updated = set()
        for index in affected_nodes:
            self._update_neighbors(index)
            neighbor_updated.add(index)

            neighbors = self.graph.neighbors(index, mode='OUT')
            for neigh in neighbors:
                if neigh not in neighbor_updated:
                    self._update_neighbors(neigh)
                    neighbor_updated.add(neigh)

    def _add_ngrams_for_token_list(self, token_list: List[str], cid_index: int, rid_index: int):
        index_list = set()
        cfg = self.ngram_config
        if not cfg:
            return index_list
        token_ns = cfg.get('token_n', [])
        skip = cfg.get('skip', 0)
        if not token_ns:
            return index_list
        toks = tuple(token_list)
        for ng in self._gen_ngrams(toks, token_ns, skip=skip):
            name = f"ng::{len(ng)}::" + "␟".join(ng)
            ng_index = self._update_token(name, 'ng')
            self._add_edge(ng_index, cid_index)
            self._add_edge(ng_index, rid_index)
            index_list.add(ng_index)
        return index_list

def dyn_graph_generation(configuration):
    """
    Generate the graph for the given dataframe following the specifications in configuration.
    :param df: dataframe to transform in graph.
    :param configuration: dictionary with all the run parameters
    :return: the generated graph
    """
    graph_cfg = _graph_config(configuration)
    meta_path = configuration.get("meta_path", graph_cfg.get("meta_path", []))
    if not meta_path:
        flatten_cfg = graph_cfg.get("flatten", [])
        if flatten_cfg:
            flatten_str = str(flatten_cfg)
            if flatten_str.lower() not in ['all', 'false', 'no']:
                flatten = flatten_str.strip().split(',')
            elif flatten_str.lower() == 'false':
                flatten = []
            else:
                flatten = 'all'
        else:
            flatten = []
    else:
        flatten = graph_cfg.get("flatten", [])

    print(f"faltten: {graph_cfg.get('flatten', [])}")
    t_start = datetime.now()
    print(OUTPUT_FORMAT.format('Starting graph construction', t_start.strftime(TIME_FORMAT)))

    node_types = graph_cfg.get("node_types", [])
    directed = graph_cfg.get("directed", False)
    smooth = graph_cfg.get("smoothing_method")
    ngram_config = graph_cfg.get("ngram")
    if not isinstance(ngram_config, dict):
        ngram_config = None
    # convert_token_value() --> token_list
    g = DynGraphIgraph(
        node_types=node_types,
        flatten=flatten,
        directed=directed,
        smooth=smooth,
        ngram_config=ngram_config,
        rare_bias='idf',
        rare_alpha=0.7,
        undirected_weighted=False,
        meta_path=meta_path
    )

    t_end = datetime.now()
    dt = t_end - t_start
    print()
    print(OUTPUT_FORMAT.format('Graph construction complete', t_end.strftime(TIME_FORMAT)))
    print(OUTPUT_FORMAT.format('Time required to build graph:', f'{dt.total_seconds():.2f} seconds.'))
    return g

# ------------------------
# test
# ------------------------
if __name__ == "__main__":
    import pandas as pd

    data = [
        {"rid": 1, "name": "Honey Basil Amber", "cat": "A"},
        {"rid": 2, "name": "Rude Hippo Honey Basil Amber", "cat": "A"},
        {"rid": 3, "name": "Mint Lemon", "cat": "B"},
    ]
    df = pd.DataFrame(data)

    g = DynGraphIgraph(
        node_types=["5#__idx", "0#__cid", "0#__tt", "0#__tn", "0#__ng"],
        flatten=['tt'],
        directed=False,
        smooth='IDF',
        ngram_config={'token_n': [2, 3], 'char_n': [], 'skip': 0},
        rare_bias='idf',
        rare_alpha=0.7,
        undirected_weighted=True,
    )

    g.build_relation(df)
    # check name2idx
    print("Vertex count:", g.graph.vcount())
    print("Edge count:", g.graph.ecount())
    print("Index of 'ng::3::Honey␟Basil␟Amber':", g.get_vertex_index("ng::3::Honey␟Basil␟Amber"))

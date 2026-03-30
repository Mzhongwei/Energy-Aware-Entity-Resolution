from __future__ import annotations
from collections import defaultdict
from datetime import datetime
from functools import lru_cache
from typing import Dict, List, Tuple, Iterable, Optional, Set

import math
from tqdm import tqdm

from models.representation_graph import RepresentationGraph
from pipeline.sampler import NodeSampler
from utils.write_log import write_log
from utils.utils import OUTPUT_FORMAT, TIME_FORMAT, convert_token_value

app_debug = write_log("logs", "debug", "dynamic_graph")


class DynGraphIgraph(RepresentationGraph):
    def __init__(
        self,
        node_types: Optional[Iterable[str]] = None,
        flatten: Iterable[str] | str = [],
        directed: bool = False,
        smooth: Optional[str] = None,
        meta_path: Optional[Iterable] = None,
        ngram_config: Optional[Dict] = {"token_n": [2, 3], "char_n": [], "skip": 0},
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
        self.ngram_config = ngram_config or {"token_n": [2, 3], "char_n": [], "skip": 0}
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
                if "node_class" not in v.attributes() and v["type"] in self.node_classes:
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

        if self.meta_path:
            if node_prefix in getattr(self, "dyn_roots", {}).keys():
                self.dyn_roots[node_prefix].add(int(node.index))
        else:
            if "node_class" in node.attributes() and node["node_class"].get("isroot", False):
                self.dyn_roots.add(int(node.index))
        return int(node.index)

    def _update_token(self, ins_value: str, prefix: str) -> int:
        node_index = super()._update_token(ins_value, prefix)
        node = self.graph.vs[node_index]
        if self.meta_path:
            if prefix in getattr(self, "dyn_roots", {}).keys():
                self.dyn_roots[prefix].add(int(node.index))
        else:
            if "node_class" in node.attributes() and node["node_class"].get("isroot", False):
                self.dyn_roots.add(int(node.index))
        return int(node.index)

    # ------------------------
    # n-gram 生成（可缓存）
    # ------------------------
    @lru_cache(maxsize=200_000)
    def _tokenize_cached(self, s: str) -> Tuple[str, ...]:
        # 与你现有的 '_' 分词一致（若要更复杂，改这里或用 convert_token_value 的 token_list 直接传入）
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
            # 简化的 k-skip-n-gram 生成（可替换成更完整的组合版本）
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

    def _emit_tokens_and_ngrams(self, og_value: str, prefix_for_token: str) -> List[Tuple[str, str]]:
        """返回 [(value_str, prefix)]，既包含 token，也包含 n-gram（前缀 'ng'）。"""
        # app_debug.info(f"og_value: {og_value}")
        res: List[Tuple[str, str]] = []
        toks = self._tokenize_cached(og_value)
        # app_debug.info(f"toks: {toks}")
        if len(toks) == 0:
            return res
        # 原 token
        for t in toks:
            if self.meta_path:
                res.append((t, prefix_for_token))
            else:
                res.append((t, 'st'))

        # # n-gram（按配置）
        # cfg = self.ngram_config
        # if cfg:
        #     token_ns = cfg.get('token_n', [])
        #     skip = cfg.get('skip', 0)

        #     if self.meta_path:
        #         pr = prefix_for_token
        #     else:
        #         pr = "st"

        #     for ng in self._gen_ngrams(toks, token_ns, skip=skip):
        #         name = f"{pr}::{len(ng)}::" + "␟".join(ng)
        #         res.append((name, pr))
        #         app_debug.info(f"debug: name-{name}, pr-{pr}")
        #         app_debug.info(f'res: {res}')
            
            # 字符 n-gram 可在此扩展（char_n）
        return res

    # ------------------------
    # 采样权重与稀有偏置
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
        # 根据 smooth 计算权重（保留你原来的策略）
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

        if self.graph.are_connected(node1_index, node2_index):
            eid = self.graph.get_eid(node1_index, node2_index)
            if use_weight:
                current = float(self.graph.es[eid]["weight"]) if "weight" in self.graph.es[eid].attributes() else 0.0
                self.graph.es[eid]["weight"] = current + edge_weight
            return

        if use_weight:
            self.graph.add_edge(node1_index, node2_index, weight=edge_weight)
        else:
            self.graph.add_edge(node1_index, node2_index)

    # ------------------------
    # 邻居采样器缓存
    # ------------------------
    def _extend_sampler(self, size: int):
        if len(self.samplers) < size:
            self.samplers.extend([None] * (size - len(self.samplers)))

    def _update_neighbors(self, index: int):
        v = self.graph.vs[index]
        neighbors = self.graph.neighbors(index, mode='OUT')  # 无向也等价
        graph = self.graph
        vs = graph.vs

        if self.meta_path:
            graph = self.graph
            vs = graph.vs
            use_w = bool(graph["weighted"])

            # 每种类型各放一桶
            buckets = defaultdict(lambda: {"neighbors": [], "weights": []})

            for nb in neighbors:
                t = vs[nb]["type"]
                w = self._edge_weight_for_sampling(index, nb) if use_w else 1.0

                buckets[t]["neighbors"].append(nb)
                buckets[t]["weights"].append(w)

            # 为每种类型构建 NodeSampler
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

            # 确保 samplers 长度足够，然后赋值
            self.samplers[index] = sampler_for_index

        else:
            # 无向图也可走加权
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
    # 将一个单元格值更新进图（原 token + n-gram）
    # ------------------------
    def _update_instance_vertex_edge(self, ins_value: str, prefix: str):
        instances_index = set()
        # 原始值自身作为一个节点（沿用你的逻辑）
        instance_index = self._update_node(f'tt__{ins_value}', prefix)
        instances_index.add(instance_index)

        # app_debug.info(f'node name: {ins_value}, node index: {instance_index}')
        if self.to_flatten == "all" or prefix in self.to_flatten:
            # 这里将原值展开为 token + n-gram（按 ngram_config）
            # app_debug.info('start flatten')
            for val, pfx in self._emit_tokens_and_ngrams(ins_value, prefix_for_token=prefix):
                # app_debug.info(f'iterate: {val}, prefixe: {pfx}')
                if not val:
                    continue
                val_index = self._update_token(f'tt__{val}', pfx)
                # print(f"after toke, idex-{val_index}, name-{val}")
                instances_index.add(val_index)
                # app_debug.info(f'token name: {val}, token index: {val_index}')

        return instances_index

    # ------------------------
    # 构图（兼容 meta_path 与普通模式）
    # ------------------------
    def build_relation(self, df):
        affected_nodes = set()
        if self.meta_path:
            for _, df_row in tqdm(df.iterrows(), total=len(df), desc="# Building/Updating graph"):
                values = {}
                for col in df.columns:
                    if col in self.meta_node:
                        if col != 'rid':
                            values.setdefault(col, [])
                            token_list, _ = convert_token_value(df_row[col])
                        else:
                            values.setdefault(col, [])
                            token_list = [df_row[col]]
                        if token_list is not None and len(token_list) > 0:
                            for el in token_list:
                                index_set = self._update_instance_vertex_edge(el, col)
                                for index in index_set:
                                    if index not in values[col]:
                                        values[col].append(index)
                                    affected_nodes.add(index)
                        else:
                            index = self._update_node("nan", col)
                            if index not in values[col]:
                                values[col].append(index)
                            affected_nodes.add(index)
                # app_debug.info(values)
                for a, b in getattr(self, 'meta_link', []):
                    if a in values and b in values:
                        for v1 in values[a]:
                            for v2 in values[b]:
                                self._add_edge(v1, v2)
        else:
            for _, df_row in tqdm(df.iterrows(), total=len(df), desc="# Building/Updating graph"):
                rid_node = str(df_row['rid'])
                rid_index = self._update_node(rid_node, "idx")
                affected_nodes.add(rid_index)

                row = df_row.dropna()
                for cid_node in df.columns:
                    if cid_node == "rid":
                        continue
                    try:
                        cid_index = self._update_node(f'cid__{cid_node}', "cid")
                        affected_nodes.add(cid_index)

                        og_value = row[cid_node]
                        # app_debug.info(f"og_value: {og_value}")

                        token_list, is_numeric = convert_token_value(og_value)
                        # app_debug.info(f"token_list: {token_list}")
                        if token_list is not None:
                            for el in token_list:
                                # app_debug.info(f"el if tokens: {el}")
                                # app_debug.info(f'idx_index: {rid_index}, id: {rid_node}')
                                node_prefix = "tn" if is_numeric else "tt"
                                instance_index = self._update_instance_vertex_edge(el, node_prefix)
                                # app_debug.info(f'instance index: {instance_index}')
                                for index in instance_index:
                                    # app_debug.info(f"2. {index}-{self.graph.vs[index]['type']}: {self.graph.vs[index]['name']}, {rid_index}-{self.graph.vs[rid_index]['type']}: {self.graph.vs[rid_index]['name']}")
                                    self._add_edge(index, cid_index)
                                    self._add_edge(index, rid_index)
                                affected_nodes.update(instance_index)

                            # —— 额外：在 token_list 级别直接加 n-gram 节点（跨 token 的 n-gram）——
                            if not is_numeric:
                                index_list = self._add_ngrams_for_token_list(token_list, cid_index, rid_index)
                                affected_nodes.update(index_list)
                    except KeyError:
                        continue

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
            return
        token_ns = cfg.get('token_n', [])
        skip = cfg.get('skip', 0)
        toks = tuple(token_list)
        for ng in self._gen_ngrams(toks, token_ns, skip=skip):
            name = f"ng::{len(ng)}::" + "␟".join(ng)
            ng_index = self._update_token(name, 'ng')
            self._add_edge(ng_index, cid_index)
            self._add_edge(ng_index, rid_index)
            # app_debug.info(f"1. {ng_index}-{self.graph.vs[ng_index]['type']}: {self.graph.vs[ng_index]['name']}, {rid_index}-{self.graph.vs[rid_index]['type']}: {self.graph.vs[rid_index]['name']}")

            index_list.add(ng_index)
        return index_list

def dyn_graph_generation(configuration):
    """
    Generate the graph for the given dataframe following the specifications in configuration.
    :param df: dataframe to transform in graph.
    :param configuration: dictionary with all the run parameters
    :return: the generated graph
    """
    meta_path = configuration['graph']['meta_path']
    if not meta_path:
        if configuration['graph']['flatten']:
            if configuration['graph']['flatten'].lower() not in ['all', 'false', 'no']:
                flatten = configuration['graph']['flatten'].strip().split(',')
            elif configuration['graph']['flatten'].lower() == 'false':
                flatten = []
            else:
                flatten = 'all'
        else:
            flatten = []
    else:
        flatten = configuration['graph']['flatten']

    print(f"faltten: {configuration['graph']['flatten']}")
    t_start = datetime.now()
    print(OUTPUT_FORMAT.format('Starting graph construction', t_start.strftime(TIME_FORMAT)))

    node_types = configuration['graph']['node_types']
    directed = configuration['graph']['directed']
    smooth = configuration['graph']['smoothing_method']
    meta_path = configuration['graph']['meta_path']
    # 你的 convert_token_value 可能按空格切词；这里假设它能把字符串拆成 token_list
    g = DynGraphIgraph(
        node_types=node_types,
        flatten=flatten,
        directed=directed,
        smooth=smooth,
        ngram_config={'token_n': [2, 3], 'char_n': [], 'skip': 0},
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
# 可选：简单的自检
# ------------------------
if __name__ == "__main__":
    import pandas as pd

    data = [
        {"rid": 1, "name": "Honey Basil Amber", "cat": "A"},
        {"rid": 2, "name": "Rude Hippo Honey Basil Amber", "cat": "A"},
        {"rid": 3, "name": "Mint Lemon", "cat": "B"},
    ]
    df = pd.DataFrame(data)

    # 你的 convert_token_value 可能按空格切词；这里假设它能把字符串拆成 token_list
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
    # 检查 name2idx 是否可用
    print("Vertex count:", g.graph.vcount())
    print("Edge count:", g.graph.ecount())
    print("Index of 'ng::3::Honey␟Basil␟Amber':", g.get_vertex_index("ng::3::Honey␟Basil␟Amber"))

from __future__ import annotations
from collections import defaultdict
from datetime import datetime
from functools import lru_cache
from typing import Dict, List, Tuple, Iterable, Optional

import math
import traceback
import numpy as np
from tqdm import tqdm
import pandas as pd
from dynamic_embedding.sampler import NodeSampler
from utils.write_log import write_log
from utils.utils import OUTPUT_FORMAT, TIME_FORMAT, convert_token_value

import igraph as ig
from igraph import Graph

app_debug = write_log("pipeline/logging", "debug", "dynamic_graph")


class DynGraphIgraph:
    def __init__(
        self,
        node_types: Optional[Iterable[str]] = None,
        flatten: Iterable[str] | str = [],
        directed: bool = False,
        smooth: Optional[str] = None,
        meta_path: Optional[Iterable] = None,
        # —— 新增配置 ——
        ngram_config: Optional[Dict] = {'token_n':[2,3], 'char_n':[], 'skip':0},   # 如 Optional[Dict[str, Dict]] = {'tt': {'token_n':[2,3], 'char_n':[], 'skip':0}}, none
        rare_bias: Optional[str] = 'idf',                 # None|'idf'|'degree'
        rare_alpha: float = 1.0,
        undirected_weighted: bool = True,                 # 无向图也按权重采样
        cache_ngram_size: int = 200_000,
    ):
        self.graph: Graph = ig.Graph(directed=directed)
        self.node_classes: Dict[str, int] = {}     # prefix -> class ID（如 {'idx':5, 'cid':0}）
        self.node_is_numeric: Dict[str, bool] = {} # prefix -> 是否数值类型
        self.to_flatten = flatten if flatten != 'no' else []
        self.meta_path = []
        self.dyn_roots = None
        self.samplers = []              # 与顶点 index 对齐的采样器缓存

        # 全局图属性
        self.graph['num_ids'] = -1
        self.graph['smooth'] = smooth        # 边权平滑策略：'IDF'|'ICF'|'log'|None
        self.graph['weighted'] = directed    # 历史字段，保留
        self.graph['words_count'] = 0

        # 新增：O(1) 名称查 index
        self.name2idx: Dict[str, int] = {}

        # n-gram 配置
        self.ngram_config = ngram_config or {'token_n': [2, 3], 'char_n': [], 'skip': 0}
            # 针对文本 token 节点（tt）生成 token n-gram；可按需扩展其他前缀
            
        
        self._cache_ngram_size = cache_ngram_size

        # 稀有偏置与采样
        self.rare_bias = rare_bias
        self.rare_alpha = float(rare_alpha)
        self.undirected_weighted = undirected_weighted

        if (not meta_path) and node_types :
            self._extract_node_types(node_types)
            self._check_flatten()

        # 设置 meta_path / meta_link / meta_node
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
        
        # dyn_roots
        if meta_path:
            self.dyn_roots = {}
            if isinstance(meta_path[0], list):
                for path in meta_path:
                    self.dyn_roots[path[0]] = set()
            else:
                self.dyn_roots[meta_path[0]] = set()
        else:
            self.dyn_roots = set()
    
    # ------------------------
    # meta_information, check during building
    # ------------------------
    def _extract_node_types(self, node_types: Iterable[str]):
        for node_type in node_types:
            class_info, name = node_type.split('__')
            rwclass = int(class_info[0])
            num_flag = class_info[1]
            self.node_classes[name] = rwclass
            self.node_is_numeric[name] = True if num_flag == '#' else False

    def _check_flatten(self):
        if self.to_flatten and self.to_flatten != 'no':
            for prefix in self.to_flatten:
                if prefix not in self.node_classes:
                    raise ValueError(f'Unknown flatten type: {prefix}')

    def _get_node_type(self, node_name: str) -> str:
        for pre in self.node_classes:
            if node_name.startswith(pre + '__'):
                return pre
        raise ValueError(f'Node {node_name} does not match any known prefix.')

    # ------------------------
    # clean attributes, for export 
    # ------------------------
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
    
    # ------------------------
    # load existing graph 
    # ------------------------
    def load_graph(self, graph_file: str):
        self.graph = Graph.Read_GraphML(graph_file)
        self.graph['num_ids'] = int(float(self.graph['num_ids']))
        self.graph['words_count'] = int(float(self.graph['words_count']))
        self._extend_sampler(self.graph.vcount())
        # 重建 name2idx
        self.rebuild_index_map()
        # rebuild vertex attributes
        if not self.meta_path:
            for v in self.graph.vs:
                v['node_class'] = self._update_node_class(v["type"]) if 'type' in v.attributes() else {
                    'isfirst': False, 'isroot': False, 'isappear': False
                }
        for v in self.graph.vs:
            self._update_neighbors(v.index)
            v["test_pretraining"] = True
            v['test_neighbors_freq'] = {}

    # ------------------------
    # For Security: Reconstructable name→index mapping (used after deletion/loading)
    # ------------------------
    def rebuild_index_map(self):
        self.name2idx.clear()
        for v in self.graph.vs:
            self.name2idx[v['name']] = int(v.index)
        # 采样器数组长度与顶点数对齐
        self._extend_sampler(self.graph.vcount())

    # ------------------------
    # some simple getters
    # ------------------------
    def get_smooth_method(self):
        return self.graph['smooth']

    def get_directed_info(self) -> bool:
        return self.graph.is_directed()

    def set_id_nums(self, id_num):
        self.graph['num_ids'] = id_num

    def accum_id_nums(self):
        self.graph['num_ids'] += 1
    
    def get_id_nums(self):
        return self.graph['num_ids']
    
    def get_smooth_method(self):
        return self.graph['smooth']
        
    def get_graph(self):
        return self.graph

    def get_neighbors(self, node_name):
        if node_name not in self.node_idx_map:
            return []
        idx = self.graph.vs.find(name=node_name)
        neighbors = self.graph.neighbors(idx, mode='ALL')
        return [self.graph.vs[n]['name'] for n in neighbors]      
    
    def show_summary(self):
        print(self.graph.summary())

    def get_node_attribute(self, node_name, attr):
        return self.graph.vs.find(name=node_name)[attr]

    def set_node_attribute(self, node_name, attr, value):
        self.graph.vs.find(name=node_name)[attr] = value

    def get_record(self, id_index):
        """Function used to get instances of one id

        :param id_index: id_num (int)
        :return  list: all instences in one record
        """
        neighbors = self.graph.neighbors(id_index, mode='OUT')
        return [self.graph.vs[n]['name'] for n in neighbors if self.graph.vs[n]['type'] != "st"]

    def get_sampler(self, index: int):
        if index < len(self.samplers):
            return self.samplers[index]
        return None
    
    # ------------------------
    # functions for updating graphs
    # ------------------------
    def build_relation(self, df):     
        # Collect batch operation data
        vertices_to_add = {}  # {name: prefix} 
        vertex_freq_updates = defaultdict(int) # {name: freq}
        edges_to_add = set()  # {(name1, name2)}
        
        # replace attributes of instance by local variables
        meta_node = self.meta_node if self.meta_path else None
        meta_link = self.meta_link if self.meta_path else None
        to_flatten = set(self.to_flatten) if self.to_flatten else set()
        
        if self.meta_path:
            # ===== Meta Path Mode =====
            # 预过滤相关列. 这部分要做吗 ?????
            relevant_cols = [col for col in df.columns if col in meta_node]
            col_to_idx = {col: idx for idx, col in enumerate(relevant_cols)}
            
            # Convert to a list of tuples
            data_tuples = list(df[relevant_cols].itertuples(index=False, name=None))
            
            for row_data in tqdm(data_tuples, total=len(df), desc="# Building/Updating graph"):
                values = {}
                
                for col in relevant_cols:
                    col_idx = col_to_idx[col]
                    cell_value = row_data[col_idx]
                    
                    if col != 'rid':
                        token_list, _ = convert_token_value(cell_value)
                    else:
                        token_list = [cell_value]
                    
                    values[col] = []
                    
                    if token_list is not None and len(token_list) > 0:
                        for el in token_list:
                            # Collection node (original value and flattened token)
                            node_names = self._collect_instance_nodes(
                                el, col, vertices_to_add, vertex_freq_updates, to_flatten
                            )
                            values[col].extend(node_names)
                    else:
                        node_name = f"{col}__nan"
                        if node_name not in self.name2idx:
                            vertices_to_add[node_name] = col
                        vertex_freq_updates[node_name] += 1
                        values[col].append(node_name)
                
                # collect edges
                for a, b in meta_link:
                    if a in values and b in values:
                        for v1 in values[a]:
                            for v2 in values[b]:
                                edges_to_add.add((v1, v2))
        
        else:
            # ===== normal mode =====
            # Prepare index mapping
            data_cols = [col for col in df.columns if col != 'rid']
            all_cols = ['rid'] + data_cols
            col_to_idx = {col: idx for idx, col in enumerate(all_cols)}

            data_tuples = list(df[all_cols].itertuples(index=False, name=None))
            
            for row_data in tqdm(data_tuples, total=len(df), desc="# Building/Updating graph"):
                # collect rid vertices
                rid_node = str(row_data[0])                
                if rid_node not in self.name2idx:
                    vertices_to_add[rid_node] = 'idx'
                vertex_freq_updates[rid_node] += 1
                
                # Iterate through the data columns
                for col_name in data_cols:
                    col_idx = col_to_idx[col_name]
                    og_value = row_data[col_idx]
                    
                    if pd.isna(og_value):
                        continue
                    
                    try:
                        cid_node = f'cid__{col_name}'
                        if cid_node not in self.name2idx:
                            vertices_to_add[cid_node] = 'cid'
                        vertex_freq_updates[cid_node] += 1
                        
                        token_list, is_numeric = convert_token_value(og_value)
                        
                        if token_list is not None:
                            node_prefix = "tn" if is_numeric else "tt"
                            
                            # for token vertices
                            for el in token_list:
                                instance_names = self._collect_instance_nodes(
                                    el, node_prefix, vertices_to_add, vertex_freq_updates, to_flatten
                                )
                                
                                # for edges
                                for inst_name in instance_names:
                                    edges_to_add.add((inst_name, cid_node))
                                    edges_to_add.add((inst_name, rid_node))
                            
                            # for n_gram vertices
                            if not is_numeric:
                                ngram_names = self._collect_ngram_nodes(
                                    token_list, vertices_to_add, vertex_freq_updates
                                )
                                # for edges
                                for ng_name in ngram_names:
                                    edges_to_add.add((ng_name, cid_node))
                                    edges_to_add.add((ng_name, rid_node))
                    
                    except KeyError:
                        continue
        
        # ===== Add Vertices en Batch =====
        print(f"\nAdding {len(vertices_to_add)} vertices (new or updated)...")
        new_vertex_map = self._batch_add_vertices(vertices_to_add, vertex_freq_updates)
        
        # ===== Add Edges en Batch  =====
        print(f"Adding {len(edges_to_add)} edges...")
        self._batch_add_edges(edges_to_add, new_vertex_map)
        
        # ===== Update sampler en Batch  =====
        print("Updating samplers...")
        self._batch_update_samplers(new_vertex_map)


    def _collect_instance_nodes(self, ins_value: str, prefix: str, 
                                vertices_dict: dict, freq_dict: defaultdict,
                                to_flatten_set: set) -> List[str]:
        """
        Collect all node names associated with an instance value (original value + flattened tokens if required)
        Do not insert; collect only
        """
        node_names = []
        name2idx = self.name2idx
        
        # original value as node name
        main_name = f'tt__{ins_value}'
        if main_name not in name2idx and main_name not in vertices_dict:
            vertices_dict[main_name] = prefix
        freq_dict[main_name] += 1
        node_names.append(main_name)
        
        # flattens if required, get tokens
        if prefix in to_flatten_set:
            for val, pfx in self._flatten(ins_value, prefix_for_token=prefix):
                if not val:
                    continue
                val_name = f'tt__{val}'
                if val_name not in name2idx and val_name not in vertices_dict:
                    vertices_dict[val_name] = pfx
                freq_dict[val_name] += 1
                node_names.append(val_name)
        
        return node_names

    def _flatten(self, og_value: str, prefix_for_token: str) -> List[Tuple[str, str]]:
        """
        flatten and get tokens 
        return: [(token_str, prefix)]
        """
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
        return res
    
    @lru_cache(maxsize=200_000)
    def _tokenize_cached(self, s: str) -> Tuple[str, ...]:
        # flattec by "_"
        return tuple([t for t in str(s).strip().split('_') if t])
    
    def _collect_ngram_nodes(self, token_list: List[str], 
                            vertices_dict: dict, freq_dict: defaultdict) -> List[str]:
        """
        collect node names for n_gram, prefixe: "st"
        """
        cfg = self.ngram_config
        if not cfg:
            return []
        
        token_ns = cfg.get('token_n', [])
        if not token_ns:
            return []
        
        skip = cfg.get('skip', 0)
        node_names = []
        name2idx = self.name2idx
        
        for el in token_list:
            toks = self._tokenize_cached(el)
            for ng in self._gen_ngrams(toks, token_ns, skip=skip):
                name = f"ng::{len(ng)}::" + "␟".join(ng)
                if name not in name2idx and name not in vertices_dict:
                    vertices_dict[name] = 'st'
                freq_dict[name] += 1
                node_names.append(name)
        return node_names

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
            # 简化的 k-skip-n-gram 生成
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

    def _batch_add_vertices(self, vertices_dict: dict, freq_dict: defaultdict) -> Dict[str, int]:
        """
        Add vertices en batch
        Return:map "name->index" for all nodes
        """
        name_to_index = {}
        name2idx = self.name2idx
        
        # Separate existing and newly added nodes
        existing = []
        new_names = []
        new_prefixes = []
        
        for name, prefix in vertices_dict.items():
            if name in name2idx:
                existing.append((name, name2idx[name]))
            else:
                new_names.append(name)
                new_prefixes.append(prefix)
        
        # existing nodes:  update Frequency
        vs = self.graph.vs
        for name, idx in existing:
            vs[idx]["frequency_in_graph"] = int(vs[idx]["frequency_in_graph"]) + freq_dict[name]
            name_to_index[name] = idx
        
        # new nodes: add en batch
        if new_names:
            start_idx = self.graph.vcount()
            self.graph.add_vertices(len(new_names))
            
            # Prefetch frequently used attributes
            meta_path = self.meta_path
            node_is_numeric = self.node_is_numeric
            dyn_roots = self.dyn_roots
            words_count = int(self.graph['words_count'])
            
            # set attributes en batch
            if meta_path:
                for i, (name, prefix) in enumerate(zip(new_names, new_prefixes)):
                    idx = start_idx + i
                    freq = freq_dict[name]
                    
                    vs[idx]['name'] = name
                    vs[idx]['type'] = prefix
                    vs[idx]['appearing_frequency'] = 0
                    vs[idx]['frequency_in_graph'] = freq
                    vs[idx]['test_pretraining'] = False
                    vs[idx]['test_neighbors_freq'] = {}
                    
                    name_to_index[name] = idx
                    name2idx[name] = idx
                    words_count += freq
                    
                    if prefix in dyn_roots.keys():
                        dyn_roots[prefix].add(idx)
            else:
                for i, (name, prefix) in enumerate(zip(new_names, new_prefixes)):
                    idx = start_idx + i
                    freq = freq_dict[name]
                    node_class = self._update_node_class(prefix)
                    
                    vs[idx]['name'] = name
                    vs[idx]['type'] = prefix
                    vs[idx]['numeric'] = node_is_numeric.get(prefix, False)
                    vs[idx]['node_class'] = node_class
                    vs[idx]['appearing_frequency'] = 0
                    vs[idx]['frequency_in_graph'] = freq
                    vs[idx]['test_pretraining'] = False
                    vs[idx]['test_neighbors_freq'] = {}
                    
                    name_to_index[name] = idx
                    name2idx[name] = idx
                    words_count += freq
                    
                    if node_class.get('isroot', False):
                        dyn_roots.add(idx)
            
            self.graph['words_count'] = words_count
        
        return name_to_index

    def _update_node_class(self, prefix: str) -> Dict[str, bool]:
        node_class_bin = '{:03b}'.format(self.node_classes[prefix])
        return {
            'isfirst': bool(int(node_class_bin[0])),
            'isroot': bool(int(node_class_bin[1])),
            'isappear': bool(int(node_class_bin[2]))
        }
    
    def _batch_add_edges(self, edges_set: set, name_to_index: Dict[str, int]):
        """
        Add edges en batch
        """
        name2idx = self.name2idx
        graph = self.graph
        
        # Analyze all index_pairs
        idx_pairs = []
        for name1, name2 in edges_set:
            idx1 = name_to_index.get(name1) or name2idx.get(name1)
            idx2 = name_to_index.get(name2) or name2idx.get(name2)
            
            if idx1 is not None and idx2 is not None:
                idx_pairs.append((idx1, idx2))
        
        if not idx_pairs:
            return
        
        # check and add edges
        edges_to_add = []
        for idx1, idx2 in idx_pairs:
            if not graph.are_adjacent(idx1, idx2):
                edges_to_add.append((idx1, idx2))
        
        if edges_to_add:
            graph.add_edges(edges_to_add)


    def _batch_update_samplers(self, new_vertex_map: Dict[str, int]):
        """
        Update samplers en batch
        """
        # Expand the sampler array
        self._extend_sampler(self.graph.vcount())
        
        # Collect all nodes that require updating
        affected = set(new_vertex_map.values())
        neighbor_updated = set()
        
        # Update the directly affected nodes
        for idx in affected:
            self._update_neighbors(idx)
            neighbor_updated.add(idx)
        
        # Update their neighbors.
        graph = self.graph
        for idx in list(affected):
            neighbors = graph.neighbors(idx, mode='OUT')
            for neigh in neighbors:
                if neigh not in neighbor_updated:
                    self._update_neighbors(neigh)
                    neighbor_updated.add(neigh)
    
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
            use_w = (self.undirected_weighted or graph.is_directed())

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
            use_w = self.undirected_weighted or graph.is_directed()
            if not use_w:
                self.samplers[index] = NodeSampler(neighbors=neighbors, weighted=False, threshold=1000)
            else:
                weights = [self._edge_weight_for_sampling(index, el) for el in neighbors]
                self.samplers[index] = NodeSampler(neighbors=neighbors, weighted=True, threshold=50, weights=weights)

            if 'node_class' in v.attributes() and not v["node_class"].get("isfirst", True):
                self.samplers[index].update_firstnode_list([vs[idx]["node_class"].get("isfirst", False) for idx in neighbors])

    def _edge_weight_for_sampling(self, v_from: int, v_to: int) -> float:
        try:
            eid = self.graph.get_eid(v_from, v_to)
            base = float(self.graph.es[eid]['weight']) if 'weight' in self.graph.es[eid].attributes() else 1.0
        except Exception:
            base = 1.0
        rare = self._rare_weight_of_vertex(v_to)
        return max(1e-12, base * rare)

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
    
    def _add_edge_weight(self, to_index: int):
        # calculate weight
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

    # -----------
    # tools
    # -----------
    def get_vertex_index(self, name: str) -> Optional[int]:
        return self.name2idx.get(name) 

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
        node_types=["3#__idx", "0#__cid", "0#__tt", "0#__tn", "0#__ng", "5$__st"],
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
    print("Index of 'tt__honey_basil_amber:", g.get_vertex_index("tt__honey_basil_amber"))
    print("Index of 'ng::3::Honey␟Basil␟Amber':", g.get_vertex_index("ng::3::Honey␟Basil␟Amber"))
    print("Index of 'ng::3::honey␟basil␟amber':", g.get_vertex_index("ng::3::honey␟basil␟amber"))

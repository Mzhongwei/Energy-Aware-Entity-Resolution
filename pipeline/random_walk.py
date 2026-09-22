import traceback
import time

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, iterable=None, **_kwargs): self.iterable = iterable
        def __iter__(self): return iter(self.iterable)
        def update(self, _count): pass
        def close(self): pass

from utils.write_log import write_log


app_debug = write_log("logs", "debug", "random_walk")
walk_info = write_log("logs", "walks", "random_walk_record")


def _random_walk_config(configuration):
    if not isinstance(configuration, dict):
        return {}
    walk_cfg = configuration.get("random_walk")
    if isinstance(walk_cfg, dict):
        return walk_cfg
    legacy_cfg = configuration.get("walks")
    if isinstance(legacy_cfg, dict):
        return legacy_cfg
    return {}


def _graph_meta_path(configuration):
    if not isinstance(configuration, dict):
        return []
    if "meta_path" in configuration:
        return configuration.get("meta_path", [])
    graph_cfg = configuration.get("graph_construction")
    if isinstance(graph_cfg, dict) and "meta_path" in graph_cfg:
        return graph_cfg.get("meta_path", [])
    legacy_cfg = configuration.get("graph")
    if isinstance(legacy_cfg, dict):
        return legacy_cfg.get("meta_path", [])
    return []


def _format_walk(graph, walk_ids, sequence_format):
    if sequence_format == "id":
        return [str(int(node_id)) for node_id in walk_ids]
    if sequence_format == "text":
        return [graph.get_node_name(int(node_id)) for node_id in walk_ids]
    raise ValueError("random_walk.sequence_format must be 'id' or 'text'")


def _sample_neighbor(graph, node_id):
    neighbors = graph.neighbors(node_id)
    if len(neighbors) == 0:
        return None
    return int(neighbors[np.random.randint(0, len(neighbors))])


class RamdomRow:
    def __init__(self, graph, row_id_index, sentence_len, sequence_format="text"):
        self.walk = []
        walk_ids = []
        while len(walk_ids) < sentence_len:
            node_index = _sample_neighbor(graph, row_id_index)
            if node_index is None:
                raise ValueError("No neighbors")
            walk_ids.append(node_index)
            if len(walk_ids) < sentence_len:
                walk_ids.append(int(row_id_index))
        self.walk = _format_walk(graph, walk_ids, sequence_format)

    def get_walk(self):
        return self.walk

    def get_reversed_walk(self):
        return self.walk[::-1]


class RandomWalk:
    def __init__(self, graph, starting_node_index, sentence_len, backtrack,
                 update_stats=True, sequence_format="text"):
        walk_ids = []
        starting_node_index = int(starting_node_index)
        starting_node_name = graph.get_node_name(starting_node_index)
        if graph.is_first(starting_node_index):
            walk_ids = [starting_node_index]
        else:
            try:
                candidates = [int(node_id) for node_id in graph.neighbors(starting_node_index)
                              if graph.is_first(int(node_id))]
                if candidates:
                    first_node_indice = candidates[np.random.randint(0, len(candidates))]
                    walk_ids = [first_node_indice, starting_node_index]
                else:
                    raise ValueError("The first node of the sentence could not be found. Please check your node_types settings.")
            except Exception:
                print(f"The first node of the sentence could not be found. Please check node {starting_node_name}.")
                app_debug.error(
                    f"The first node of the sentence could not be found. Please check node {starting_node_name}, index {starting_node_index}."
                )

        if not walk_ids:
            self.walk = []
            return

        current_node_indice = starting_node_index
        sentence_step = len(walk_ids)

        while sentence_step < sentence_len:
            previous_node_index = current_node_indice
            current_node_indice = _sample_neighbor(graph, previous_node_index)
            if current_node_indice is None:
                raise ValueError("No neighbors")

            if not backtrack and current_node_indice == walk_ids[-1]:
                continue
            if not graph.is_appear(current_node_indice):
                continue

            walk_ids.append(current_node_indice)
            sentence_step += 1
        self.walk = _format_walk(graph, walk_ids, sequence_format)

    def get_walk(self):
        return self.walk

    def get_reversed_walk(self):
        return self.walk[::-1]


class RandomWalk_MetaPath:
    def __init__(self, graph, starting_node_index, sentence_len, meta_path):
        i_graph = graph.get_graph()
        self.walk = []
        current_node = i_graph.vs[starting_node_index]
        current_node_indice = starting_node_index
        self.walk.append(current_node["name"])
        meta_index = 1

        sentence_step = len(self.walk)
        while sentence_step < sentence_len:
            try:
                next_type = meta_path[meta_index % len(meta_path)]
                sampler = graph.get_sampler(current_node_indice)
                next_node_indice = sampler[next_type].sample()
                next_node = i_graph.vs[next_node_indice]
                self.walk.append(next_node["name"])
                current_node = next_node
                current_node_indice = next_node_indice
            except Exception:
                print(f"NO neighbors found for node: {current_node_indice}, name: {current_node['name']}. Looking for node type: {next_type}")
                app_debug.error(
                    f"NO neighbors found for node: {current_node}, name: {current_node['name']}. Looking for node type: {next_type}"
                )
                break
            sentence_step += 1
            meta_index += 1

    def get_walk(self):
        return self.walk

    def get_reversed_walk(self):
        return self.walk[::-1]


def start_walk_multiscale(roots_index, graph, walks_number, walk_length, write_walks, walk_rules, update_stats=False):
    sentences = []
    if roots_index == 0 or roots_index is None:
        return
    wl = [15, 30, 60]
    dis = [2, 2, 1]
    for wl_i in range(len(wl)):
        wn = int(walks_number * (dis[wl_i] / sum(dis)))
        pbar = tqdm(desc="# Sentence generation progress: ", total=len(roots_index) * wn)
        for root in roots_index:
            walks = []
            for _r in range(wn):
                try:
                    if isinstance(walk_rules, bool):
                        w = RandomWalk(graph, root, wl[wl_i], walk_rules, update_stats=update_stats)
                    else:
                        w = RandomWalk_MetaPath(graph, root, wl[wl_i], walk_rules)
                except Exception as e:
                    print("node: ", _r)
                    print(e)
                    print(traceback.print_exc())
                    break

                if w.get_walk() != []:
                    walks.append(w.get_walk())
                else:
                    raise ValueError("random walk anormal")

            if write_walks and len(walks) > 0:
                ws = [" ".join(_) for _ in walks]
                s = "\n".join(ws) + "\n"
                walk_info.info(s)
            sentences += walks
            pbar.update(wn)
        pbar.close()
    return sentences


def start_walk(roots_index, graph, walks_number, walk_length, write_walks, walk_rules, row,
               update_stats=False, sequence_format="text"):
    sentences = []
    if roots_index == 0 or roots_index is None:
        return

    pbar = tqdm(desc="# Sentence generation progress: ", total=len(roots_index) * walks_number)

    for root in roots_index:
        walks = []

        root_id = graph.get_node_name(root)
        if row:
            walks_number_basic = int(walks_number * 0.2)
            walks_number_row = int(walks_number - walks_number_basic)
            if root_id.startswith("idx"):
                for _r in range(walks_number_row):
                    w = RamdomRow(graph, root, walk_length, sequence_format)
                    if w.get_walk() != []:
                        walks.append(w.get_walk())
                        # app_debug.info(f"walks of token {w.get_walk()}")
            else:
                for _r in range(walks_number_basic):
                    w = RandomWalk(graph, root, walk_length, walk_rules, update_stats=update_stats,
                                   sequence_format=sequence_format)
                    if w.get_walk() != []:
                        walks.append(w.get_walk())
                        # app_debug.info(f"walks of token {w.get_walk()}")
        else:
            for _r in range(walks_number):
                try:
                    if isinstance(walk_rules, bool):
                        w = RandomWalk(graph, root, walk_length, walk_rules, update_stats=update_stats,
                                       sequence_format=sequence_format)
                    else:
                        w = RandomWalk_MetaPath(graph, root, walk_length, walk_rules)
                except Exception as e:
                    print("node: ", _r)
                    print(e)
                    print(traceback.print_exc())
                    break

                if w.get_walk() != []:
                    walks.append(w.get_walk())
                else:
                    if not row:
                        raise ValueError("random walk anormal")

        if write_walks and len(walks) > 0:
            ws = [" ".join(_) for _ in walks]
            s = "\n".join(ws) + "\n"
            walk_info.info(s)
        sentences += walks
        pbar.update(walks_number)
    pbar.close()
    return sentences


def dynrandom_walks_generation(configuration, graph):
    """Generate one window of walks through the backend-neutral reader interface."""
    started = time.perf_counter()
    try:
        return _generate_walks(configuration, graph)
    finally:
        if hasattr(graph, "clear_samplers"):
            graph.clear_samplers()
        print(f"[random-walk] seconds={time.perf_counter() - started:.6f} num_nodes={graph.vertex_count()}")


def _generate_walks(configuration, graph):
    walk_cfg = _random_walk_config(configuration)
    walk_nums = int(walk_cfg.get("walks_number", 0))
    walk_length = int(walk_cfg.get("walk_length", 60))
    backtrack = walk_cfg.get("backtrack", False)
    update_stats = bool(walk_cfg.get("rw_stat", False))
    meta_path = _graph_meta_path(configuration)
    write_walks = walk_cfg.get("write_walks", True)
    sequence_format = walk_cfg.get("sequence_format", "text")
    if sequence_format not in {"id", "text"}:
        raise ValueError("random_walk.sequence_format must be 'id' or 'text'")
    compact = configuration.get("graph_construction", {}).get("backend") == "compact_adjacency"
    if compact and meta_path:
        raise NotImplementedError("compact_adjacency backend V1 does not support meta_path")
    if compact and update_stats:
        raise NotImplementedError("compact_adjacency backend V1 does not support rw_stat")

    sentences = []
    if walk_nums > 0:
        if not meta_path:
            roots_index = graph.dyn_roots
            sentences = start_walk(roots_index, graph, walk_nums, walk_length, write_walks,
                                   backtrack, row=False, update_stats=update_stats,
                                   sequence_format=sequence_format)
            graph.dyn_roots.clear()
        else:
            if isinstance(meta_path, list):
                if isinstance(meta_path[0], list):
                    for path in meta_path:
                        roots_index = graph.dyn_roots[path[0]]
                        sentences += start_walk(roots_index, graph, walk_nums, walk_length, write_walks, path, row=False, update_stats=update_stats)
                        graph.dyn_roots[path[0]].clear()
                else:
                    roots_index = graph.dyn_roots[meta_path[0]]
                    sentences = start_walk(roots_index, graph, walk_nums, walk_length, write_walks, meta_path, row=False, update_stats=update_stats)
                    graph.dyn_roots[meta_path[0]].clear()
    return sentences


__all__ = [
    "RandomWalk",
    "RandomWalk_MetaPath",
    "RamdomRow",
    "dynrandom_walks_generation",
    "start_walk",
    "start_walk_multiscale",
]

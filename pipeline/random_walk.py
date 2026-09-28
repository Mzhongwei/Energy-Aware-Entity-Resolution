import json
import math
import os
import shutil
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


def resolve_walk_process_count(configuration, environ=None):
    """Resolve worker count and cap it to the Pod CPU limit when Kubernetes exposes one."""
    walk_cfg = _random_walk_config(configuration)
    requested = int(walk_cfg.get("processes", 1))
    if requested < 1:
        raise ValueError("random_walk.processes must be at least 1")

    environment = os.environ if environ is None else environ
    raw_millicores = str(environment.get("EAER_CPU_LIMIT_MILLICORES", "")).strip()
    if not raw_millicores:
        return requested
    try:
        millicores = float(raw_millicores.removesuffix("m"))
    except ValueError as exc:
        raise ValueError(
            "EAER_CPU_LIMIT_MILLICORES must be a numeric Kubernetes CPU quantity"
        ) from exc
    if not math.isfinite(millicores) or millicores <= 0:
        raise ValueError("EAER_CPU_LIMIT_MILLICORES must be greater than zero")

    cpu_workers = max(1, math.floor(millicores / 1000.0))
    return min(requested, cpu_workers)


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


def _format_walk(graph, walk_ids):
    return [graph.get_node_name(int(node_id)) for node_id in walk_ids]


def _cached_neighbors(graph, node_id, cache):
    """Return node_id's neighbors, querying the graph only on the first lookup.

    `cache` is a plain dict shared by every walk inside one dynrandom_walks_generation()
    call (one batch/window). A node visited by many different walks in that batch -- the
    same root walked `walks_number` times, or a popular hub node reached from several
    roots -- is fetched from the graph (igraph's C call, or the compact/CSR array slice)
    only once; every later lookup in the same batch is a dict hit. The cache is never
    stored on `graph` or returned to the caller, so it is released for garbage collection
    as soon as the batch's call returns -- no explicit clear step is needed.
    """
    node_id = int(node_id)
    if node_id not in cache:
        cache[node_id] = graph.neighbors(node_id)
    return cache[node_id]


def _sample_neighbor(graph, node_id, cache, sampling_method="uniform"):
    if sampling_method == "legacy_sampler" and hasattr(graph, "get_sampler"):
        sampler = graph.get_sampler(int(node_id))
        return None if sampler is None else sampler.sample()

    neighbors = _cached_neighbors(graph, node_id, cache)
    if len(neighbors) == 0:
        return None
    if sampling_method == "legacy_sampler" and len(neighbors) <= 1000:
        # Match NodeSampler's unweighted simple-s RNG path.
        return int(np.random.choice(neighbors))
    return int(neighbors[np.random.randint(0, len(neighbors))])


class RandomWalk:
    def __init__(self, graph, starting_node_index, sentence_len, backtrack,
                 update_stats=True, neighbor_cache=None, sampling_method="uniform"):
        cache = {} if neighbor_cache is None else neighbor_cache
        walk_ids = []
        starting_node_index = int(starting_node_index)
        starting_node_name = graph.get_node_name(starting_node_index)
        if graph.is_first(starting_node_index):
            walk_ids = [starting_node_index]
        else:
            try:
                if sampling_method == "legacy_sampler" and hasattr(graph, "get_sampler"):
                    sampler = graph.get_sampler(starting_node_index)
                    first_node_indice = sampler.sample_firstnode()
                else:
                    candidates = [int(node_id) for node_id in _cached_neighbors(graph, starting_node_index, cache)
                                  if graph.is_first(int(node_id))]
                    if not candidates:
                        raise ValueError("The first node of the sentence could not be found. Please check your node_types settings.")
                    if sampling_method == "legacy_sampler" and len(candidates) <= 1000:
                        first_node_indice = np.random.choice(candidates)
                    else:
                        first_node_indice = candidates[np.random.randint(0, len(candidates))]
                if first_node_indice is not None:
                    walk_ids = [int(first_node_indice), starting_node_index]
                else:
                    raise ValueError("The first node of the sentence could not be found. Please check your node_types settings.")
            except Exception as exc:
                raise ValueError(
                    "The first node of the sentence could not be found for "
                    f"node {starting_node_name}, index {starting_node_index}"
                ) from exc

        if not walk_ids:
            self.walk = []
            return

        current_node_indice = starting_node_index
        sentence_step = len(walk_ids)

        while sentence_step < sentence_len:
            previous_node_index = current_node_indice
            current_node_indice = _sample_neighbor(
                graph, previous_node_index, cache, sampling_method=sampling_method
            )
            if current_node_indice is None:
                raise ValueError("No neighbors")

            if not backtrack and current_node_indice == walk_ids[-1]:
                continue
            if not graph.is_appear(current_node_indice):
                continue

            walk_ids.append(current_node_indice)
            sentence_step += 1
        self.walk = _format_walk(graph, walk_ids)

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
                break
            sentence_step += 1
            meta_index += 1

    def get_walk(self):
        return self.walk

    def get_reversed_walk(self):
        return self.walk[::-1]

def start_walk(roots_index, graph, walks_number, walk_length, walk_rules,
               update_stats=False, neighbor_cache=None, sampling_method="uniform", seed=None):
    sentences = []
    if roots_index == 0 or roots_index is None:
        return

    # Shared across every root/walk in this call. A caller that generates the whole batch
    # through several start_walk() calls (dynrandom_walks_generation's meta-path branches)
    # passes its own dict in so those calls share one cache too; a bare call still works,
    # it just gets a private cache local to itself.
    cache = {} if neighbor_cache is None else neighbor_cache
    pbar = tqdm(desc="# Sentence generation progress: ", total=len(roots_index) * walks_number)

    for root in roots_index:
        # Derive each root's stream independently so changing the number (and boundaries)
        # of process shards does not change the generated walks or downstream F1.
        if seed is not None:
            root_seed = int(np.random.SeedSequence([int(seed), int(root)]).generate_state(1)[0])
            np.random.seed(root_seed)
        walks = []
    
        for _r in range(walks_number):
            try:
                if isinstance(walk_rules, bool):
                    w = RandomWalk(graph, root, walk_length, walk_rules, update_stats=update_stats,
                                    neighbor_cache=cache, sampling_method=sampling_method)
                else:
                    w = RandomWalk_MetaPath(graph, root, walk_length, walk_rules)
            except Exception as exc:
                raise RuntimeError(
                    f"Random walk failed for root {root} on walk {_r + 1}/{walks_number}"
                ) from exc

            if w.get_walk() != []:
                walks.append(w.get_walk())
            else:
                raise ValueError("random walk anormal")

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


def partition_walk_roots(roots, partition_count):
    """Split roots into balanced contiguous partitions without changing their order."""
    ordered_roots = list(roots or [])
    if not ordered_roots:
        return []
    partition_count = max(1, min(int(partition_count), len(ordered_roots)))
    width, remainder = divmod(len(ordered_roots), partition_count)
    partitions = []
    offset = 0
    for partition_index in range(partition_count):
        size = width + (1 if partition_index < remainder else 0)
        partitions.append(ordered_roots[offset:offset + size])
        offset += size
    return partitions


def generate_walk_shard(configuration, checkpoint_path, roots):
    """Process entry point for one root shard using its own reader and cache."""
    # Keep these imports local: this function is imported by spawned child processes and
    # graph_construction already contains the backend-specific loading policy.
    from pipeline.graph_construction import load_graph_reader, restore_dyn_roots

    # ProcessPoolExecutor uses fork on the current Linux deployment. Without reseeding,
    # every child would inherit the same NumPy RNG state and generate correlated walks.
    np.random.seed()
    graph = load_graph_reader(configuration, checkpoint_path)
    restore_dyn_roots(graph, roots)
    return dynrandom_walks_generation(configuration, graph) if graph.dyn_roots else []


def generate_walk_shard_file(configuration, checkpoint_path, roots, output_path):
    """Generate one shard, publish it atomically, and return only small metadata."""
    sequences = generate_walk_shard(configuration, checkpoint_path, roots)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temporary_path = f"{output_path}.{os.getpid()}.tmp"
    try:
        with open(temporary_path, "w", encoding="utf-8") as output:
            for walk in sequences:
                output.write(json.dumps(walk, ensure_ascii=False, separators=(",", ":")))
                output.write("\n")
        os.replace(temporary_path, output_path)
    except Exception:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
        raise
    return {"file": os.path.basename(output_path), "walk_count": len(sequences)}


def _load_complete_shard_manifest(shard_directory):
    manifest_path = os.path.join(shard_directory, "manifest.json")
    if not os.path.isfile(manifest_path):
        return None
    try:
        with open(manifest_path, encoding="utf-8") as source:
            manifest = json.load(source)
        if manifest.get("format") != "walk-shards-v1":
            return None
        if os.path.abspath(manifest.get("directory", "")) != os.path.abspath(shard_directory):
            return None
        counted_walks = 0
        for part in manifest.get("parts", []):
            filename = part["file"]
            if os.path.basename(filename) != filename:
                return None
            if not os.path.isfile(os.path.join(shard_directory, filename)):
                return None
            counted_walks += int(part["walk_count"])
        if counted_walks != int(manifest.get("walk_count", -1)):
            return None
        return manifest
    except (KeyError, OSError, TypeError, ValueError):
        return None


def parallel_walks_to_shards(
    configuration, checkpoint_path, roots, executor, process_count, shard_directory,
):
    """Write process results as JSONL shards and publish a manifest after the barrier."""
    partitions = partition_walk_roots(roots, process_count)
    if not partitions:
        return []

    # A worker may have crashed after publishing the external manifest but before acking
    # its graph checkpoint. Reuse a complete directory so an embedding reader is never
    # racing with deletion/recreation of already-published shard files.
    existing = _load_complete_shard_manifest(shard_directory)
    if existing is not None:
        return existing
    if os.path.isdir(shard_directory):
        shutil.rmtree(shard_directory)
    os.makedirs(shard_directory, exist_ok=True)

    futures = [
        executor.submit(
            generate_walk_shard_file,
            configuration,
            checkpoint_path,
            partition,
            os.path.join(shard_directory, f"part-{index:05d}.jsonl"),
        )
        for index, partition in enumerate(partitions)
    ]
    try:
        parts = [future.result() for future in futures]
    except Exception:
        for future in futures:
            future.cancel()
        raise

    manifest = {
        "format": "walk-shards-v1",
        "directory": os.path.abspath(shard_directory),
        "parts": parts,
        "walk_count": sum(part["walk_count"] for part in parts),
    }
    manifest_path = os.path.join(shard_directory, "manifest.json")
    temporary_path = f"{manifest_path}.{os.getpid()}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as output:
        json.dump(manifest, output, ensure_ascii=False)
    os.replace(temporary_path, manifest_path)
    return manifest


def _generate_walks(configuration, graph):
    walk_cfg = _random_walk_config(configuration)
    walk_nums = int(walk_cfg.get("walks_number", 10))
    walk_length = int(walk_cfg.get("walk_length", 30))
    backtrack = walk_cfg.get("backtrack", False)
    update_stats = bool(walk_cfg.get("rw_stat", False))
    sampling_method = str(walk_cfg.get("sampling_method", "uniform")).strip().lower()
    seed = walk_cfg.get("seed")
    if sampling_method not in {"uniform", "legacy_sampler"}:
        raise ValueError("random_walk.sampling_method must be uniform or legacy_sampler")
    meta_path = _graph_meta_path(configuration)
    compact = configuration.get("graph_construction", {}).get("backend") == "compact_adjacency"
    if compact and sampling_method == "legacy_sampler":
        raise NotImplementedError(
            "legacy_sampler requires the igraph backend because compact_adjacency "
            "does not retain legacy edge multiplicities"
        )
    if compact and meta_path:
        raise NotImplementedError("compact_adjacency backend V1 does not support meta_path")
    if compact and update_stats:
        raise NotImplementedError("compact_adjacency backend V1 does not support rw_stat")

    neighbor_cache = {}
    sentences = []

    if walk_nums > 0:
        if not meta_path:
            roots_index = graph.dyn_roots
            sentences = start_walk(roots_index, graph, walk_nums, walk_length,
                                   backtrack, update_stats=update_stats,
                                   neighbor_cache=neighbor_cache, sampling_method=sampling_method,
                                   seed=seed)
            graph.dyn_roots.clear()
        else:
            if isinstance(meta_path, list):
                if isinstance(meta_path[0], list):
                    for path in meta_path:
                        roots_index = graph.dyn_roots[path[0]]
                        sentences += start_walk(roots_index, graph, walk_nums, walk_length, path,
                                                update_stats=update_stats, neighbor_cache=neighbor_cache,
                                                sampling_method=sampling_method, seed=seed)
                        graph.dyn_roots[path[0]].clear()
                else:
                    roots_index = graph.dyn_roots[meta_path[0]]
                    sentences = start_walk(roots_index, graph, walk_nums, walk_length, meta_path,
                                           update_stats=update_stats, neighbor_cache=neighbor_cache,
                                           sampling_method=sampling_method, seed=seed)
                    graph.dyn_roots[meta_path[0]].clear()

    return sentences


__all__ = [
    "RandomWalk",
    "RandomWalk_MetaPath",
    "dynrandom_walks_generation",
    "generate_walk_shard",
    "generate_walk_shard_file",
    "parallel_walks_to_shards",
    "partition_walk_roots",
    "resolve_walk_process_count",
    "start_walk",
]

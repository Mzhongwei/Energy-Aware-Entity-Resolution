import json
import math
import os
import shutil
import time

import numpy as np

from models.graph_backend import ISAPPEAR, ISFIRST

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


WALK_SHARD_FORMATS = ("walk-shards-v1", "walk-shards-v2")
MIN_WALK_LENGTH = 2


def _as_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _validate_walk_parameters(walks_number=None, walk_length=None):
    """Validate shared walk dimensions before selecting a backend or execution path."""
    if walks_number is not None and int(walks_number) < 0:
        raise ValueError("random_walk.walks_number must be at least 0")
    if walk_length is not None and int(walk_length) < MIN_WALK_LENGTH:
        raise ValueError(
            f"random_walk.walk_length must be at least {MIN_WALK_LENGTH}"
        )


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


def _has_other_neighbor(graph, node_id, excluded_id, cache):
    return any(int(neighbor) != excluded_id for neighbor in _cached_neighbors(graph, node_id, cache))


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
                 update_stats=True, neighbor_cache=None, sampling_method="uniform",
                 first_candidates=None):
        sentence_len = int(sentence_len)
        _validate_walk_parameters(walk_length=sentence_len)
        cache = {} if neighbor_cache is None else neighbor_cache
        # Neighbors of a non-first root that may open its sentence. start_walk passes the list
        # computed by the root's first walk back in, so it is filtered once per root.
        self.first_candidates = first_candidates
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
                    if self.first_candidates is None:
                        self.first_candidates = [
                            int(node_id) for node_id in _cached_neighbors(graph, starting_node_index, cache)
                            if graph.is_first(int(node_id))
                        ]
                    candidates = self.first_candidates
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
        # The node the walker physically came from (a non-first root is entered from its first
        # node), which may differ from walk_ids[-1] when non-appearing nodes are skipped.
        previous_node_index = walk_ids[0] if len(walk_ids) == 2 else None
        sentence_step = len(walk_ids)
        stalled_steps = 0
        max_stalled_steps = 100 * sentence_len

        while sentence_step < sentence_len:
            next_node_index = _sample_neighbor(
                graph, current_node_indice, cache, sampling_method=sampling_method
            )
            if next_node_index is None:
                raise ValueError("No neighbors")
            stalled_steps += 1
            if stalled_steps > max_stalled_steps:
                raise RuntimeError(
                    "Random walk made no progress after "
                    f"{stalled_steps} sampling attempts from root {starting_node_name}; "
                    "check backtrack and node_types appear settings"
                )

            # Without backtracking, resample from the same node instead of stepping back,
            # unless going back is the only move available (otherwise the walk never ends).
            if (not backtrack and next_node_index == previous_node_index
                    and _has_other_neighbor(graph, current_node_indice, previous_node_index, cache)):
                continue
            previous_node_index, current_node_indice = current_node_indice, next_node_index
            if not graph.is_appear(current_node_indice):
                continue

            walk_ids.append(current_node_indice)
            sentence_step += 1
            stalled_steps = 0
        self.walk = _format_walk(graph, walk_ids)

    def get_walk(self):
        return self.walk

    def get_reversed_walk(self):
        return self.walk[::-1]


class RandomWalk_MetaPath:
    def __init__(self, graph, starting_node_index, sentence_len, meta_path):
        sentence_len = int(sentence_len)
        _validate_walk_parameters(walk_length=sentence_len)
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
    walks_number = int(walks_number)
    walk_length = int(walk_length)
    _validate_walk_parameters(walks_number, walk_length)
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
        first_candidates = None
        for _r in range(walks_number):
            try:
                if isinstance(walk_rules, bool):
                    w = RandomWalk(graph, root, walk_length, walk_rules, update_stats=update_stats,
                                    neighbor_cache=cache, sampling_method=sampling_method,
                                    first_candidates=first_candidates)
                    first_candidates = w.first_candidates
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


# ---------------------------------------------------------------------------------------
# Batched uniform walk kernel for CSR snapshots.
#
# All walkers of a root batch advance together with NumPy array operations instead of one
# Python call per step. Each walker draws from its own counter-based stream (SplitMix64 of
# seed, root, walk number and draw index), so the output does not depend on how roots are
# split across processes or batches -- the same invariance the per-root seeding above gives.
# ---------------------------------------------------------------------------------------
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_MIX1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX2 = np.uint64(0x94D049BB133111EB)
_TO_UNIT = 1.0 / float(1 << 53)
WALK_KERNEL_ROOT_BATCH = 8192


def _mix64(values):
    with np.errstate(over="ignore"):
        values = (values ^ (values >> np.uint64(30))) * _MIX1
        values = (values ^ (values >> np.uint64(27))) * _MIX2
    return values ^ (values >> np.uint64(31))


def _uniform(keys, counters):
    with np.errstate(over="ignore"):
        bits = _mix64(keys + counters * _GOLDEN)
    return (bits >> np.uint64(11)).astype(np.float64) * _TO_UNIT


def _pick(uniform, counts):
    # floor(u * n) can round up to n for huge n; clamp to the last slot.
    return np.minimum((uniform * counts).astype(np.int64), counts - 1)


def uses_walk_kernel(configuration):
    """Whether this configuration walks CSR snapshots with the batched kernel."""
    walk_cfg = _random_walk_config(configuration)
    graph_cfg = configuration.get("graph_construction", {}) if isinstance(configuration, dict) else {}
    return (graph_cfg.get("backend") == "compact_adjacency"
            and str(walk_cfg.get("sampling_method", "uniform")).strip().lower() == "uniform"
            and not _graph_meta_path(configuration))


def _first_candidates(indptr, indices, first_flags, roots):
    """Per-root candidate lists (CSR form) of first-typed neighbors for non-first roots."""
    starts, counts = indptr[roots], indptr[roots + 1] - indptr[roots]
    owners = np.repeat(np.arange(roots.size), counts)
    offsets = np.arange(owners.size) - np.repeat(np.cumsum(counts) - counts, counts)
    neighbors = indices[np.repeat(starts, counts) + offsets].astype(np.int64)
    keep = first_flags[neighbors]
    candidate_counts = np.bincount(owners[keep], minlength=roots.size)
    return neighbors[keep], np.cumsum(candidate_counts) - candidate_counts, candidate_counts


def _walk_batch(indptr, indices, flags, roots, walks_number, walk_length, backtrack, seed_key):
    """Walk ``walks_number`` times from every root in ``roots``; returns [walkers, length] node
    IDs in the snapshot's index dtype."""
    root_of = np.repeat(roots, walks_number)
    walk_of = np.tile(np.arange(walks_number, dtype=np.uint64), roots.size)
    with np.errstate(over="ignore"):
        keys = _mix64(_mix64(seed_key ^ _mix64(root_of.astype(np.uint64))) + walk_of * _GOLDEN)
    counters = np.zeros(root_of.size, dtype=np.uint64)
    walks = np.empty((root_of.size, walk_length), dtype=indices.dtype)
    filled = np.ones(root_of.size, dtype=np.int64)
    current = root_of.copy()
    previous = np.full(root_of.size, -1, dtype=np.int64)
    first_flags = (flags & ISFIRST).astype(bool)
    appear_flags = (flags & ISAPPEAR).astype(bool)

    # A non-first root opens its sentence with a uniformly chosen first-typed neighbor.
    walks[:, 0] = root_of
    opening = ~first_flags[root_of]
    if opening.any() and walk_length > 1:
        unique_roots, walker_root = np.unique(root_of[opening], return_inverse=True)
        values, starts, counts = _first_candidates(indptr, indices, first_flags, unique_roots)
        if (counts == 0).any():
            missing = int(unique_roots[np.flatnonzero(counts == 0)[0]])
            raise ValueError(f"The first node of the sentence could not be found for node index {missing}. "
                             "Please check your node_types settings.")
        walker_counts = counts[walker_root]
        chosen = values[starts[walker_root] + _pick(_uniform(keys[opening], counters[opening]), walker_counts)]
        counters[opening] += np.uint64(1)
        walks[opening, 0] = chosen
        walks[opening, 1] = root_of[opening]
        previous[opening] = chosen
        filled[opening] = 2

    stalled = 0
    active = np.flatnonzero(filled < walk_length)
    while active.size:
        nodes = current[active]
        starts = indptr[nodes]
        degrees = indptr[nodes + 1] - starts
        if (degrees == 0).any():
            raise ValueError(f"No neighbors for node index {int(nodes[np.flatnonzero(degrees == 0)[0]])}")
        draws = indices[starts + _pick(_uniform(keys[active], counters[active]), degrees)].astype(np.int64)
        counters[active] += np.uint64(1)
        if not backtrack:
            # Resample instead of stepping back, unless the previous node is the only neighbor
            # (CSR rows are deduplicated, so degree > 1 means another neighbor exists).
            accepted = (draws != previous[active]) | (degrees <= 1)
            active, draws = active[accepted], draws[accepted]
        previous[active] = current[active]
        current[active] = draws
        recorded = active[appear_flags[draws]]
        walks[recorded, filled[recorded]] = draws[appear_flags[draws]]
        filled[recorded] += 1
        # Walkers crossing only non-appearing nodes record nothing; guard against a graph
        # region where that can go on forever.
        stalled = 0 if recorded.size else stalled + 1
        if stalled > 100 * walk_length:
            raise RuntimeError("Random walk made no progress; check node_types appear flags")
        active = np.flatnonzero(filled < walk_length)
    return walks


def iter_walk_id_batches(graph, roots, walks_number, walk_length, backtrack, seed=None,
                         root_batch=WALK_KERNEL_ROOT_BATCH):
    """Yield node-ID walk matrices for sorted ``roots`` of a CSR reader, one root batch at a time."""
    walks_number = int(walks_number)
    walk_length = int(walk_length)
    _validate_walk_parameters(walks_number, walk_length)
    roots = np.asarray(sorted(int(root) for root in roots), dtype=np.int64)
    if seed is None:
        seed = int(np.random.randint(0, 2 ** 63, dtype=np.int64))
    seed_key = _mix64(np.uint64(int(seed) & 0xFFFFFFFFFFFFFFFF))
    indptr = np.asarray(graph.indptr, dtype=np.int64)
    indices = getattr(graph, "_indices", graph.indices)
    flags = np.frombuffer(graph.node_class_flags, dtype=np.uint8)
    for offset in range(0, roots.size, int(root_batch)):
        yield _walk_batch(indptr, indices, flags, roots[offset:offset + int(root_batch)],
                          walks_number, walk_length, bool(backtrack), seed_key)


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


def _write_walk_matrix_shard(configuration, checkpoint_path, roots, output_path):
    """Kernel shard: a uint32 [walks, walk_length] .npy of shard-local token positions plus a
    JSON vocabulary naming each position. The snapshot is deleted after the walk stage, so
    the shard carries its own names; each distinct name is decoded once per shard."""
    from pipeline.graph_construction import load_graph_reader, restore_dyn_roots

    np.random.seed()
    walk_cfg = _random_walk_config(configuration)
    walks_number = int(walk_cfg.get("walks_number", 10))
    walk_length = int(walk_cfg.get("walk_length", 30))
    _validate_walk_parameters(walks_number, walk_length)
    graph = load_graph_reader(configuration, checkpoint_path)
    restore_dyn_roots(graph, roots)
    walk_count = len(graph.dyn_roots) * max(walks_number, 0)
    vocab_path = output_path[:-len(".npy")] + ".vocab.json"
    temporary_path = f"{output_path}.{os.getpid()}.tmp.npy"
    temporary_vocab = f"{vocab_path}.{os.getpid()}.tmp"
    seconds = {"generate": 0.0, "names": 0.0, "write": 0.0}
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        started = time.perf_counter()
        matrix = np.lib.format.open_memmap(temporary_path, mode="w+", dtype=np.uint32,
                                           shape=(walk_count, walk_length))
        local_of = np.full(graph.vertex_count(), -1, dtype=np.int64)
        vocabulary = []
        row = 0
        seconds["write"] += time.perf_counter() - started
        batches = iter_walk_id_batches(graph, graph.dyn_roots, walks_number, walk_length,
                                       _as_bool(walk_cfg.get("backtrack", False)), walk_cfg.get("seed"))
        while True:
            started = time.perf_counter()
            batch = next(batches, None)
            if batch is None:
                break
            unseen = np.unique(batch)
            unseen = unseen[local_of[unseen] < 0]
            local_of[unseen] = np.arange(len(vocabulary), len(vocabulary) + unseen.size)
            vocabulary.extend(unseen.tolist())
            seconds["generate"] += time.perf_counter() - started
            started = time.perf_counter()
            matrix[row:row + batch.shape[0]] = local_of[batch]
            row += batch.shape[0]
            seconds["write"] += time.perf_counter() - started
        started = time.perf_counter()
        names = [graph.get_node_name(node_id) for node_id in vocabulary]
        seconds["names"] = time.perf_counter() - started
        started = time.perf_counter()
        matrix.flush()
        del matrix
        with open(temporary_vocab, "w", encoding="utf-8") as output:
            json.dump(names, output, ensure_ascii=False, separators=(",", ":"))
        os.replace(temporary_vocab, vocab_path)
        os.replace(temporary_path, output_path)
        seconds["write"] += time.perf_counter() - started
    except Exception:
        for path in (temporary_path, temporary_vocab):
            if os.path.exists(path):
                os.remove(path)
        raise
    print("[random-walk] shard=" + os.path.basename(output_path) + " walks=" + str(walk_count) + " "
          + " ".join(f"{name}_seconds={value:.6f}" for name, value in seconds.items()), flush=True)
    return {"file": os.path.basename(output_path), "vocab": os.path.basename(vocab_path),
            "walk_count": walk_count, "walk_length": walk_length, "seconds": seconds}


def generate_walk_shard_file(configuration, checkpoint_path, roots, output_path):
    """Generate one shard, publish it atomically, and return only small metadata."""
    if output_path.endswith(".npy"):
        return _write_walk_matrix_shard(configuration, checkpoint_path, roots, output_path)
    started = time.perf_counter()
    sequences = generate_walk_shard(configuration, checkpoint_path, roots)
    generated = time.perf_counter()
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
    seconds = {"generate": generated - started, "write": time.perf_counter() - generated}
    print("[random-walk] shard=" + os.path.basename(output_path) + " walks=" + str(len(sequences)) + " "
          + " ".join(f"{name}_seconds={value:.6f}" for name, value in seconds.items()), flush=True)
    return {"file": os.path.basename(output_path), "walk_count": len(sequences), "seconds": seconds}


def _load_complete_shard_manifest(shard_directory):
    manifest_path = os.path.join(shard_directory, "manifest.json")
    if not os.path.isfile(manifest_path):
        return None
    try:
        with open(manifest_path, encoding="utf-8") as source:
            manifest = json.load(source)
        if manifest.get("format") not in WALK_SHARD_FORMATS:
            return None
        if os.path.abspath(manifest.get("directory", "")) != os.path.abspath(shard_directory):
            return None
        counted_walks = 0
        for part in manifest.get("parts", []):
            for filename in (part["file"], part.get("vocab", part["file"])):
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
    """Write process results as shards and publish a manifest after the barrier.

    Kernel configurations write uint32 walk matrices (walk-shards-v2); the Python walker
    (igraph backend, legacy sampler, meta paths) writes JSONL (walk-shards-v1).
    """
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

    matrix_shards = uses_walk_kernel(configuration)
    extension = "npy" if matrix_shards else "jsonl"
    futures = [
        executor.submit(
            generate_walk_shard_file,
            configuration,
            checkpoint_path,
            partition,
            os.path.join(shard_directory, f"part-{index:05d}.{extension}"),
        )
        for index, partition in enumerate(partitions)
    ]
    try:
        parts = [future.result() for future in futures]
    except Exception:
        for future in futures:
            future.cancel()
        raise

    for part in parts:
        part.pop("seconds", None)
    manifest = {
        "format": "walk-shards-v2" if matrix_shards else "walk-shards-v1",
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
    _validate_walk_parameters(walk_nums, walk_length)
    backtrack = _as_bool(walk_cfg.get("backtrack", False))
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

    if walk_nums > 0 and uses_walk_kernel(configuration) and hasattr(graph, "indptr"):
        # Convert each root batch as it is produced, so only one ID matrix is alive at a time.
        generate_seconds = names_seconds = 0.0
        batches = iter_walk_id_batches(graph, graph.dyn_roots, walk_nums, walk_length, backtrack, seed)
        while True:
            started = time.perf_counter()
            batch = next(batches, None)
            generate_seconds += time.perf_counter() - started
            if batch is None:
                break
            started = time.perf_counter()
            names = graph.get_node_names(batch)
            sentences += [names[offset:offset + walk_length] for offset in range(0, len(names), walk_length)]
            names_seconds += time.perf_counter() - started
        print(f"[random-walk] generate_seconds={generate_seconds:.6f} "
              f"names_seconds={names_seconds:.6f} walks={len(sentences)}")
        graph.dyn_roots.clear()
    elif walk_nums > 0:
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
    "WALK_SHARD_FORMATS",
    "dynrandom_walks_generation",
    "iter_walk_id_batches",
    "generate_walk_shard",
    "generate_walk_shard_file",
    "parallel_walks_to_shards",
    "partition_walk_roots",
    "resolve_walk_process_count",
    "start_walk",
    "uses_walk_kernel",
]

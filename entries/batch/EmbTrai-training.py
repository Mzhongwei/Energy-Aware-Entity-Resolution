"""One long-running training stage per Pod, with bounded file handoffs.

All six stages start together. Normalization advances only after embedding and
index updates acknowledge the window. Artifacts live under the Workflow name;
final models retain the paths consumed by incremental workers.
"""
import argparse
from contextlib import contextmanager
import gc
import os
from pathlib import Path
import pickle
import signal
import time


STAGES = (
    "normalization", "graph-construction", "random-walk", "embedding-training",
    "cg-feature-extraction", "feature-index-construction",
)


class Handoff:
    def __init__(self, root, poll_seconds=0.5, timeout_seconds=86400):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.poll_seconds = float(poll_seconds)
        self.timeout_seconds = float(timeout_seconds)
        if self.poll_seconds <= 0 or self.timeout_seconds <= 0:
            raise ValueError("Training handoff poll and timeout must be positive")

    def path(self, channel, window):
        return self.root / f"{channel}-{window}.pkl"

    def put(self, channel, window, value):
        path = self.path(channel, window)
        temporary = path.with_suffix(f".{os.getpid()}.tmp")
        with temporary.open("wb") as stream:
            pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, path)

    def take(self, channel, window):
        path = self.path(channel, window)
        start = time.monotonic()
        while True:
            failures = list(self.root.glob("failed-*.pkl"))
            if failures:
                with failures[0].open("rb") as stream:
                    raise RuntimeError(f"Training peer failed: {pickle.load(stream)}")
            if path.exists():
                # Only internally produced, run-scoped artifacts are accepted here.
                with path.open("rb") as stream:
                    value = pickle.load(stream)
                path.unlink()
                return value
            if time.monotonic() - start > self.timeout_seconds:
                raise TimeoutError(f"Timed out waiting for training {channel}, window={window}")
            time.sleep(self.poll_seconds)


@contextmanager
def timed(stage, window, operation):
    start = time.perf_counter()
    print(f"[training] stage={stage} window={window} operation={operation} start", flush=True)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"[training] stage={stage} window={window} operation={operation} seconds={elapsed:.3f}", flush=True)


def normalize(config, bus):
    from pipeline.normalization import index_normalization
    from utils.record_batches import iter_record_batches

    cfg = config.get("batch_processing", {})
    count = 0
    for count, raw in enumerate(iter_record_batches(
        config.get("data_source_A"),
        batch_rows=int(cfg.get("rows_per_batch", 10000)),
        max_batch_bytes=int(cfg.get("max_bytes_per_batch", 128 * 1024 * 1024)),
        record_format=cfg.get("input_format"),
    ), start=1):
        with timed("normalization", count, "compute"):
            processed = index_normalization(config=config, raw_data=raw, is_training=True)
        with timed("normalization", count, "write"):
            bus.put("processed-graph", count, processed)
            bus.put("processed-features", count, processed)
        del raw, processed
        with timed("normalization", count, "wait-downstream"):
            bus.take("embedding-done", count)
            bus.take("index-done", count)
        gc.collect()
    if count == 0:
        raise ValueError("Training source contains no records")
    bus.put("processed-graph", count + 1, None)
    bus.put("processed-features", count + 1, None)


def graph_stage(config, bus):
    from pipeline.graph_construction import clear_dyn_roots, load_or_create_graph, persist_graph
    from utils.pipeline_io import get_model_directory

    final_path = os.path.join(get_model_directory(config, "graph"), "graph.graphml")
    graph = load_or_create_graph(config, final_path, enable_samplers=False)
    window = 1
    while True:
        data = bus.take("processed-graph", window)
        if data is None:
            with timed("graph-construction", window, "save-final"):
                persist_graph(graph, final_path)
            bus.put("graph", window, None)
            return
        with timed("graph-construction", window, "compute"):
            graph.build_relation(data)
        del data
        graph._tokenize_cached.cache_clear()
        roots = graph.dyn_roots
        names = lambda indices: [graph.graph.vs[int(i)]["name"] for i in indices]
        root_names = {key: names(value) for key, value in roots.items()} if isinstance(roots, dict) else names(roots)
        snapshot = str(bus.root / f"graph-{window}.graphml")
        with timed("graph-construction", window, "write-snapshot"):
            persist_graph(graph, snapshot)
            bus.put("graph", window, {"path": snapshot, "roots": root_names})
        clear_dyn_roots(graph)
        # Do not mutate or replace a snapshot while the reader is using it.
        bus.take("graph-done", window)
        window += 1


def walk_stage(config, bus):
    from pipeline.graph_construction import load_or_create_graph
    from pipeline.random_walk import dynrandom_walks_generation

    window = 1
    while True:
        handoff = bus.take("graph", window)
        if handoff is None:
            bus.put("sequences", window, None)
            return
        with timed("random-walk", window, "load-and-prepare-samplers"):
            graph = load_or_create_graph(config, handoff["path"])
            restore = lambda names: {graph.name2idx[name] for name in names}
            roots = handoff["roots"]
            graph.dyn_roots = {key: restore(value) for key, value in roots.items()} if isinstance(roots, dict) else restore(roots)
        with timed("random-walk", window, "compute"):
            sequences = dynrandom_walks_generation(config, graph) if graph.dyn_roots else []
        del graph
        gc.collect()
        with timed("random-walk", window, "write"):
            bus.put("sequences", window, sequences)
        del sequences
        os.unlink(handoff["path"])
        bus.put("graph-done", window, True)
        window += 1


def embedding_stage(config, bus):
    from pipeline.embedding_training import load_or_create_model, train_embeddings
    from utils.pipeline_io import get_model_directory

    path = os.path.join(get_model_directory(config, "embedding"), "embedding.emb")
    model = load_or_create_model(config, path)
    window = 1
    while True:
        sequences = bus.take("sequences", window)
        if sequences is None:
            with timed("embedding-training", window, "save-final"):
                model.save(path)
            return
        with timed("embedding-training", window, "compute"):
            if sequences:
                model = train_embeddings(config, model, sequences)
        del sequences
        gc.collect()
        bus.put("embedding-done", window, True)
        window += 1


def feature_stage(config, bus):
    from pipeline.cg_feature_extraction import compute_features

    method = config["candidate_generation"]["method"]
    window = 1
    while True:
        data = bus.take("processed-features", window)
        if data is None:
            bus.put("features", window, None)
            return
        with timed("cg-feature-extraction", window, "compute"):
            features = compute_features(data, method, config)
        del data
        with timed("cg-feature-extraction", window, "write"):
            bus.put("features", window, features)
        del features
        window += 1


def index_stage(config, bus):
    from pipeline.feature_index_construction import build_index, create_cg_index
    from utils.pipeline_io import get_model_directory

    index = create_cg_index(config["candidate_generation"]["method"], config,
                            index_dir=get_model_directory(config, "index"))
    window = 1
    while True:
        features = bus.take("features", window)
        if features is None:
            with timed("feature-index-construction", window, "save-final"):
                index.persist()
            return
        with timed("feature-index-construction", window, "compute"):
            build_index(features, index)
        del features
        gc.collect()
        bus.put("index-done", window, True)
        window += 1


RUNNERS = dict(zip(STAGES, (normalize, graph_stage, walk_stage, embedding_stage, feature_stage, index_stage)))


def run_stage(stage, config, bus):
    try:
        if not config.get("candidate_generation", {}).get("method"):
            raise ValueError("candidate_generation.method is required")
        RUNNERS[stage](config, bus)
    except BaseException as exc:
        bus.put(f"failed-{stage}", 0, f"{stage}: {type(exc).__name__}: {exc}")
        raise


def main():
    from utils.pipeline_io import get_transfer_data_directory, load_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="/app/config/examples/config-embedding.yaml")
    parser.add_argument("--workload", required=True, help="Unique Argo Workflow name")
    parser.add_argument("--stage", required=True, choices=STAGES)
    args = parser.parse_args()
    config = load_config(args.config)
    cfg = config.get("batch_processing", {})
    bus = Handoff(get_transfer_data_directory(args.workload, "embedding-training"),
                  cfg.get("poll_interval_seconds", 0.5), cfg.get("handoff_timeout_seconds", 86400))

    def stop(signum, frame):
        raise InterruptedError(f"Training stage received signal {signum}")

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    run_stage(args.stage, config, bus)


if __name__ == "__main__":
    main()

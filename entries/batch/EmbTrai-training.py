"""One long-running training stage per Pod, with bounded file handoffs.

All six stages start together. Normalization advances only after embedding and
index updates acknowledge the window. Artifacts live under the Workflow name;
final models retain the paths consumed by incremental workers.

The wait/load/EOS/timing loop is the shared StreamStage from utils.pipeline_io (the same one
the incremental workers use); this file only supplies each stage's handler and the
run-scoped Handoff transport.
"""
import argparse
from contextlib import contextmanager
import gc
import os
import signal
import time

from utils.pipeline_io import (
    Handoff,
    HandoffIO,
    StreamStage,
    begin_window,
    end_window,
    get_model_directory,
    get_transfer_data_directory,
    load_config,
)


STAGES = (
    "normalization", "graph-construction", "random-walk", "embedding-training",
    "cg-feature-extraction", "feature-index-construction",
)


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
    """Source stage: reads dataset chunks itself, so it drives the window clock directly."""
    from pipeline.normalization import index_normalization
    from utils.record_batches import iter_record_batches

    cfg = config.get("batch_processing", {})
    count = 0
    begin_window("normalization")
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
        end_window(count)
    if count == 0:
        raise ValueError("Training source contains no records")
    bus.put("processed-graph", count + 1, None)
    bus.put("processed-features", count + 1, None)
    end_window("eos")


def graph_stage(config, bus):
    from pipeline.graph_construction import (clear_dyn_roots, load_or_create_graph,
                                               persist_graph, serialize_dyn_roots)

    compact = config.get("graph_construction", {}).get("backend") == "compact_adjacency"
    final_path = os.path.join(get_model_directory(config, "graph"),
                              "graph_snapshot_current" if compact else "graph.graphml")
    graph = load_or_create_graph(config, final_path)
    stage = StreamStage("graph-construction", HandoffIO(bus, "processed-graph", "graph"), handle_signals=False)

    def process(window):
        data = window.take()
        with timed("graph-construction", window.index, "compute"):
            graph.build_relation(data)
        del data
        graph._tokenize_cached.cache_clear()
        root_names = serialize_dyn_roots(graph)
        snapshot = str(bus.root / (f"graph_snapshot_{window.index:06d}" if compact
                                   else f"graph-{window.index}.graphml"))
        with timed("graph-construction", window.index, "write-snapshot"):
            persist_graph(graph, snapshot)
            stage.send(window.index, {"path": snapshot, "roots": root_names})
        clear_dyn_roots(graph)
        # Do not mutate or replace a snapshot while the reader is using it.
        bus.take("graph-done", window.index)

    def save_final():
        with timed("graph-construction", "final", "save-final"):
            persist_graph(graph, final_path)

    stage.run(process, finalize=save_final)


def walk_stage(config, bus):
    from pipeline.graph_construction import load_graph_reader, restore_dyn_roots
    from pipeline.random_walk import dynrandom_walks_generation

    stage = StreamStage("random-walk", HandoffIO(bus, "graph", "sequences"), handle_signals=False)

    def process(window):
        handoff = window.take()
        with timed("random-walk", window.index, "load-graph"):
            graph = load_graph_reader(config, handoff["path"])
            restore_dyn_roots(graph, handoff["roots"])
        with timed("random-walk", window.index, "compute"):
            sequences = dynrandom_walks_generation(config, graph) if graph.dyn_roots else []
        del graph
        gc.collect()
        with timed("random-walk", window.index, "write"):
            stage.send(window.index, sequences)
        del sequences
        if os.path.isdir(handoff["path"]):
            import shutil
            shutil.rmtree(handoff["path"])
        else:
            os.unlink(handoff["path"])
        bus.put("graph-done", window.index, True)

    stage.run(process)


def embedding_stage(config, bus):
    from pipeline.embedding_training import load_or_create_model, train_embeddings

    path = os.path.join(get_model_directory(config, "embedding"), "embedding.emb")
    model = load_or_create_model(config, path)
    stage = StreamStage("embedding-training", HandoffIO(bus, "sequences"), handle_signals=False)

    def process(window):
        nonlocal model
        sequences = window.take()
        with timed("embedding-training", window.index, "compute"):
            if sequences:
                model = train_embeddings(config, model, sequences)
        del sequences
        gc.collect()
        bus.put("embedding-done", window.index, True)

    def save_final():
        with timed("embedding-training", "final", "save-final"):
            model.save(path)

    stage.run(process, finalize=save_final)


def feature_stage(config, bus):
    from pipeline.cg_feature_extraction import compute_features

    method = config["candidate_generation"]["method"]
    stage = StreamStage("cg-feature-extraction", HandoffIO(bus, "processed-features", "features"), handle_signals=False)

    def process(window):
        data = window.take()
        with timed("cg-feature-extraction", window.index, "compute"):
            features = compute_features(data, method, config)
        del data
        with timed("cg-feature-extraction", window.index, "write"):
            stage.send(window.index, features)
        del features

    stage.run(process)


def index_stage(config, bus):
    from pipeline.feature_index_construction import build_index, create_cg_index

    index = create_cg_index(config["candidate_generation"]["method"], config,
                            index_dir=get_model_directory(config, "index"))
    stage = StreamStage("feature-index-construction", HandoffIO(bus, "features"), handle_signals=False)

    def process(window):
        features = window.take()
        with timed("feature-index-construction", window.index, "compute"):
            build_index(features, index)
        del features
        gc.collect()
        bus.put("index-done", window.index, True)

    def save_final():
        with timed("feature-index-construction", "final", "save-final"):
            index.persist()

    stage.run(process, finalize=save_final)


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

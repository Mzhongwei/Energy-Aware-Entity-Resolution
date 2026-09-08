import argparse
import gc
import os

from pipeline.cg_feature_extraction import compute_features
from pipeline.embedding_training import load_or_create_model, train_embeddings
from pipeline.feature_index_construction import build_index, create_cg_index
from pipeline.graph_construction import clear_dyn_roots, load_or_create_graph, persist_graph
from pipeline.normalization import index_normalization
from pipeline.random_walk import dynrandom_walks_generation
from utils.pipeline_io import get_model_directory, load_config
from utils.record_batches import iter_record_batches


GRAPH_FILE_NAME = "graph.graphml"
MODEL_FILE_NAME = "embedding.emb"


def _batch_config(config):
    batch_config = config.get("batch_processing", {})
    if not isinstance(batch_config, dict):
        raise ValueError("batch_processing must be a mapping.")
    return (
        int(batch_config.get("rows_per_batch", 10_000)),
        int(batch_config.get("max_bytes_per_batch", 128 * 1024 * 1024)),
        batch_config.get("input_format"),
    )


def train_in_windows(config):
    source_path = config.get("data_source_A")
    rows_per_batch, max_bytes_per_batch, input_format = _batch_config(config)
    graph_path = os.path.join(get_model_directory(config, "graph"), GRAPH_FILE_NAME)
    model_path = os.path.join(get_model_directory(config, "embedding"), MODEL_FILE_NAME)
    index_dir = get_model_directory(config, "index")

    graph = load_or_create_graph(config, graph_path)
    model = load_or_create_model(config, model_path)
    candidate_config = config.get("candidate_generation")
    if not isinstance(candidate_config, dict) or not candidate_config.get("method"):
        raise ValueError("candidate_generation.method is required for embedding training.")
    method = str(candidate_config["method"]).strip()
    index = create_cg_index(method, config, index_dir=index_dir)

    window_count = 0
    record_count = 0
    for window_count, raw_batch in enumerate(
        iter_record_batches(
            source_path,
            batch_rows=rows_per_batch,
            max_batch_bytes=max_bytes_per_batch,
            record_format=input_format,
        ),
        start=1,
    ):
        processed = index_normalization(
            config=config,
            raw_data=raw_batch,
            raw_data_path=None,
            is_training=True,
        )
        record_count += len(processed)
        del raw_batch

        graph.build_relation(processed)
        sequences = dynrandom_walks_generation(config, graph) if graph.dyn_roots else []
        if sequences:
            model = train_embeddings(config, model, sequences)
        del sequences
        clear_dyn_roots(graph)

        features = compute_features(processed, method, config)
        build_index(features, index)
        del features
        del processed
        gc.collect()
        print(f"[embedding-training] completed window={window_count} records={record_count}", flush=True)

    if window_count == 0:
        raise ValueError(f"Training source contains no records: {source_path}")

    persist_graph(graph, graph_path)
    model.save(model_path)
    index.persist()
    print(
        f"[embedding-training] complete windows={window_count} records={record_count} "
        f"graph={graph_path} embedding={model_path} index={index_dir}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Windowed embedding training for CSV or JSONL data.")
    parser.add_argument("--config", default="/app/config/examples/config-embedding.yaml")
    args = parser.parse_args()
    train_in_windows(load_config(args.config))


if __name__ == "__main__":
    main()

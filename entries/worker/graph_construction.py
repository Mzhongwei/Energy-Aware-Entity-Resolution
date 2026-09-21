import argparse
import os

from utils.pipeline_io import (
    BufferIO,
    StageStop,
    StreamStage,
    get_model_directory,
    is_empty_payload,
    is_window_published,
    latest_graph_checkpoint,
    load_config,
    load_window_checkpoint,
    mark_window_published,
    prune_window_checkpoints,
    wait_for_window_checkpoint_ack,
    write_checkpoint_reference,
    write_window_checkpoint,
)
from pipeline.graph_construction import (
    clear_dyn_roots,
    load_or_create_graph,
    persist_graph,
    serialize_dyn_roots,
)

"""
task: representation graph construction
mode: incremental + embedding
input: processed data for graph tasks [buffer]
output: graph handoff (snapshot path + dyn_roots) [buffer]
description: seeds from the batch-trained graph, then keeps one immutable checkpoint and
ships only its path and dyn_roots downstream.
"""

INPUT_DATA_TYPE = "processed_data_graph"
OUTPUT_DATA_TYPE = "graph"
GRAPH_FILE_NAME = "graph.graphml"


def main():
    parser = argparse.ArgumentParser(description="Worker entry for incremental graph construction.")
    parser.add_argument("--config", default="/app/config/examples/config-embedding.yaml")
    parser.add_argument("--workload", default="default")
    args = parser.parse_args()

    config = load_config(args.config)
    io = BufferIO(args.workload, INPUT_DATA_TYPE, OUTPUT_DATA_TYPE, config)
    graph_path = os.path.join(get_model_directory(config, "graph"), GRAPH_FILE_NAME)
    checkpoint_dir = os.path.join(os.path.dirname(graph_path), "incremental_checkpoints")
    latest_checkpoint = latest_graph_checkpoint(checkpoint_dir)
    checkpoint_window = latest_checkpoint[0] if latest_checkpoint else -1
    checkpoint_graph_path = latest_checkpoint[1] if latest_checkpoint else graph_path
    graph = load_or_create_graph(config, checkpoint_graph_path)
    if latest_checkpoint:
        write_checkpoint_reference(os.path.dirname(graph_path), "current", checkpoint_graph_path, checkpoint_window)
        if os.path.isfile(graph_path):
            os.remove(graph_path)
        prune_window_checkpoints(checkpoint_dir, checkpoint_window)

    def wait_for_previous_window_ack():
        # Do not mutate the graph while the previous snapshot is still being read downstream.
        if (
            checkpoint_window >= 0
            and is_window_published(checkpoint_dir, checkpoint_window)
            and not wait_for_window_checkpoint_ack(
                checkpoint_dir, checkpoint_window,
                should_stop=stage.should_stop, poll_interval_seconds=io.poll_interval,
            )
        ):
            raise StageStop

    stage = StreamStage(
        "graph_construction", io, is_empty=is_empty_payload, before_load=wait_for_previous_window_ack,
    )

    def process(window):
        nonlocal checkpoint_window, checkpoint_graph_path
        processed_data = window.take()
        window_index = window.index

        if window_index <= checkpoint_window:
            # Replay after a restart: re-publish the checkpoint reference if it was lost.
            if window_index == checkpoint_window and not is_window_published(checkpoint_dir, window_index):
                checkpoint = load_window_checkpoint(checkpoint_dir, window_index) or {}
                write_checkpoint_reference(
                    io.output_dir(), window_index, checkpoint_graph_path, checkpoint_window,
                    dyn_roots=checkpoint.get("dyn_roots", []),
                )
                mark_window_published(checkpoint_dir, window_index)
            prune_window_checkpoints(checkpoint_dir, checkpoint_window)
            return

        graph.build_relation(processed_data)
        del processed_data
        graph._tokenize_cached.cache_clear()

        dyn_roots = serialize_dyn_roots(graph)
        checkpoint_graph_path = os.path.join(checkpoint_dir, f"{window_index}.graphml")
        persist_graph(graph, checkpoint_graph_path)
        write_window_checkpoint(checkpoint_dir, window_index, {"dyn_roots": dyn_roots})
        write_checkpoint_reference(os.path.dirname(graph_path), "current", checkpoint_graph_path, window_index)
        write_checkpoint_reference(
            io.output_dir(), window_index, checkpoint_graph_path, window_index, dyn_roots=dyn_roots,
        )
        mark_window_published(checkpoint_dir, window_index)
        if os.path.isfile(graph_path):
            os.remove(graph_path)
        clear_dyn_roots(graph)
        checkpoint_window = window_index
        prune_window_checkpoints(checkpoint_dir, checkpoint_window)

    stage.run(process)


if __name__ == "__main__":
    main()

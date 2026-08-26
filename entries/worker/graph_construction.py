import argparse
import os
import signal
import sys

from utils.pipeline_io import (
    copy_file_atomic,
    delete_earliest_buffer_file,
    get_buffer_directory,
    get_earliest_window_index,
    get_incremental_wait_config,
    get_model_directory,
    is_window_published,
    latest_graph_checkpoint,
    load_config,
    load_earliest_buffer,
    load_window_checkpoint,
    mark_window_published,
    prune_window_checkpoints,
    wait_for_buffer,
    write_buffer,
    write_eos,
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
description: seeds from the batch-trained graph in the shared model directory (if any) and
keeps growing it in memory across windows; the shared model directory is never overwritten
from this incremental path -- only per-window snapshots are shipped downstream.
"""

INPUT_DATA_TYPE = "processed_data_graph"
OUTPUT_DATA_TYPE = "graph"
GRAPH_FILE_NAME = "graph.graphml"
stop_requested = False

def handle_sigterm(signum, frame):
    global stop_requested
    stop_requested = True

signal.signal(signal.SIGTERM, handle_sigterm)
signal.signal(signal.SIGINT, handle_sigterm)

def main():
    parser = argparse.ArgumentParser(description="Worker entry for incremental graph construction.")
    parser.add_argument("--config", default="/app/config/examples/config-embedding.yaml")
    parser.add_argument("--workload", default="default")
    args = parser.parse_args()

    INPUT_BUFFER = get_buffer_directory(args.workload, INPUT_DATA_TYPE)
    OUTPUT_BUFFER = get_buffer_directory(args.workload, OUTPUT_DATA_TYPE)
    SNAPSHOT_DIR = os.path.join(OUTPUT_BUFFER, "snapshots")

    config = load_config(args.config)
    startup_timeout, poll_interval = get_incremental_wait_config(config)
    graph_path = os.path.join(get_model_directory(config, "graph"), GRAPH_FILE_NAME)
    checkpoint_dir = os.path.join(os.path.dirname(graph_path), "incremental_checkpoints")
    latest_checkpoint = latest_graph_checkpoint(checkpoint_dir)
    checkpoint_window = latest_checkpoint[0] if latest_checkpoint else -1
    checkpoint_graph_path = latest_checkpoint[1] if latest_checkpoint else graph_path
    graph = load_or_create_graph(config, checkpoint_graph_path, enable_samplers=False)

    seen_first_item = False
    while not stop_requested:
        timeout = None if seen_first_item else startup_timeout
        ready = wait_for_buffer(
            INPUT_BUFFER, timeout_seconds=timeout, should_stop=lambda: stop_requested,
            poll_interval_seconds=poll_interval,
        )
        if ready is None:
            if stop_requested:
                return
            raise TimeoutError(
                f"[graph_construction] startup timed out after {startup_timeout}s waiting for {INPUT_BUFFER}"
            )
        seen_first_item = True
        processed_data = load_earliest_buffer(INPUT_BUFFER)
        if processed_data is None:
            write_eos(OUTPUT_BUFFER)
            print("[graph_construction] worker completed after upstream EOS", file=sys.stderr)
            return
        if processed_data.empty:
            delete_earliest_buffer_file(INPUT_BUFFER)
            continue

        window_index = get_earliest_window_index(INPUT_BUFFER)
        snapshot_path = os.path.join(SNAPSHOT_DIR, f"graph_window_{window_index}.graphml")

        if window_index <= checkpoint_window:
            if window_index == checkpoint_window and not is_window_published(checkpoint_dir, window_index):
                checkpoint = load_window_checkpoint(checkpoint_dir, window_index) or {}
                copy_file_atomic(checkpoint_graph_path, snapshot_path)
                write_buffer(
                    {"graph_path": snapshot_path, "dyn_roots": checkpoint.get("dyn_roots", [])},
                    OUTPUT_BUFFER,
                    window_index,
                    extension="json",
                )
                mark_window_published(checkpoint_dir, window_index)
            delete_earliest_buffer_file(INPUT_BUFFER)
            prune_window_checkpoints(checkpoint_dir, checkpoint_window)
            continue

        graph.build_relation(processed_data)
        del processed_data
        graph._tokenize_cached.cache_clear()

        dyn_roots = serialize_dyn_roots(graph)
        checkpoint_graph_path = os.path.join(checkpoint_dir, f"{window_index}.graphml")
        persist_graph(graph, checkpoint_graph_path)
        write_window_checkpoint(checkpoint_dir, window_index, {"dyn_roots": dyn_roots})
        copy_file_atomic(checkpoint_graph_path, snapshot_path)

        write_buffer(
            {"graph_path": snapshot_path, "dyn_roots": dyn_roots},
            OUTPUT_BUFFER,
            window_index,
            extension="json",
        )
        mark_window_published(checkpoint_dir, window_index)
        clear_dyn_roots(graph)
        delete_earliest_buffer_file(INPUT_BUFFER)
        checkpoint_window = window_index
        prune_window_checkpoints(checkpoint_dir, checkpoint_window)

    print("[graph_construction] worker stopped without emitting EOS", file=sys.stderr)

if __name__ == "__main__":
    main()

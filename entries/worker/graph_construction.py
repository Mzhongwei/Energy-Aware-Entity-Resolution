import argparse
import os
import signal
import sys

from utils.pipeline_io import (
    delete_earliest_buffer_file,
    get_buffer_directory,
    get_earliest_window_index,
    get_incremental_wait_config,
    get_model_directory,
    load_config,
    load_earliest_buffer,
    wait_for_buffer,
    write_buffer,
    write_eos,
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
    graph = load_or_create_graph(config, graph_path, enable_samplers=False)

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
        graph.build_relation(processed_data)
        del processed_data

        snapshot_path = os.path.join(SNAPSHOT_DIR, f"graph_window_{window_index}.graphml")
        persist_graph(graph, snapshot_path)
        handoff = {"graph_path": snapshot_path, "dyn_roots": serialize_dyn_roots(graph)}

        write_buffer(handoff, OUTPUT_BUFFER, window_index, extension="json")
        del handoff
        clear_dyn_roots(graph)
        delete_earliest_buffer_file(INPUT_BUFFER)

    print("[graph_construction] worker stopped without emitting EOS", file=sys.stderr)

if __name__ == "__main__":
    main()

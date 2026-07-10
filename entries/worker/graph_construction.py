import argparse
import os
import signal
import sys

from utils.pipeline_io import (
    delete_earliest_buffer_file,
    get_buffer_directory,
    get_earliest_window_index,
    get_model_directory,
    load_config,
    load_earliest_buffer,
    wait_for_buffer,
    write_buffer,
    write_eos,
)
from pipeline.graph_construction import load_or_create_graph, persist_graph, serialize_dyn_roots

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
    graph_path = os.path.join(get_model_directory(config, "graph"), GRAPH_FILE_NAME)
    graph = load_or_create_graph(config, graph_path)  # seed from batch-trained graph if present

    while not stop_requested:
        ready = wait_for_buffer(INPUT_BUFFER, timeout_seconds=120)
        if ready is None:
            break
        processed_data = load_earliest_buffer(INPUT_BUFFER)
        if processed_data is None:
            break
        if processed_data.empty:
            delete_earliest_buffer_file(INPUT_BUFFER)
            continue

        window_index = get_earliest_window_index(INPUT_BUFFER)
        graph.build_relation(processed_data)

        snapshot_path = os.path.join(SNAPSHOT_DIR, f"graph_window_{window_index}.graphml")
        persist_graph(graph, snapshot_path)
        handoff = {"graph_path": snapshot_path, "dyn_roots": serialize_dyn_roots(graph)}

        write_buffer(handoff, OUTPUT_BUFFER, window_index, extension="json")
        delete_earliest_buffer_file(INPUT_BUFFER)

    write_eos(OUTPUT_BUFFER, reason="stream_completed")
    print("[graph_construction] worker completed", file=sys.stderr)

if __name__ == "__main__":
    main()

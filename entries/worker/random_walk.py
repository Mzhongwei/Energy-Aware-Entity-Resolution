import argparse
import os
import signal
import sys

from utils.pipeline_io import (
    delete_earliest_buffer_file,
    get_buffer_directory,
    get_earliest_window_index,
    load_config,
    load_earliest_buffer,
    wait_for_buffer,
    write_buffer,
    write_eos,
)
from pipeline.graph_construction import load_or_create_graph, restore_dyn_roots
from pipeline.random_walk import dynrandom_walks_generation

"""
task: random walk sequence generation
mode: incremental + embedding
input: graph handoff (snapshot path + dyn_roots) [buffer]
output: walk sequences [buffer]
description: dyn_roots restored from the handoff drive each window's walk directly --
no whole-graph traversal is needed since graph construction already carries the roots.
"""

INPUT_DATA_TYPE = "graph"
OUTPUT_DATA_TYPE = "sequences"
TASK_CONFIG_KEY = "random_walk"

# See normalization_embedding.py for why first/steady waits are split and what a timeout
# (vs an explicit upstream EOS) means.
DEFAULT_FIRST_WAIT_TIMEOUT_SECONDS = 1800
DEFAULT_WAIT_TIMEOUT_SECONDS = 120

stop_requested = False

def handle_sigterm(signum, frame):
    global stop_requested
    stop_requested = True

signal.signal(signal.SIGTERM, handle_sigterm)
signal.signal(signal.SIGINT, handle_sigterm)

def main():
    parser = argparse.ArgumentParser(description="Worker entry for incremental random walk.")
    parser.add_argument("--config", default="/app/config/examples/config-embedding.yaml")
    parser.add_argument("--workload", default="default")
    args = parser.parse_args()

    INPUT_BUFFER = get_buffer_directory(args.workload, INPUT_DATA_TYPE)
    OUTPUT_BUFFER = get_buffer_directory(args.workload, OUTPUT_DATA_TYPE)

    config = load_config(args.config)
    task_config = config.get(TASK_CONFIG_KEY, {}) or {}
    first_wait_timeout = int(task_config.get("first_wait_timeout_seconds", DEFAULT_FIRST_WAIT_TIMEOUT_SECONDS))
    wait_timeout = int(task_config.get("wait_timeout_seconds", DEFAULT_WAIT_TIMEOUT_SECONDS))

    seen_first_item = False
    eos_reason = "stream_completed"
    while not stop_requested:
        timeout = wait_timeout if seen_first_item else first_wait_timeout
        ready = wait_for_buffer(INPUT_BUFFER, timeout_seconds=timeout, should_stop=lambda: stop_requested)
        if ready is None:
            if not stop_requested:
                print(
                    f"[WARNING] [random_walk] timed out after {timeout}s waiting for {INPUT_BUFFER} "
                    "with no upstream EOS seen; exiting as if stream ended (possible silent data loss upstream).",
                    file=sys.stderr,
                )
                eos_reason = "timeout_no_upstream_eos"
            break
        seen_first_item = True
        handoff = load_earliest_buffer(INPUT_BUFFER)
        if handoff is None:
            break
        if not handoff:
            delete_earliest_buffer_file(INPUT_BUFFER)
            continue

        window_index = get_earliest_window_index(INPUT_BUFFER)
        graph_path = handoff["graph_path"]
        graph = load_or_create_graph(config, graph_path)
        restore_dyn_roots(graph, handoff.get("dyn_roots"))

        sequences = dynrandom_walks_generation(config, graph) if graph.dyn_roots else []
        write_buffer(sequences, OUTPUT_BUFFER, window_index, extension="json")

        if os.path.exists(graph_path):
            os.remove(graph_path)
        delete_earliest_buffer_file(INPUT_BUFFER)

    write_eos(OUTPUT_BUFFER, reason=eos_reason)
    print("[random_walk] worker completed", file=sys.stderr)

if __name__ == "__main__":
    main()

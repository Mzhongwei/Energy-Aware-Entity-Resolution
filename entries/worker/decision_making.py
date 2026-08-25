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
from models.embedding_model import EmbeddingModel
from pipeline.decision_making import decide_matches
from utils.utils import load_scored_pairs_from_graphml

"""
task: decision making
mode: incremental + embedding
input: matching pairs [buffer]
output: decision event, triggering evaluation [buffer]
description: rescores conflicts against the latest embedding model in the shared model
directory (not a per-window snapshot -- the model keeps evolving), and overwrites the
predicted matching graph at a fixed path so it always reflects the most recent decision
(for external inspection). Also persists a per-window snapshot under the output buffer and
references it in the buffer event, so evaluation -- which may run concurrently with this
worker -- evaluates the exact graph produced for that window rather than whatever the fixed
path happens to hold when it gets around to reading it.
"""

INPUT_DATA_TYPE = "matching_pairs"
OUTPUT_DATA_TYPE = "predicted_matching"
MODEL_FILE_NAME = "embedding.emb"
PREDICTED_MATCH_FILE_NAME = "predicted_matching.graphml"
SNAPSHOT_FILE_NAME_TEMPLATE = "predicted_matching_window_{window_index}.graphml"
TASK_CONFIG_KEY = "decision_making"

stop_requested = False

def handle_sigterm(signum, frame):
    global stop_requested
    stop_requested = True

signal.signal(signal.SIGTERM, handle_sigterm)
signal.signal(signal.SIGINT, handle_sigterm)

def main():
    parser = argparse.ArgumentParser(description="Worker entry for incremental decision making.")
    parser.add_argument("--config", default="/app/config/examples/config-embedding.yaml")
    parser.add_argument("--workload", default="default")
    args = parser.parse_args()

    INPUT_BUFFER = get_buffer_directory(args.workload, INPUT_DATA_TYPE)
    OUTPUT_BUFFER = get_buffer_directory(args.workload, OUTPUT_DATA_TYPE)
    SNAPSHOT_DIR = os.path.join(OUTPUT_BUFFER, "snapshots")
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)

    config = load_config(args.config)
    task_config = config.get(TASK_CONFIG_KEY, {}) or {}
    output_format = task_config.get("output_format", "graphml")
    startup_timeout, poll_interval = get_incremental_wait_config(config)
    model_path = os.path.join(get_model_directory(config, "embedding"), MODEL_FILE_NAME)
    predicted_match_path = os.path.join(get_model_directory(config, "predicted_match"), PREDICTED_MATCH_FILE_NAME)
    os.makedirs(os.path.dirname(predicted_match_path), exist_ok=True)

    previous_pairs = (
        load_scored_pairs_from_graphml(config, predicted_match_path)
        if os.path.isfile(predicted_match_path)
        else None
    )
    if previous_pairs:
        print(f"[decision_making] restored {len(previous_pairs)} previous pairs from {predicted_match_path}")
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
                f"[decision_making] startup timed out after {startup_timeout}s waiting for {INPUT_BUFFER}"
            )
        seen_first_item = True
        matching_pairs = load_earliest_buffer(INPUT_BUFFER)
        if matching_pairs is None:
            write_eos(OUTPUT_BUFFER)
            print("[decision_making] worker completed after upstream EOS", file=sys.stderr)
            return
        if not matching_pairs:
            delete_earliest_buffer_file(INPUT_BUFFER)
            continue

        window_index = get_earliest_window_index(INPUT_BUFFER)
        model = EmbeddingModel.load(model_path)
        previous_pairs, predicted_graph = decide_matches(
            mutualtop_pairs=matching_pairs,
            previous_pairs=previous_pairs,
            model=model,
            output_format=output_format,
        )
        del model
        del matching_pairs
        predicted_graph.export_graphml(predicted_match_path)

        snapshot_path = os.path.join(SNAPSHOT_DIR, SNAPSHOT_FILE_NAME_TEMPLATE.format(window_index=window_index))
        predicted_graph.export_graphml(snapshot_path)

        write_buffer(
            {"pair_count": len(previous_pairs), "predicted_match_path": snapshot_path},
            OUTPUT_BUFFER,
            window_index,
            extension="json",
        )
        del predicted_graph
        delete_earliest_buffer_file(INPUT_BUFFER)

    print("[decision_making] worker stopped without emitting EOS", file=sys.stderr)

if __name__ == "__main__":
    main()

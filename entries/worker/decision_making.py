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
from models.embedding_model import EmbeddingModel
from pipeline.decision_making import decide_matches

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
    first_wait_timeout = int(task_config.get("first_wait_timeout_seconds", DEFAULT_FIRST_WAIT_TIMEOUT_SECONDS))
    wait_timeout = int(task_config.get("wait_timeout_seconds", DEFAULT_WAIT_TIMEOUT_SECONDS))
    model_path = os.path.join(get_model_directory(config, "embedding"), MODEL_FILE_NAME)
    predicted_match_path = os.path.join(get_model_directory(config, "predicted_match"), PREDICTED_MATCH_FILE_NAME)

    previous_pairs = None
    seen_first_item = False
    eos_reason = "stream_completed"
    while not stop_requested:
        timeout = wait_timeout if seen_first_item else first_wait_timeout
        ready = wait_for_buffer(INPUT_BUFFER, timeout_seconds=timeout, should_stop=lambda: stop_requested)
        if ready is None:
            if not stop_requested:
                print(
                    f"[WARNING] [decision_making] timed out after {timeout}s waiting for {INPUT_BUFFER} "
                    "with no upstream EOS seen; exiting as if stream ended (possible silent data loss upstream).",
                    file=sys.stderr,
                )
                eos_reason = "timeout_no_upstream_eos"
            break
        seen_first_item = True
        matching_pairs = load_earliest_buffer(INPUT_BUFFER)
        if matching_pairs is None:
            break
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
        predicted_graph.export_graphml(predicted_match_path)

        snapshot_path = os.path.join(SNAPSHOT_DIR, SNAPSHOT_FILE_NAME_TEMPLATE.format(window_index=window_index))
        predicted_graph.export_graphml(snapshot_path)

        write_buffer(
            {"pair_count": len(previous_pairs), "predicted_match_path": snapshot_path},
            OUTPUT_BUFFER,
            window_index,
            extension="json",
        )
        delete_earliest_buffer_file(INPUT_BUFFER)

    write_eos(OUTPUT_BUFFER, reason=eos_reason)
    print("[decision_making] worker completed", file=sys.stderr)

if __name__ == "__main__":
    main()

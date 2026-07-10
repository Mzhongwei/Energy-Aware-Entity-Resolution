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
predicted matching graph so it always reflects the most recent decision.
"""

INPUT_DATA_TYPE = "matching_pairs"
OUTPUT_DATA_TYPE = "predicted_matching"
MODEL_FILE_NAME = "embedding.emb"
PREDICTED_MATCH_FILE_NAME = "predicted_matching.graphml"

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

    config = load_config(args.config)
    output_format = config.get("decision_making", {}).get("output_format", "graphml")
    model_path = os.path.join(get_model_directory(config, "embedding"), MODEL_FILE_NAME)
    predicted_match_path = os.path.join(get_model_directory(config, "predicted_match"), PREDICTED_MATCH_FILE_NAME)

    previous_pairs = None
    while not stop_requested:
        ready = wait_for_buffer(INPUT_BUFFER, timeout_seconds=120)
        if ready is None:
            break
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

        write_buffer({"pair_count": len(previous_pairs)}, OUTPUT_BUFFER, window_index, extension="json")
        delete_earliest_buffer_file(INPUT_BUFFER)

    write_eos(OUTPUT_BUFFER, reason="stream_completed")
    print("[decision_making] worker completed", file=sys.stderr)

if __name__ == "__main__":
    main()

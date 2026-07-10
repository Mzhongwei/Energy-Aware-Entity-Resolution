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
    wait_for_embedding_buffer,
    write_buffer,
    write_eos,
)
from models.embedding_model import EmbeddingModel
from pipeline.calculating_similarity import score_mutual_top1_candidate_pairs

"""
task: similarity calculation
mode: incremental + embedding
input: candidate pairs [buffer], embedding model snapshot [buffer]
output: matching pairs [buffer]
"""

INPUT_DATA_TYPE = "candidate_pairs"
EMBEDDING_INPUT_DATA_TYPE = "embedding_calculating"
OUTPUT_DATA_TYPE = "matching_pairs"

stop_requested = False

def handle_sigterm(signum, frame):
    global stop_requested
    stop_requested = True

signal.signal(signal.SIGTERM, handle_sigterm)
signal.signal(signal.SIGINT, handle_sigterm)

def main():
    parser = argparse.ArgumentParser(description="Worker entry for incremental similarity calculation.")
    parser.add_argument("--config", default="/app/config/examples/config-embedding.yaml")
    parser.add_argument("--workload", default="default")
    args = parser.parse_args()

    INPUT_BUFFER = get_buffer_directory(args.workload, INPUT_DATA_TYPE)
    EMBEDDING_BUFFER = get_buffer_directory(args.workload, EMBEDDING_INPUT_DATA_TYPE)
    OUTPUT_BUFFER = get_buffer_directory(args.workload, OUTPUT_DATA_TYPE)

    config = load_config(args.config)
    batch_threshold = int(config.get("calculating_similarity", {}).get("batch_threshold", 2048))

    while not stop_requested:
        ready = wait_for_buffer(INPUT_BUFFER, timeout_seconds=120)
        if ready is None:
            break
        candidate_pairs = load_earliest_buffer(INPUT_BUFFER)
        if candidate_pairs is None:
            break
        if not candidate_pairs:
            delete_earliest_buffer_file(INPUT_BUFFER)
            continue

        window_index = get_earliest_window_index(INPUT_BUFFER)
        embedding_path = wait_for_embedding_buffer(EMBEDDING_BUFFER, window_index, timeout_seconds=30)
        if embedding_path is None:
            break

        model = EmbeddingModel.load(embedding_path)
        matching_pairs = score_mutual_top1_candidate_pairs(model, candidate_pairs, batch_threshold=batch_threshold)
        write_buffer(matching_pairs, OUTPUT_BUFFER, window_index, extension="json")

        for path in (embedding_path, f"{embedding_path}.meta.json"):
            if os.path.exists(path):
                os.remove(path)
        delete_earliest_buffer_file(INPUT_BUFFER)

    write_eos(OUTPUT_BUFFER, reason="stream_completed")
    print("[calculating_similarity] worker completed", file=sys.stderr)

if __name__ == "__main__":
    main()

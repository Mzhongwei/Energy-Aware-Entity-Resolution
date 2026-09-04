import argparse
import os
import signal
import sys

from utils.pipeline_io import (
    delete_earliest_buffer_file,
    get_buffer_directory,
    get_earliest_window_index,
    get_incremental_wait_config,
    load_config,
    load_checkpoint_reference,
    load_earliest_buffer,
    wait_for_buffer,
    wait_for_checkpoint_reference,
    write_buffer,
    write_eos,
)
from models.embedding_model import EmbeddingModel
from pipeline.calculating_similarity import get_mutual_top_k, score_mutual_topk_candidate_pairs

"""
task: similarity calculation
mode: incremental + embedding
input: candidate pairs [buffer], embedding model snapshot [buffer]
output: matching pairs [buffer]
description: reads an immutable per-window embedding checkpoint by reference.
"""

INPUT_DATA_TYPE = "candidate_pairs"
EMBEDDING_INPUT_DATA_TYPE = "embedding_calculating"
OUTPUT_DATA_TYPE = "matching_pairs"
TASK_CONFIG_KEY = "calculating_similarity"

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
    task_config = config.get(TASK_CONFIG_KEY, {}) or {}
    top_k = get_mutual_top_k(config)
    batch_threshold = int(task_config.get("batch_threshold", 2048))
    chunk_size = int(task_config.get("chunk_size", 4096))
    startup_timeout, poll_interval = get_incremental_wait_config(config)

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
                f"[calculating_similarity] startup timed out after {startup_timeout}s waiting for {INPUT_BUFFER}"
            )
        seen_first_item = True
        candidate_pairs = load_earliest_buffer(INPUT_BUFFER)
        if candidate_pairs is None:
            write_eos(OUTPUT_BUFFER)
            print("[calculating_similarity] worker completed after upstream EOS", file=sys.stderr)
            return
        window_index = get_earliest_window_index(INPUT_BUFFER)
        reference_path = wait_for_checkpoint_reference(
            EMBEDDING_BUFFER,
            window_index,
            timeout_seconds=None,
            should_stop=lambda: stop_requested,
            poll_interval_seconds=poll_interval,
        )
        if reference_path is None:
            return
        if os.path.basename(reference_path).startswith("eos_"):
            raise RuntimeError(
                f"[calculating_similarity] embedding stream ended before window {window_index} was produced"
            )
        reference = load_checkpoint_reference(reference_path)
        embedding_path = reference["checkpoint_path"]

        if candidate_pairs:
            model = EmbeddingModel.load(embedding_path)
            matching_pairs = score_mutual_topk_candidate_pairs(
                model,
                candidate_pairs,
                top_k=top_k,
                batch_threshold=batch_threshold,
                chunk_size=chunk_size,
            )
            del model
        else:
            matching_pairs = []
        del candidate_pairs
        write_buffer(matching_pairs, OUTPUT_BUFFER, window_index, extension="json")
        del matching_pairs

        delete_earliest_buffer_file(INPUT_BUFFER)

    print("[calculating_similarity] worker stopped without emitting EOS", file=sys.stderr)

if __name__ == "__main__":
    main()

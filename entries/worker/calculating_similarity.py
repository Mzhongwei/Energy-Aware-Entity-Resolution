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
description: reads the embedding snapshot without deleting it -- decision_making reads the
same snapshot downstream and owns its cleanup.
"""

INPUT_DATA_TYPE = "candidate_pairs"
EMBEDDING_INPUT_DATA_TYPE = "embedding_calculating"
OUTPUT_DATA_TYPE = "matching_pairs"
TASK_CONFIG_KEY = "calculating_similarity"

# See normalization_embedding.py for why first/steady waits are split and what a timeout
# (vs an explicit upstream EOS) means. The embedding-buffer wait has no EOS concept of its
# own (it waits for one specific window's .emb snapshot, not a stream), so a timeout there
# is always ambiguous and always logged.
DEFAULT_FIRST_WAIT_TIMEOUT_SECONDS = 1800
DEFAULT_WAIT_TIMEOUT_SECONDS = 120
DEFAULT_EMBEDDING_FIRST_WAIT_TIMEOUT_SECONDS = 1800
DEFAULT_EMBEDDING_WAIT_TIMEOUT_SECONDS = 30

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
    batch_threshold = int(task_config.get("batch_threshold", 2048))
    first_wait_timeout = int(task_config.get("first_wait_timeout_seconds", DEFAULT_FIRST_WAIT_TIMEOUT_SECONDS))
    wait_timeout = int(task_config.get("wait_timeout_seconds", DEFAULT_WAIT_TIMEOUT_SECONDS))
    embedding_first_wait_timeout = int(
        task_config.get("embedding_first_wait_timeout_seconds", DEFAULT_EMBEDDING_FIRST_WAIT_TIMEOUT_SECONDS)
    )
    embedding_wait_timeout = int(
        task_config.get("embedding_wait_timeout_seconds", DEFAULT_EMBEDDING_WAIT_TIMEOUT_SECONDS)
    )

    seen_first_item = False
    seen_first_embedding = False
    eos_reason = "stream_completed"
    while not stop_requested:
        timeout = wait_timeout if seen_first_item else first_wait_timeout
        ready = wait_for_buffer(INPUT_BUFFER, timeout_seconds=timeout, should_stop=lambda: stop_requested)
        if ready is None:
            if not stop_requested:
                print(
                    f"[WARNING] [calculating_similarity] timed out after {timeout}s waiting for {INPUT_BUFFER} "
                    "with no upstream EOS seen; exiting as if stream ended (possible silent data loss upstream).",
                    file=sys.stderr,
                )
                eos_reason = "timeout_no_upstream_eos"
            break
        seen_first_item = True
        candidate_pairs = load_earliest_buffer(INPUT_BUFFER)
        if candidate_pairs is None:
            break
        if not candidate_pairs:
            delete_earliest_buffer_file(INPUT_BUFFER)
            continue

        window_index = get_earliest_window_index(INPUT_BUFFER)
        embedding_timeout = embedding_wait_timeout if seen_first_embedding else embedding_first_wait_timeout
        embedding_path = wait_for_embedding_buffer(
            EMBEDDING_BUFFER, window_index, timeout_seconds=embedding_timeout, should_stop=lambda: stop_requested
        )
        if embedding_path is None:
            if not stop_requested:
                print(
                    f"[WARNING] [calculating_similarity] timed out after {embedding_timeout}s waiting for window "
                    f"{window_index}'s embedding snapshot in {EMBEDDING_BUFFER}; exiting (embedding_training may "
                    "have stalled or died -- this window's candidate pairs are left unprocessed).",
                    file=sys.stderr,
                )
                eos_reason = "timeout_no_upstream_eos"
            break
        seen_first_embedding = True

        model = EmbeddingModel.load(embedding_path)
        matching_pairs = score_mutual_top1_candidate_pairs(model, candidate_pairs, batch_threshold=batch_threshold)
        write_buffer(matching_pairs, OUTPUT_BUFFER, window_index, extension="json")

        for path in (embedding_path, f"{embedding_path}.meta.json"):
            if os.path.exists(path):
                os.remove(path)
        delete_earliest_buffer_file(INPUT_BUFFER)

    write_eos(OUTPUT_BUFFER, reason=eos_reason)
    print("[calculating_similarity] worker completed", file=sys.stderr)

if __name__ == "__main__":
    main()

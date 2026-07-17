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
from pipeline.embedding_training import load_or_create_model, train_embeddings

"""
task: embedding model training
mode: incremental + embedding
input: walk sequences [buffer]
output: embedding model snapshot for similarity calculation [buffer]
description: seeds from the batch-trained embedding model in the shared model directory
(if any) and keeps training it in memory across windows, persisting the growing model back
to the shared model directory on every window so training survives pod restarts.
"""

INPUT_DATA_TYPE = "sequences"
OUTPUT_DATA_TYPE = "embedding_calculating"
MODEL_FILE_NAME = "embedding.emb"
TASK_CONFIG_KEY = "embeddings_training"

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
    parser = argparse.ArgumentParser(description="Worker entry for incremental embedding training.")
    parser.add_argument("--config", default="/app/config/examples/config-embedding.yaml")
    parser.add_argument("--workload", default="default")
    args = parser.parse_args()

    INPUT_BUFFER = get_buffer_directory(args.workload, INPUT_DATA_TYPE)
    OUTPUT_BUFFER = get_buffer_directory(args.workload, OUTPUT_DATA_TYPE)

    config = load_config(args.config)
    task_config = config.get(TASK_CONFIG_KEY, {}) or {}
    first_wait_timeout = int(task_config.get("first_wait_timeout_seconds", DEFAULT_FIRST_WAIT_TIMEOUT_SECONDS))
    wait_timeout = int(task_config.get("wait_timeout_seconds", DEFAULT_WAIT_TIMEOUT_SECONDS))
    model_path = os.path.join(get_model_directory(config, "embedding"), MODEL_FILE_NAME)
    model = load_or_create_model(config, model_path)  # seed from batch-trained model if present

    seen_first_item = False
    eos_reason = "stream_completed"
    while not stop_requested:
        timeout = wait_timeout if seen_first_item else first_wait_timeout
        ready = wait_for_buffer(INPUT_BUFFER, timeout_seconds=timeout, should_stop=lambda: stop_requested)
        if ready is None:
            if not stop_requested:
                print(
                    f"[WARNING] [embedding_training] timed out after {timeout}s waiting for {INPUT_BUFFER} "
                    "with no upstream EOS seen; exiting as if stream ended (possible silent data loss upstream).",
                    file=sys.stderr,
                )
                eos_reason = "timeout_no_upstream_eos"
            break
        seen_first_item = True
        sequences = load_earliest_buffer(INPUT_BUFFER)
        if sequences is None:
            break
        if not sequences:
            delete_earliest_buffer_file(INPUT_BUFFER)
            continue

        window_index = get_earliest_window_index(INPUT_BUFFER)

        model = train_embeddings(config, model, sequences)
        model.save(model_path)

        write_buffer(model, OUTPUT_BUFFER, window_index, extension="emb")
        delete_earliest_buffer_file(INPUT_BUFFER)

    write_eos(OUTPUT_BUFFER, reason=eos_reason)
    print("[embedding_training] worker completed", file=sys.stderr)

if __name__ == "__main__":
    main()

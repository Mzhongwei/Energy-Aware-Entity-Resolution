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
    is_window_published,
    latest_embedding_checkpoint,
    load_config,
    load_earliest_buffer,
    mark_window_published,
    prune_window_checkpoints,
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
    startup_timeout, poll_interval = get_incremental_wait_config(config)
    model_path = os.path.join(get_model_directory(config, "embedding"), MODEL_FILE_NAME)
    checkpoint_dir = os.path.join(os.path.dirname(model_path), "incremental_checkpoints")
    latest_checkpoint = latest_embedding_checkpoint(checkpoint_dir)
    checkpoint_window = latest_checkpoint[0] if latest_checkpoint else -1
    checkpoint_model_path = latest_checkpoint[1] if latest_checkpoint else model_path
    model = load_or_create_model(config, checkpoint_model_path)
    if latest_checkpoint:
        model.save(model_path)

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
                f"[embedding_training] startup timed out after {startup_timeout}s waiting for {INPUT_BUFFER}"
            )
        seen_first_item = True
        sequences = load_earliest_buffer(INPUT_BUFFER)
        if sequences is None:
            write_eos(OUTPUT_BUFFER)
            print("[embedding_training] worker completed after upstream EOS", file=sys.stderr)
            return
        if not sequences:
            delete_earliest_buffer_file(INPUT_BUFFER)
            continue

        window_index = get_earliest_window_index(INPUT_BUFFER)

        if window_index <= checkpoint_window:
            if window_index == checkpoint_window and not is_window_published(checkpoint_dir, window_index):
                write_buffer(model, OUTPUT_BUFFER, window_index, extension="emb")
                mark_window_published(checkpoint_dir, window_index)
            del sequences
            delete_earliest_buffer_file(INPUT_BUFFER)
            prune_window_checkpoints(checkpoint_dir, checkpoint_window)
            continue

        model = train_embeddings(config, model, sequences)
        del sequences
        model.save(model_path)

        write_buffer(model, checkpoint_dir, window_index, extension="emb")
        write_buffer(model, OUTPUT_BUFFER, window_index, extension="emb")
        mark_window_published(checkpoint_dir, window_index)
        delete_earliest_buffer_file(INPUT_BUFFER)
        checkpoint_window = window_index
        prune_window_checkpoints(checkpoint_dir, checkpoint_window)

    print("[embedding_training] worker stopped without emitting EOS", file=sys.stderr)

if __name__ == "__main__":
    main()

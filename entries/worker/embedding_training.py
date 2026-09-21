import argparse
import os

from utils.pipeline_io import (
    BufferIO,
    StageStop,
    StreamStage,
    get_model_directory,
    is_empty_payload,
    is_window_published,
    latest_embedding_checkpoint,
    load_config,
    mark_window_published,
    prune_window_checkpoints,
    wait_for_window_checkpoint_ack,
    write_buffer,
    write_checkpoint_reference,
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


def main():
    parser = argparse.ArgumentParser(description="Worker entry for incremental embedding training.")
    parser.add_argument("--config", default="/app/config/examples/config-embedding.yaml")
    parser.add_argument("--workload", default="default")
    args = parser.parse_args()

    config = load_config(args.config)
    io = BufferIO(args.workload, INPUT_DATA_TYPE, OUTPUT_DATA_TYPE, config)
    model_path = os.path.join(get_model_directory(config, "embedding"), MODEL_FILE_NAME)
    checkpoint_dir = os.path.join(os.path.dirname(model_path), "incremental_checkpoints")

    def remove_batch_seed():
        for seed_path in (model_path, f"{model_path}.meta.json"):
            if os.path.isfile(seed_path):
                os.remove(seed_path)

    latest_checkpoint = latest_embedding_checkpoint(checkpoint_dir)
    checkpoint_window = latest_checkpoint[0] if latest_checkpoint else -1
    checkpoint_model_path = latest_checkpoint[1] if latest_checkpoint else model_path
    model = load_or_create_model(config, checkpoint_model_path)
    if latest_checkpoint:
        write_checkpoint_reference(os.path.dirname(model_path), "current", checkpoint_model_path, checkpoint_window)
        remove_batch_seed()
        prune_window_checkpoints(checkpoint_dir, checkpoint_window)

    def wait_for_previous_window_ack():
        # Do not train over the model while the previous snapshot is still being read downstream.
        if (
            checkpoint_window >= 0
            and is_window_published(checkpoint_dir, checkpoint_window)
            and not wait_for_window_checkpoint_ack(
                checkpoint_dir, checkpoint_window,
                should_stop=stage.should_stop, poll_interval_seconds=io.poll_interval,
            )
        ):
            raise StageStop

    stage = StreamStage(
        "embedding_training", io, is_empty=is_empty_payload, before_load=wait_for_previous_window_ack,
    )

    def process(window):
        nonlocal model, checkpoint_window, checkpoint_model_path
        sequences = window.take()
        window_index = window.index

        if window_index <= checkpoint_window:
            # Replay after a restart: re-publish the checkpoint reference if it was lost.
            if window_index == checkpoint_window and not is_window_published(checkpoint_dir, window_index):
                write_checkpoint_reference(io.output_dir(), window_index, checkpoint_model_path, checkpoint_window)
                mark_window_published(checkpoint_dir, window_index)
            prune_window_checkpoints(checkpoint_dir, checkpoint_window)
            return

        model = train_embeddings(config, model, sequences)
        del sequences

        checkpoint_model_path = write_buffer(model, checkpoint_dir, window_index, extension="emb")
        write_checkpoint_reference(os.path.dirname(model_path), "current", checkpoint_model_path, window_index)
        write_checkpoint_reference(io.output_dir(), window_index, checkpoint_model_path, window_index)
        mark_window_published(checkpoint_dir, window_index)
        remove_batch_seed()
        checkpoint_window = window_index
        prune_window_checkpoints(checkpoint_dir, checkpoint_window)

    stage.run(process)


if __name__ == "__main__":
    main()

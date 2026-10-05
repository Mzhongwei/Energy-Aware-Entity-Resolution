import argparse
import os

from utils.pipeline_io import (
    BufferIO,
    StageStop,
    StreamStage,
    acknowledge_window_checkpoint,
    get_buffer_directory,
    get_model_directory,
    is_window_checkpoint_acknowledged,
    load_checkpoint_reference,
    load_config,
    transferring,
    wait_for_checkpoint_reference,
    waiting,
)
from models.embedding_model import EmbeddingModel
from pipeline.calculating_similarity import (
    get_mutual_top_k,
    get_similarity_top_k,
    score_mutual_topk_candidate_pairs,
)
from pipeline.decision_making import decide_matches, expand_matches
from utils.utils import load_scored_pairs, serialize_scored_pairs, write_bytes_atomically

"""
task: similarity calculation + decision making
mode: incremental + embedding
input: candidate pairs [buffer], embedding model snapshot [buffer]
output: decision event, triggering evaluation [buffer]
description: reads an immutable per-window embedding checkpoint by reference, loads it once
to score the window's candidate pairs (mutual top-k) and to rescore them merged with the
previous decision. The decision is written once as CSV and stored twice: at a fixed path,
overwritten so it always reflects the most recent decision (for external inspection and
restarts), and as a per-window snapshot referenced in the buffer event, so evaluation --
which may run concurrently -- evaluates the exact matches produced for that window.
Acknowledging the checkpoint lets embedding training move on to the next window.
"""

INPUT_DATA_TYPE = "candidate_pairs"
EMBEDDING_INPUT_DATA_TYPE = "embedding_calculating"
OUTPUT_DATA_TYPE = "predicted_matching"
TASK_CONFIG_KEY = "calculating_similarity"
DECISION_CONFIG_KEY = "decision_making"
PREDICTED_MATCH_FILE_NAME = "predicted_matching.csv"
SNAPSHOT_FILE_NAME_TEMPLATE = "predicted_matching_window_{window_index}.csv"


def main():
    parser = argparse.ArgumentParser(description="Worker entry for incremental similarity calculation and decision making.")
    parser.add_argument("--config", default="/app/config/examples/config-embedding.yaml")
    parser.add_argument("--workload", default="default")
    args = parser.parse_args()

    config = load_config(args.config)
    task_config = config.get(TASK_CONFIG_KEY, {}) or {}
    decision_config = config.get(DECISION_CONFIG_KEY, {}) or {}
    similarity_top_k = get_similarity_top_k(config)
    decision_top_k = get_mutual_top_k(config)
    chunk_size = int(task_config.get("chunk_size", 2048))
    min_similarity = decision_config.get("min_similarity")
    io = BufferIO(args.workload, INPUT_DATA_TYPE, OUTPUT_DATA_TYPE, config)
    stage = StreamStage("calculating_similarity", io)
    embedding_buffer = get_buffer_directory(args.workload, EMBEDDING_INPUT_DATA_TYPE)
    snapshot_dir = os.path.join(io.output_dir(), "snapshots")
    os.makedirs(snapshot_dir, exist_ok=True)
    checkpoint_dir = os.path.join(get_model_directory(config, "embedding"), "incremental_checkpoints")
    predicted_match_path = os.path.join(get_model_directory(config, "predicted_match"), PREDICTED_MATCH_FILE_NAME)
    os.makedirs(os.path.dirname(predicted_match_path), exist_ok=True)

    previous_pairs = (
        load_scored_pairs(config, predicted_match_path)
        if os.path.isfile(predicted_match_path)
        else None
    )
    if previous_pairs:
        print(f"[calculating_similarity] restored {len(previous_pairs)} previous pairs from {predicted_match_path}")

    def release_checkpoint(window_index):
        acknowledge_window_checkpoint(checkpoint_dir, window_index)
        reference_path = os.path.join(embedding_buffer, f"{window_index}.json")
        if os.path.isfile(reference_path):
            os.remove(reference_path)

    def process(window):
        nonlocal previous_pairs
        candidate_pairs = window.take()
        window_index = window.index
        if is_window_checkpoint_acknowledged(checkpoint_dir, window_index):
            # Replay after a restart: this window's decision was already published.
            release_checkpoint(window_index)
            return

        with waiting():
            reference_path = wait_for_checkpoint_reference(
                embedding_buffer, window_index, timeout_seconds=None,
                should_stop=stage.should_stop, poll_interval_seconds=io.poll_interval,
            )
        if reference_path is None:
            raise StageStop
        if os.path.basename(reference_path).startswith("eos_"):
            raise RuntimeError(
                f"[calculating_similarity] embedding stream ended before window {window_index} was produced"
            )

        if candidate_pairs:
            embedding_path = load_checkpoint_reference(reference_path)["checkpoint_path"]
            with transferring("read", "embedding") as transfer:
                transfer.path = os.path.dirname(embedding_path)
                model = EmbeddingModel.load(embedding_path)
            matching_pairs = score_mutual_topk_candidate_pairs(
                model, candidate_pairs, top_k=similarity_top_k, chunk_size=chunk_size,
            )
            del candidate_pairs
            if matching_pairs:
                previous_pairs = decide_matches(
                    matching_pairs, previous_pairs, model,
                    top_k=decision_top_k, min_similarity=min_similarity, chunk_size=chunk_size,
                )
                matches = expand_matches(config, previous_pairs, decision_top_k)
                data = serialize_scored_pairs(matches)
                write_bytes_atomically(predicted_match_path, data)
                snapshot_path = os.path.join(snapshot_dir, SNAPSHOT_FILE_NAME_TEMPLATE.format(window_index=window_index))
                with transferring("write", "match_snapshot") as transfer:
                    write_bytes_atomically(snapshot_path, data)
                    transfer.path = snapshot_path
                stage.send(window_index, {"pair_count": len(previous_pairs), "predicted_match_path": snapshot_path})
                del matches, data
            del model, matching_pairs
        release_checkpoint(window_index)

    stage.run(process)


if __name__ == "__main__":
    main()

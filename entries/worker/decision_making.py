import argparse
import os

from utils.pipeline_io import (
    BufferIO,
    StageStop,
    StreamStage,
    acknowledge_window_checkpoint,
    get_buffer_directory,
    get_model_directory,
    is_empty_payload,
    is_window_checkpoint_acknowledged,
    load_checkpoint_reference,
    load_config,
    wait_for_checkpoint_reference,
    waiting,
)
from models.embedding_model import EmbeddingModel
from pipeline.calculating_similarity import get_mutual_top_k
from pipeline.decision_making import decide_matches
from utils.utils import load_scored_pairs_from_graphml

"""
task: decision making
mode: incremental + embedding
input: matching pairs [buffer]
output: decision event, triggering evaluation [buffer]
description: rescores conflicts against the same immutable embedding checkpoint used by
similarity calculation, and overwrites the
predicted matching graph at a fixed path so it always reflects the most recent decision
(for external inspection). Also persists a per-window snapshot under the output buffer and
references it in the buffer event, so evaluation -- which may run concurrently with this
worker -- evaluates the exact graph produced for that window rather than whatever the fixed
path happens to hold when it gets around to reading it.
"""

INPUT_DATA_TYPE = "matching_pairs"
EMBEDDING_INPUT_DATA_TYPE = "embedding_calculating"
OUTPUT_DATA_TYPE = "predicted_matching"
PREDICTED_MATCH_FILE_NAME = "predicted_matching.graphml"
SNAPSHOT_FILE_NAME_TEMPLATE = "predicted_matching_window_{window_index}.graphml"
TASK_CONFIG_KEY = "decision_making"


def main():
    parser = argparse.ArgumentParser(description="Worker entry for incremental decision making.")
    parser.add_argument("--config", default="/app/config/examples/config-embedding.yaml")
    parser.add_argument("--workload", default="default")
    args = parser.parse_args()

    config = load_config(args.config)
    task_config = config.get(TASK_CONFIG_KEY, {}) or {}
    output_format = task_config.get("output_format", "graphml")
    top_k = get_mutual_top_k(config)
    io = BufferIO(args.workload, INPUT_DATA_TYPE, OUTPUT_DATA_TYPE, config)
    # An empty matching-pairs window is *not* skipped by the runner: it still has to
    # acknowledge the embedding checkpoint so the trainer can move on.
    stage = StreamStage("decision_making", io)
    embedding_buffer = get_buffer_directory(args.workload, EMBEDDING_INPUT_DATA_TYPE)
    snapshot_dir = os.path.join(io.output_dir(), "snapshots")
    os.makedirs(snapshot_dir, exist_ok=True)
    checkpoint_dir = os.path.join(get_model_directory(config, "embedding"), "incremental_checkpoints")
    predicted_match_path = os.path.join(get_model_directory(config, "predicted_match"), PREDICTED_MATCH_FILE_NAME)
    os.makedirs(os.path.dirname(predicted_match_path), exist_ok=True)

    previous_pairs = (
        load_scored_pairs_from_graphml(config, predicted_match_path)
        if os.path.isfile(predicted_match_path)
        else None
    )
    if previous_pairs:
        print(f"[decision_making] restored {len(previous_pairs)} previous pairs from {predicted_match_path}")

    def process(window):
        nonlocal previous_pairs
        matching_pairs = window.take()
        window_index = window.index
        reference_path = os.path.join(embedding_buffer, f"{window_index}.json")

        already_acknowledged = is_window_checkpoint_acknowledged(checkpoint_dir, window_index)
        if not already_acknowledged and is_empty_payload(matching_pairs):
            acknowledge_window_checkpoint(checkpoint_dir, window_index)
            already_acknowledged = True
        if already_acknowledged:
            if os.path.isfile(reference_path):
                os.remove(reference_path)
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
                f"[decision_making] embedding stream ended before window {window_index} was produced"
            )
        reference = load_checkpoint_reference(reference_path)
        model = EmbeddingModel.load(reference["checkpoint_path"])
        previous_pairs, predicted_graph = decide_matches(
            mutual_topk_pairs=matching_pairs,
            previous_pairs=previous_pairs,
            model=model,
            output_format=output_format,
            top_k=top_k,
        )
        del model
        del matching_pairs
        del reference
        predicted_graph.export_graphml(predicted_match_path)

        snapshot_path = os.path.join(snapshot_dir, SNAPSHOT_FILE_NAME_TEMPLATE.format(window_index=window_index))
        predicted_graph.export_graphml(snapshot_path)

        stage.send(window_index, {"pair_count": len(previous_pairs), "predicted_match_path": snapshot_path})
        del predicted_graph
        acknowledge_window_checkpoint(checkpoint_dir, window_index)
        if os.path.isfile(reference_path):
            os.remove(reference_path)

    stage.run(process)


if __name__ == "__main__":
    main()

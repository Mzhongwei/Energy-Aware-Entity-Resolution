import argparse
import os

from utils.pipeline_io import (
    BufferIO,
    StageStop,
    StreamStage,
    get_buffer_directory,
    load_checkpoint_reference,
    load_config,
    transferring,
    wait_for_checkpoint_reference,
    waiting,
)
from models.embedding_model import EmbeddingModel
from pipeline.calculating_similarity import get_similarity_top_k, score_mutual_topk_candidate_pairs

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


def main():
    parser = argparse.ArgumentParser(description="Worker entry for incremental similarity calculation.")
    parser.add_argument("--config", default="/app/config/examples/config-embedding.yaml")
    parser.add_argument("--workload", default="default")
    args = parser.parse_args()

    config = load_config(args.config)
    task_config = config.get(TASK_CONFIG_KEY, {}) or {}
    top_k = get_similarity_top_k(config)
    chunk_size = int(task_config.get("chunk_size", 2048))
    io = BufferIO(args.workload, INPUT_DATA_TYPE, OUTPUT_DATA_TYPE, config)
    stage = StreamStage("calculating_similarity", io)
    embedding_buffer = get_buffer_directory(args.workload, EMBEDDING_INPUT_DATA_TYPE)

    def process(window):
        candidate_pairs = window.take()
        with waiting():
            reference_path = wait_for_checkpoint_reference(
                embedding_buffer, window.index, timeout_seconds=None,
                should_stop=stage.should_stop, poll_interval_seconds=io.poll_interval,
            )
        if reference_path is None:
            raise StageStop
        if os.path.basename(reference_path).startswith("eos_"):
            raise RuntimeError(
                f"[calculating_similarity] embedding stream ended before window {window.index} was produced"
            )
        embedding_path = load_checkpoint_reference(reference_path)["checkpoint_path"]

        if candidate_pairs:
            with transferring("read", "embedding") as transfer:
                transfer.path = os.path.dirname(embedding_path)
                model = EmbeddingModel.load(embedding_path)
            matching_pairs = score_mutual_topk_candidate_pairs(
                model, candidate_pairs, top_k=top_k, chunk_size=chunk_size,
            )
            del model
        else:
            matching_pairs = []
        del candidate_pairs
        stage.send(window.index, matching_pairs)
        del matching_pairs

    stage.run(process)


if __name__ == "__main__":
    main()

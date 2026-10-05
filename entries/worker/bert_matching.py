import argparse
import os
import sys

from utils.pipeline_io import BufferIO, StreamStage, get_model_directory, load_config, transferring
from pipeline.bert_matching import BertMatcher
from pipeline.calculating_similarity import get_mutual_top_k
from pipeline.decision_making import expand_matches
from utils.utils import load_scored_pairs, serialize_scored_pairs, write_bytes_atomically

"""
task: BERT matching
mode: incremental + embedding, enabled by bert_matching.enabled
input: decision event [buffer] -- the embedding decision's predicted-matching snapshot
output: decision event with BERT-filtered snapshot, triggering evaluation [buffer]
description: the embedding decision acts as blocking; BERT keeps only the pairs it classifies
as matches. Each distinct pair is judged once and its probability reused in later windows.
Record texts come from the record store written by normalization. Like decision making, it
overwrites a fixed CSV for inspection and hands evaluation a per-window snapshot, then
deletes the decision snapshot it consumed.
"""

INPUT_DATA_TYPE = "predicted_matching"
OUTPUT_DATA_TYPE = "bert_matching"
PREDICTED_MATCH_FILE_NAME = "bert_matching.csv"
SNAPSHOT_FILE_NAME_TEMPLATE = "bert_matching_window_{window_index}.csv"


def main():
    parser = argparse.ArgumentParser(description="Worker entry for incremental BERT matching.")
    parser.add_argument("--config", default="/app/config/examples/config-embedding.yaml")
    parser.add_argument("--workload", default="default")
    args = parser.parse_args()

    config = load_config(args.config)
    top_k = get_mutual_top_k(config)
    io = BufferIO(args.workload, INPUT_DATA_TYPE, OUTPUT_DATA_TYPE, config)
    stage = StreamStage("bert_matching", io)
    matcher = BertMatcher(config)
    snapshot_dir = os.path.join(io.output_dir(), "snapshots")
    os.makedirs(snapshot_dir, exist_ok=True)
    predicted_match_path = os.path.join(get_model_directory(config, "predicted_match"), PREDICTED_MATCH_FILE_NAME)
    os.makedirs(os.path.dirname(predicted_match_path), exist_ok=True)

    def process(window):
        decision_event = window.take()
        source_path = decision_event.get("predicted_match_path") if isinstance(decision_event, dict) else None
        if not source_path or not os.path.exists(source_path):
            print(f"[bert_matching] no snapshot for event {decision_event}; skipping", file=sys.stderr)
            return

        with transferring("read", "match_snapshot") as transfer:
            transfer.path = source_path
            scored_pairs = load_scored_pairs(config, source_path)
        accepted = matcher.filter_pairs(scored_pairs)
        data = serialize_scored_pairs(expand_matches(config, accepted, top_k))
        write_bytes_atomically(predicted_match_path, data)
        snapshot_path = os.path.join(snapshot_dir, SNAPSHOT_FILE_NAME_TEMPLATE.format(window_index=window.index))
        with transferring("write", "match_snapshot") as transfer:
            write_bytes_atomically(snapshot_path, data)
            transfer.path = snapshot_path
        stage.send(window.index, {"pair_count": len(accepted), "predicted_match_path": snapshot_path})
        os.remove(source_path)

    stage.run(process)


if __name__ == "__main__":
    main()

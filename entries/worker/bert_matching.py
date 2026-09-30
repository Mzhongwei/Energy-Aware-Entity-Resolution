import argparse
import os
import sys

from utils.pipeline_io import BufferIO, StreamStage, get_model_directory, load_config
from pipeline.bert_matching import BertMatcher
from pipeline.calculating_similarity import get_mutual_top_k
from pipeline.decision_making import build_similarity_graph
from utils.utils import load_scored_pairs_from_graphml

"""
task: BERT matching
mode: incremental + embedding, enabled by bert_matching.enabled
input: decision event [buffer] -- the embedding decision's predicted-matching snapshot
output: decision event with BERT-filtered snapshot, triggering evaluation [buffer]
description: the embedding decision acts as blocking; BERT keeps only the pairs it classifies
as matches. Each distinct pair is judged once and its probability reused in later windows.
Record texts come from the record store written by normalization. Like decision making, it
overwrites a fixed graphml for inspection and hands evaluation a per-window snapshot, then
deletes the decision snapshot it consumed.
"""

INPUT_DATA_TYPE = "predicted_matching"
OUTPUT_DATA_TYPE = "bert_matching"
PREDICTED_MATCH_FILE_NAME = "bert_matching.graphml"
SNAPSHOT_FILE_NAME_TEMPLATE = "bert_matching_window_{window_index}.graphml"


def main():
    parser = argparse.ArgumentParser(description="Worker entry for incremental BERT matching.")
    parser.add_argument("--config", default="/app/config/examples/config-embedding.yaml")
    parser.add_argument("--workload", default="default")
    args = parser.parse_args()

    config = load_config(args.config)
    output_format = config.get("decision_making", {}).get("output_format", "graphml")
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

        accepted = matcher.filter_pairs(load_scored_pairs_from_graphml(config, source_path))
        graph = build_similarity_graph(accepted, output_format=output_format, top_k=top_k)
        graph.export_graphml(predicted_match_path)
        snapshot_path = os.path.join(snapshot_dir, SNAPSHOT_FILE_NAME_TEMPLATE.format(window_index=window.index))
        graph.export_graphml(snapshot_path)
        stage.send(window.index, {"pair_count": len(accepted), "predicted_match_path": snapshot_path})
        os.remove(source_path)

    stage.run(process)


if __name__ == "__main__":
    main()

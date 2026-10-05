import argparse
import os
import sys

from utils.pipeline_io import (
    BufferIO,
    StreamStage,
    get_transfer_data_directory,
    is_bert_matching_enabled,
    load_config,
    write_step_output,
)
from pipeline.evaluation import compare_ground_truth

"""
task: evaluation
mode: incremental + embedding
input: decision event [buffer] -- from bert_matching when bert_matching.enabled, else from
decision_making
output: evaluation report [transfer]
description: triggered by every upstream decision window; each event carries the path to that
window's predicted-matching snapshot (see decision_making.py / bert_matching.py), so
evaluation reads the exact graph produced for this window even when running concurrently with
its producer, then deletes the snapshot. The report itself is overwritten every time, so only the latest
evaluation is kept.
"""

INPUT_DATA_TYPE = "predicted_matching"
BERT_INPUT_DATA_TYPE = "bert_matching"
REPORT_DATA_TYPE = "report"
REPORT_FILE_NAME = "evaluation_report"
REPORT_EXTENSION = "json"


def main():
    parser = argparse.ArgumentParser(description="Worker entry for incremental evaluation.")
    parser.add_argument("--config", default="/app/config/examples/config-embedding.yaml")
    parser.add_argument("--workload", default="default")
    args = parser.parse_args()

    config = load_config(args.config)
    # Evaluation is the end of the chain: it forwards no EOS, only writes the report.
    input_type = BERT_INPUT_DATA_TYPE if is_bert_matching_enabled(config) else INPUT_DATA_TYPE
    stage = StreamStage("evaluation", BufferIO(args.workload, input_type, None, config))

    def process(window):
        decision_event = window.take()
        snapshot_path = decision_event.get("predicted_match_path") if isinstance(decision_event, dict) else None
        if not snapshot_path or not os.path.exists(snapshot_path):
            print(f"[evaluation] no snapshot for event {decision_event}; skipping", file=sys.stderr)
            return
        del decision_event

        result = compare_ground_truth(config, similarity_file=snapshot_path)
        write_step_output(
            directory=get_transfer_data_directory(args.workload, REPORT_DATA_TYPE),
            file_name=REPORT_FILE_NAME,
            extension=REPORT_EXTENSION,
            output=result,
        )
        del result
        os.remove(snapshot_path)

    stage.run(process)


if __name__ == "__main__":
    main()

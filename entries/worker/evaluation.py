import argparse
import os
import signal
import sys

from utils.pipeline_io import (
    delete_earliest_buffer_file,
    get_buffer_directory,
    get_model_directory,
    get_transfer_data_directory,
    load_config,
    load_earliest_buffer,
    wait_for_buffer,
    write_step_output,
)
from pipeline.evaluation import compare_ground_truth

"""
task: evaluation
mode: incremental + embedding
input: decision event [buffer]
output: evaluation report [transfer]
description: triggered by every decision_making window; re-evaluates against the just-updated
predicted matching graph and overwrites the report, so only the latest evaluation is kept.
"""

INPUT_DATA_TYPE = "predicted_matching"
PREDICTED_MATCH_FILE_NAME = "predicted_matching.graphml"
REPORT_DATA_TYPE = "report"
REPORT_FILE_NAME = "evaluation_report"
REPORT_EXTENSION = "json"

stop_requested = False

def handle_sigterm(signum, frame):
    global stop_requested
    stop_requested = True

signal.signal(signal.SIGTERM, handle_sigterm)
signal.signal(signal.SIGINT, handle_sigterm)

def main():
    parser = argparse.ArgumentParser(description="Worker entry for incremental evaluation.")
    parser.add_argument("--config", default="/app/config/examples/config-embedding.yaml")
    parser.add_argument("--workload", default="default")
    args = parser.parse_args()

    INPUT_BUFFER = get_buffer_directory(args.workload, INPUT_DATA_TYPE)

    config = load_config(args.config)
    config["output_format"] = config.get("decision_making", {}).get("output_format", "graphml")
    predicted_match_path = os.path.join(get_model_directory(config, "predicted_match"), PREDICTED_MATCH_FILE_NAME)

    while not stop_requested:
        ready = wait_for_buffer(INPUT_BUFFER, timeout_seconds=120)
        if ready is None:
            break
        decision_event = load_earliest_buffer(INPUT_BUFFER)
        if decision_event is None:
            break

        result = compare_ground_truth(config, similarity_file=predicted_match_path)
        write_step_output(
            directory=get_transfer_data_directory(args.workload, REPORT_DATA_TYPE),
            file_name=REPORT_FILE_NAME,
            extension=REPORT_EXTENSION,
            output=result,
        )

        delete_earliest_buffer_file(INPUT_BUFFER)

    print("[evaluation] worker completed", file=sys.stderr)

if __name__ == "__main__":
    main()

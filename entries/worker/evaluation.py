import argparse
import os
import signal
import sys

from utils.pipeline_io import (
    delete_earliest_buffer_file,
    get_buffer_directory,
    get_incremental_wait_config,
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
description: triggered by every decision_making window; each event carries the path to that
window's predicted-matching snapshot (see decision_making.py), so evaluation reads the exact
graph produced for this window even when running concurrently with decision_making, then
deletes the snapshot. The report itself is overwritten every time, so only the latest
evaluation is kept.
"""

INPUT_DATA_TYPE = "predicted_matching"
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
    startup_timeout, poll_interval = get_incremental_wait_config(config)

    seen_first_event = False
    while not stop_requested:
        timeout = None if seen_first_event else startup_timeout
        ready = wait_for_buffer(
            INPUT_BUFFER, timeout_seconds=timeout, should_stop=lambda: stop_requested,
            poll_interval_seconds=poll_interval,
        )
        if ready is None:
            if stop_requested:
                return
            raise TimeoutError(f"[evaluation] startup timed out after {startup_timeout}s waiting for {INPUT_BUFFER}")
        seen_first_event = True
        decision_event = load_earliest_buffer(INPUT_BUFFER)
        if decision_event is None:
            print("[evaluation] worker completed after upstream EOS", file=sys.stderr)
            return

        snapshot_path = decision_event.get("predicted_match_path") if decision_event else None
        if not snapshot_path or not os.path.exists(snapshot_path):
            print(f"[evaluation] no snapshot for event {decision_event}; skipping", file=sys.stderr)
            delete_earliest_buffer_file(INPUT_BUFFER)
            continue

        result = compare_ground_truth(config, similarity_file=snapshot_path)
        write_step_output(
            directory=get_transfer_data_directory(args.workload, REPORT_DATA_TYPE),
            file_name=REPORT_FILE_NAME,
            extension=REPORT_EXTENSION,
            output=result,
        )

        os.remove(snapshot_path)
        delete_earliest_buffer_file(INPUT_BUFFER)

    print("[evaluation] worker stopped", file=sys.stderr)

if __name__ == "__main__":
    main()

import argparse
import os
import signal
import sys

from utils.pipeline_io import (
    delete_earliest_buffer_file,
    get_buffer_directory,
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

# The first decision event can take much longer than steady-state windows to arrive --
# candidate_enumeration/calculating_similarity/decision_making all have to produce their
# first output first. Steady-state windows only need a short idle timeout to detect that
# the upstream stream has ended.
DEFAULT_FIRST_WAIT_TIMEOUT_SECONDS = 1800
DEFAULT_WAIT_TIMEOUT_SECONDS = 600

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
    eval_config = config.get("evaluation", {}) or {}
    first_wait_timeout = int(eval_config.get("first_wait_timeout_seconds", DEFAULT_FIRST_WAIT_TIMEOUT_SECONDS))
    wait_timeout = int(eval_config.get("wait_timeout_seconds", DEFAULT_WAIT_TIMEOUT_SECONDS))

    seen_first_event = False
    while not stop_requested:
        timeout = wait_timeout if seen_first_event else first_wait_timeout
        ready = wait_for_buffer(INPUT_BUFFER, timeout_seconds=timeout, should_stop=lambda: stop_requested)
        if ready is None:
            if not stop_requested:
                print(
                    f"[WARNING] [evaluation] timed out after {timeout}s waiting for {INPUT_BUFFER} "
                    "with no upstream EOS seen; exiting as if stream ended (possible silent data loss upstream).",
                    file=sys.stderr,
                )
            break
        seen_first_event = True
        decision_event = load_earliest_buffer(INPUT_BUFFER)
        if decision_event is None:
            break

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

    print("[evaluation] worker completed", file=sys.stderr)

if __name__ == "__main__":
    main()

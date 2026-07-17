import argparse
import signal
import sys

from utils.pipeline_io import (
    delete_earliest_buffer_file,
    get_buffer_directory,
    get_earliest_window_index,
    load_config,
    load_earliest_buffer,
    wait_for_buffer,
    write_buffer,
    write_eos,
)
from pipeline.cg_feature_extraction import compute_features

"""
task: construct feature
mode: incremental + embedding
input: processed data[buffer]
output: feature[buffer]
"""

INPUT_DATA_TYPE = "processed_data_feature"
OUTPUT_DATA_TYPE = "cg_feature"
TASK_CONFIG_KEY = "cg_feature_extraction"

# See normalization_embedding.py for why first/steady waits are split and what a timeout
# (vs an explicit upstream EOS) means.
DEFAULT_FIRST_WAIT_TIMEOUT_SECONDS = 1800
DEFAULT_WAIT_TIMEOUT_SECONDS = 120

stop_requested = False


def handle_sigterm(signum, frame):
    global stop_requested
    stop_requested = True


signal.signal(signal.SIGTERM, handle_sigterm)
signal.signal(signal.SIGINT, handle_sigterm)


def main():
    parser = argparse.ArgumentParser(description="Worker entry for CG feature extraction.")
    parser.add_argument("--config", default="/app/config/examples/config-embedding.yaml")
    parser.add_argument("--workload", default="default")
    args = parser.parse_args()
    INPUT_BUFFER = get_buffer_directory(args.workload, INPUT_DATA_TYPE)
    OUTPUT_BUFFER = get_buffer_directory(args.workload, OUTPUT_DATA_TYPE)

    config = load_config(args.config)
    task_config = config.get(TASK_CONFIG_KEY, {}) or {}
    first_wait_timeout = int(task_config.get("first_wait_timeout_seconds", DEFAULT_FIRST_WAIT_TIMEOUT_SECONDS))
    wait_timeout = int(task_config.get("wait_timeout_seconds", DEFAULT_WAIT_TIMEOUT_SECONDS))

    seen_first_item = False
    eos_reason = "stream_completed"
    while not stop_requested:
        timeout = wait_timeout if seen_first_item else first_wait_timeout
        ready = wait_for_buffer(INPUT_BUFFER, timeout_seconds=timeout, should_stop=lambda: stop_requested)
        if ready is None:
            if not stop_requested:
                print(
                    f"[WARNING] [cg_feature_extraction] timed out after {timeout}s waiting for {INPUT_BUFFER} "
                    "with no upstream EOS seen; exiting as if stream ended (possible silent data loss upstream).",
                    file=sys.stderr,
                )
                eos_reason = "timeout_no_upstream_eos"
            break

        seen_first_item = True
        processed_data = load_earliest_buffer(INPUT_BUFFER)
        if processed_data is None:
            break
        if processed_data.empty:
            delete_earliest_buffer_file(INPUT_BUFFER)
            continue

        window_index = get_earliest_window_index(INPUT_BUFFER)
        method = config.get("candidate_generation", {}).get("method", "fullindexing")
        cg_feature = compute_features(processed_data, method, config)
        if cg_feature is not None:
            write_buffer(cg_feature, OUTPUT_BUFFER, window_index, extension="json")

        delete_earliest_buffer_file(INPUT_BUFFER)

    write_eos(OUTPUT_BUFFER, reason=eos_reason)
    print("[cg_feature_extraction] worker completed", file=sys.stderr)


if __name__ == "__main__":
    main()

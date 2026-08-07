import argparse
import signal
import sys

from utils.pipeline_io import (
    delete_earliest_buffer_file,
    get_buffer_directory,
    get_earliest_window_index,
    get_incremental_wait_config,
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
    startup_timeout, poll_interval = get_incremental_wait_config(config)

    seen_first_item = False
    while not stop_requested:
        timeout = None if seen_first_item else startup_timeout
        ready = wait_for_buffer(
            INPUT_BUFFER,
            timeout_seconds=timeout,
            should_stop=lambda: stop_requested,
            poll_interval_seconds=poll_interval,
        )
        if ready is None:
            if stop_requested:
                return
            raise TimeoutError(
                f"[cg_feature_extraction] startup timed out after {startup_timeout}s waiting for {INPUT_BUFFER}"
            )

        seen_first_item = True
        processed_data = load_earliest_buffer(INPUT_BUFFER)
        if processed_data is None:
            write_eos(OUTPUT_BUFFER)
            print("[cg_feature_extraction] worker completed after upstream EOS", file=sys.stderr)
            return
        if processed_data.empty:
            delete_earliest_buffer_file(INPUT_BUFFER)
            continue

        window_index = get_earliest_window_index(INPUT_BUFFER)
        method = config.get("candidate_generation", {}).get("method", "fullindexing")
        cg_feature = compute_features(processed_data, method, config)
        if cg_feature is not None:
            write_buffer(cg_feature, OUTPUT_BUFFER, window_index, extension="json")

        delete_earliest_buffer_file(INPUT_BUFFER)

    print("[cg_feature_extraction] worker stopped without emitting EOS", file=sys.stderr)


if __name__ == "__main__":
    main()

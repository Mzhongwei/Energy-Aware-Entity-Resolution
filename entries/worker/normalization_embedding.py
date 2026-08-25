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
from pipeline.normalization import index_normalization
"""
mode: incremental + embedding
input: raw data[buffer]
output: processed data for feature tasks[buffer], processed data for graph tasks[buffer]
"""

INPUT_DATA_TYPE = "raw_data"
GRAPH_OUTPUT_DATA_TYPE = "processed_data_graph"
FEATURE_OUTPUT_DATA_TYPE = "processed_data_feature"
stop_requested = False

def handle_sigterm(signum, frame):
    """Record termination requests so long-running buffer loops can stop cleanly.

    When a Pod or process receives a stop signal, do not immediately force a termination; instead, first log a “stop request” to give the loop a chance to wrap up and exit on its own.
    """
    global stop_requested
    stop_requested = True

signal.signal(signal.SIGTERM, handle_sigterm)
signal.signal(signal.SIGINT, handle_sigterm)

def main():
    """
    k8s jobs entry for mode [incremental][embedding]
    We don't accecpt training in incremental embedding mode
    """
    # parse config
    parser = argparse.ArgumentParser(description="Worker entry for embedding normalization.")
    parser.add_argument("--config", default="/app/config/examples/config-embedding.yaml")
    parser.add_argument("--workload", default="default")
    args = parser.parse_args()
    # input 
    RAW_BUFFER = get_buffer_directory(args.workload, INPUT_DATA_TYPE)
    # output
    GRAPH_BUFFER = get_buffer_directory(args.workload, GRAPH_OUTPUT_DATA_TYPE)
    FEATURE_BUFFER = get_buffer_directory(args.workload, FEATURE_OUTPUT_DATA_TYPE)

    # load configuration
    config = load_config(args.config)
    startup_timeout, poll_interval = get_incremental_wait_config(config)

    print(f"[normalization] get raw buffer from {RAW_BUFFER}", file=sys.stderr)
    seen_first_item = False
    while not stop_requested:
        timeout = None if seen_first_item else startup_timeout
        ready = wait_for_buffer(
            RAW_BUFFER,
            timeout_seconds=timeout,
            should_stop=lambda: stop_requested,
            poll_interval_seconds=poll_interval,
        )
        if ready is None:
            if stop_requested:
                return
            raise TimeoutError(f"[normalization] startup timed out after {startup_timeout}s waiting for {RAW_BUFFER}")
        seen_first_item = True
        raw_data = load_earliest_buffer(RAW_BUFFER)
        if raw_data is None:
            write_eos(GRAPH_BUFFER)
            write_eos(FEATURE_BUFFER)
            print("[normalization] worker completed after upstream EOS", file=sys.stderr)
            return
        if raw_data.empty:
            delete_earliest_buffer_file(RAW_BUFFER)
            continue

        print(
            f"[normalization_distribution] window_index={get_earliest_window_index(RAW_BUFFER)} "
            f"raw_type={type(raw_data).__name__} rows={len(raw_data)}",
            file=sys.stderr,
        )
        returned = index_normalization(config=config, raw_data=raw_data, raw_data_path=None, is_training=False)    
        del raw_data
        window_index = get_earliest_window_index(RAW_BUFFER)

        if returned is not None:
            write_buffer(returned, GRAPH_BUFFER, window_index, extension="csv")
            write_buffer(returned, FEATURE_BUFFER, window_index, extension="csv")
        del returned

        delete_earliest_buffer_file(RAW_BUFFER)
    print("[normalization] worker stopped without emitting EOS", file=sys.stderr)

if __name__ == "__main__":
    main()

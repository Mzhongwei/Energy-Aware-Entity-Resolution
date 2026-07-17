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
from pipeline.normalization import index_normalization
"""
mode: incremental + embedding
input: raw data[buffer]
output: processed data for feature tasks[buffer], processed data for graph tasks[buffer]
"""

INPUT_DATA_TYPE = "raw_data"
GRAPH_OUTPUT_DATA_TYPE = "processed_data_graph"
FEATURE_OUTPUT_DATA_TYPE = "processed_data_feature"
TASK_CONFIG_KEY = "normalization"

# The first raw_data window depends on producer/consumer startup and Kafka delivering the
# first batch, which can take much longer than steady-state windows. A timeout here (first
# or steady) with no upstream EOS is logged as a WARNING and treated as end-of-stream, but
# it's ambiguous -- upstream may just be slow rather than actually done.
DEFAULT_FIRST_WAIT_TIMEOUT_SECONDS = 1800
DEFAULT_WAIT_TIMEOUT_SECONDS = 120

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
    task_config = config.get(TASK_CONFIG_KEY, {}) or {}
    first_wait_timeout = int(task_config.get("first_wait_timeout_seconds", DEFAULT_FIRST_WAIT_TIMEOUT_SECONDS))
    wait_timeout = int(task_config.get("wait_timeout_seconds", DEFAULT_WAIT_TIMEOUT_SECONDS))

    print(f"[normalization] get raw buffer from {RAW_BUFFER}", file=sys.stderr)
    seen_first_item = False
    eos_reason = "stream_completed"
    while not stop_requested:
        timeout = wait_timeout if seen_first_item else first_wait_timeout
        ready = wait_for_buffer(RAW_BUFFER, timeout_seconds=timeout, should_stop=lambda: stop_requested)
        if ready is None:
            if not stop_requested:
                print(
                    f"[WARNING] [normalization] timed out after {timeout}s waiting for {RAW_BUFFER} "
                    "with no upstream EOS seen; exiting as if stream ended (possible silent data loss upstream).",
                    file=sys.stderr,
                )
                eos_reason = "timeout_no_upstream_eos"
            break
        seen_first_item = True
        raw_data = load_earliest_buffer(RAW_BUFFER)
        if raw_data is None:
            break
        if raw_data.empty:
            delete_earliest_buffer_file(RAW_BUFFER)
            continue

        print(
            f"[normalization_distribution] window_index={get_earliest_window_index(RAW_BUFFER)} "
            f"raw_type={type(raw_data).__name__} rows={len(raw_data)}",
            file=sys.stderr,
        )
        returned = index_normalization(config=config, raw_data=raw_data, raw_data_path=None, is_training=False)    
        window_index = get_earliest_window_index(RAW_BUFFER)

        if returned is not None:
            write_buffer(returned, GRAPH_BUFFER, window_index, extension="csv")
            write_buffer(returned, FEATURE_BUFFER, window_index, extension="csv")

        delete_earliest_buffer_file(RAW_BUFFER)
    # close buffers
    write_eos(GRAPH_BUFFER, reason=eos_reason)
    write_eos(FEATURE_BUFFER, reason=eos_reason)

if __name__ == "__main__":
    main()

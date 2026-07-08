import argparse
import signal
import sys

from utils.config_io import load_config
from utils.buffers import wait_for_buffer, write_eos, load_earliest_buffer, get_earliest_window_index, write_buffer, delete_earliest_buffer_file
from pipeline.normalization import index_normalization
"""
mode: incremental + embedding
"""

BUFFER_PATH = "/app/data/buffers/"
# input 
RAW_BUFFER = f"{BUFFER_PATH}/incremental/embedding/raw_data"
# output
GRAPH_BUFFER = f"{BUFFER_PATH}/incremental/embedding/processed_data_graph"
FEATURE_BUFFER = f"{BUFFER_PATH}/incremental/embedding/processed_data_feature"

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
    We don't accecpt training in embedding mode
    """
    # parse config
    parser = argparse.ArgumentParser(description="Worker entry for embedding normalization.")
    parser.add_argument("--config", default="/app/config/examples/config-embedding.yaml")
    parser.add_argument("--mode", default="embedding-inference-inc")
    args = parser.parse_args()
    # load configuration
    config = load_config(args.config)
    
    print(f"[normalization] get raw buffer from {RAW_BUFFER}", file=sys.stderr)
    while not stop_requested:
        ready = wait_for_buffer(RAW_BUFFER, timeout_seconds=120)
        if ready is None:
            break
        raw_data = load_earliest_buffer(RAW_BUFFER)
        if raw_data is None:
            break
        if raw_data.empty:
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
    write_eos(GRAPH_BUFFER, reason="stream_completed")
    write_eos(FEATURE_BUFFER, reason="stream_completed")

if __name__ == "__main__":
    main()

import argparse
import signal
import sys

from utils.pipeline_io import (
    delete_earliest_buffer_file,
    get_buffer_directory,
    get_earliest_window_index,
    get_model_directory,
    load_config,
    load_earliest_buffer,
    wait_for_buffer,
    write_buffer,
    write_eos,
)
from pipeline.feature_index_construction import build_index, create_cg_index
from pipeline.candidate_enumeration import enumerate_candidates, fetch_candidates
from models.cg_index import CGIndex

"""
task: candidate enumeration
mode: incremental + embedding
input: feature[buffer]
output: candidate[buffer], feature index[model]
description: now we accept only the two datasets comparisons but not deduplication. the pre-training of dataset A is required.
"""

INPUT_DATA_TYPE = "cg_feature"
OUTPUT_DATA_TYPE = "candidate_pairs"

stop_requested = False


def handle_sigterm(signum, frame):
    global stop_requested
    stop_requested = True


signal.signal(signal.SIGTERM, handle_sigterm)
signal.signal(signal.SIGINT, handle_sigterm)


def main():
    parser = argparse.ArgumentParser(description="Worker entry for feature index construction and candidate enumeration.")
    parser.add_argument("--config", default="/app/config/examples/config-embedding.yaml")
    parser.add_argument("--workload", default="default")
    args = parser.parse_args()
    INPUT_BUFFER = get_buffer_directory(args.workload, INPUT_DATA_TYPE)
    OUTPUT_BUFFER = get_buffer_directory(args.workload, OUTPUT_DATA_TYPE)

    config = load_config(args.config)

    while not stop_requested:
        ready = wait_for_buffer(INPUT_BUFFER, timeout_seconds=120)
        if ready is None:
            break

        cg_feature = load_earliest_buffer(INPUT_BUFFER)
        if cg_feature is None:
            break
        if not cg_feature:
            delete_earliest_buffer_file(INPUT_BUFFER)
            continue

        window_index = get_earliest_window_index(INPUT_BUFFER)
        if config.get("candidate_generation", {}).get("status", True):
            # LOAD index 
            index = CGIndex.from_disk(config, get_model_directory(config, "index"))
            # calculate candidate pairs 
            candidate_pairs = enumerate_candidates(cg_feature, index)
        else:
            data_pairs_file = config.get("candidate_generation", {}).get("data_pairs_fixed", "")
            if data_pairs_file:
                print(f"[candidate_enumeration] fetch from file {data_pairs_file}")
                candidate_pairs = fetch_candidates(data_pairs_file)
            else:
                print(f"error: [candidate_enumeration] fetch from file {data_pairs_file}, file not found")
        write_buffer(candidate_pairs, OUTPUT_BUFFER, window_index, extension="json")
        delete_earliest_buffer_file(INPUT_BUFFER)

    write_eos(OUTPUT_BUFFER, reason="stream_completed")
    print("[feature_index_candidate] worker completed", file=sys.stderr)


if __name__ == "__main__":
    main()

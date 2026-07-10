import argparse
import os

from utils.pipeline_io import load_config
from utils.pipeline_io import get_model_directory, get_transfer_data_directory, load_processed_data, write_step_output
from pipeline.candidate_enumeration import enumerate_candidates, fetch_candidates
from models.cg_index import CGIndex

"""
task: candidate enumeration
mode: batch + embedding
input: feature index [model], featrue [buffer]
output: candidate pairs [buffer]
"""

INPUT_DATA_TYPE = "embedding"
INPUT_FILE_NAME = "prediction_cg_feature.json"
OUTPUT_DATA_TYPE = "embedding"
OUTPUT_FILE_NAME = "candidate_pairs"
OUTPUT_EXTENSION = "json"

def main():
    parser = argparse.ArgumentParser(description="Batch entry for candidate enumeration.")
    parser.add_argument("--config", default="/app/config/examples/config-embedding.yaml")
    parser.add_argument("--workload", default="default")
    args = parser.parse_args()

    config = load_config(args.config)
    if config.get("candidate_generation", {}).get("status", True):
        # read data from file
        input_directory = get_transfer_data_directory(args.workload, INPUT_DATA_TYPE)
        cg_feature = load_processed_data(os.path.join(input_directory, INPUT_FILE_NAME))
        # load and complete index
        index = CGIndex.from_disk(config, get_model_directory(config, "index"))

        ## save index not required 
        # index.index_dir = os.path.join(get_transfer_data_directory(args.workload, OUTPUT_DATA_TYPE), INDEX_FILE_NAME)
        # index.persist()

        # get candidates
        candidate_pairs = enumerate_candidates(cg_feature, index)
    else:
        data_pairs_file = config.get("candidate_generation", {}).get("data_pairs_fixed", "")
        if data_pairs_file:
            print(f"[candidate_enumeration] fetch from file {data_pairs_file}")
            candidate_pairs = fetch_candidates(data_pairs_file)
        else:
            print(f"error: [candidate_enumeration] fetch from file {data_pairs_file}, file not found")
    write_step_output(
        directory=get_transfer_data_directory(args.workload, OUTPUT_DATA_TYPE),
        file_name=OUTPUT_FILE_NAME,
        extension=OUTPUT_EXTENSION,
        output=candidate_pairs,
    )

if __name__ == "__main__":
    main()

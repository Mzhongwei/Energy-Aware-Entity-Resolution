import argparse
import os

from utils.pipeline_io import load_config
from utils.pipeline_io import get_model_directory, get_transfer_data_directory, load_processed_data
from pipeline.feature_index_construction import build_index, create_cg_index
"""
task: construct feature index
mode: batch + embedding, training for incrmental jobs
"""
INPUT_DATA_TYPE = "embedding"
INPUT_FILE_NAME = "training_cg_feature.json"

def main():
    parser = argparse.ArgumentParser(description="Batch entry for feature index construction.")
    parser.add_argument("--config", default="/app/config/examples/config-embedding.yaml")
    parser.add_argument("--workload", default="default")
    args = parser.parse_args()

    config = load_config(args.config)
    # read data from file
    input_directory = get_transfer_data_directory(args.workload, INPUT_DATA_TYPE)
    cg_feature = load_processed_data(os.path.join(input_directory, INPUT_FILE_NAME))
    # build index 
    method = config.get("candidate_generation", {}).get("method", "fullindexing")
    index = create_cg_index(method, config)
    build_index(cg_feature, index)
    # save index 
    index.index_dir = get_model_directory(config, "index")
    index.persist()

if __name__ == "__main__":
    main()

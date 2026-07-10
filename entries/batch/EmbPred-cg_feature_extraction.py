import argparse
import os

from utils.pipeline_io import load_config
from utils.pipeline_io import get_transfer_data_directory, load_processed_data, write_step_output
from pipeline.cg_feature_extraction import compute_features

"""
task: feature extraction
mode: batch + embedding
"""

INPUT_DATA_TYPE = "embedding"
INPUT_FILE_NAME = "prediction_processed_data.csv"
OUTPUT_DATA_TYPE = "embedding"
OUTPUT_FILE_NAME = "prediction_cg_feature"
OUTPUT_EXTENSION = "json"

def main():
    parser = argparse.ArgumentParser(description="Batch entry for prediction CG feature extraction.")
    parser.add_argument("--config", default="/app/config/examples/config-embedding.yaml")
    parser.add_argument("--workload", default="default")
    args = parser.parse_args()

    config = load_config(args.config)
    
    input_directory = get_transfer_data_directory(args.workload, INPUT_DATA_TYPE)
    processed_data = load_processed_data(os.path.join(input_directory, INPUT_FILE_NAME))

    cg_feature = None
    # compute features
    if config.get("candidate_generation", {}).get("status", True):
        method = config.get("candidate_generation", {}).get("method", "fullindexing")
        cg_feature = compute_features(processed_data, method, config)

    write_step_output(
        directory=get_transfer_data_directory(args.workload, OUTPUT_DATA_TYPE),
        file_name=OUTPUT_FILE_NAME,
        extension=OUTPUT_EXTENSION,
        output=cg_feature,
    )


if __name__ == "__main__":
    main()

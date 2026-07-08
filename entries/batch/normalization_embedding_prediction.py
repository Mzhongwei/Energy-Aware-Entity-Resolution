import argparse
import os

from utils.config_io import load_config
from pipeline.normalization import index_normalization
from utils.pipeline_io import write_step_output
"""
mode: batch + embedding
"""

TRANSFER_DATA_DIRECTORY = "batch/embedding/"
OUTPUT_FILE_NAME = "prediction_processed_data.csv"

def main():
    parser = argparse.ArgumentParser(description="Batch entry for embedding normalization.")
    parser.add_argument("--config", default="/app/config/examples/config-embedding.yaml")
    parser.add_argument("--run_dir", default="-")
    args = parser.parse_args()
    config = load_config(args.config)
    
    raw_data_path = config.get("data_source_B")
    output = index_normalization(config=config, raw_data_path=raw_data_path, is_training=False)

    output_path = os.path.join(args.run_dir, TRANSFER_DATA_DIRECTORY, OUTPUT_FILE_NAME)
    write_step_output(output_path=output_path, output=output)

if __name__ == "__main__":
    main()

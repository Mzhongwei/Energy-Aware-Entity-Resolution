import argparse

from pipeline.normalization import index_normalization
from utils.pipeline_io import get_transfer_data_directory, load_config, write_step_output
"""
mode: batch + embedding, training for incrmental jobs
"""

OUTPUT_DATA_TYPE = "embedding"
OUTPUT_FILE_NAME = "training_processed_data"
OUTPUT_EXTENSION = "csv"

def main():
    parser = argparse.ArgumentParser(description="Batch entry for embedding normalization.")
    parser.add_argument("--config", default="/app/config/examples/config-embedding.yaml")
    parser.add_argument("--workload", default="default")
    args = parser.parse_args()
    config = load_config(args.config)
    
    raw_data_path = config.get("data_source_A")
    output = index_normalization(config=config, raw_data=None, raw_data_path=raw_data_path, is_training=True)

    write_step_output(
        directory=get_transfer_data_directory(args.workload, OUTPUT_DATA_TYPE),
        file_name=OUTPUT_FILE_NAME,
        extension=OUTPUT_EXTENSION,
        output=output,
    )

if __name__ == "__main__":
    main()

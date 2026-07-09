import argparse
import pandas as pd

from utils.pipeline_io import load_config
from pipeline.normalization import sequence_generating_m1
from utils.pipeline_io import get_transfer_data_directory, write_step_output
"""
mode: batch + bert
"""
OUTPUT_DATA_TYPE = "bert/training_processed_data"

def main():
    parser = argparse.ArgumentParser(description="Batch entry for BERT normalization. Preparing data for training.")
    parser.add_argument("--config", default="/app/config/examples/config-bert.yaml")
    parser.add_argument("--workload", default="default")
    args = parser.parse_args()
    config = load_config(args.config)

    raw_data = {
        "train": pd.read_csv(config.get("trainset_path")),
        "eval": pd.read_csv(config.get("evalset_path")),
    }
   
    for key, df in raw_data.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            raw_data[key] = sequence_generating_m1(df)
    processed_data = raw_data
    # write transfer data
    write_step_output(
        directory=get_transfer_data_directory(args.workload, OUTPUT_DATA_TYPE),
        file_name="train",
        extension="csv",
        output=processed_data["train"],
    )
    write_step_output(
        directory=get_transfer_data_directory(args.workload, OUTPUT_DATA_TYPE),
        file_name="eval",
        extension="csv",
        output=processed_data["eval"],
    )

if __name__ == "__main__":
    main()

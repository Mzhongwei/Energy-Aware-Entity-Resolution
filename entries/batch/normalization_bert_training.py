import argparse
import os

import pandas as pd

from utils.config_io import load_config
from pipeline.normalization import sequence_generating_m1
from utils.pipeline_io import write_step_output
"""
mode: batch + bert
"""
TRANSFER_DATA_DIRECTORY = "batch/bert/"
OUTPUT_DIRECTORY = "training_processed_data"

def main():
    parser = argparse.ArgumentParser(description="Batch entry for BERT normalization. Preparing data for training.")
    parser.add_argument("--config", default="/app/config/examples/config-bert.yaml")
    parser.add_argument("--run_dir", default="/app/data/runs/default")
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
    output_dir = os.path.join(args.run_dir, TRANSFER_DATA_DIRECTORY, OUTPUT_DIRECTORY)
    write_step_output(output_path=os.path.join(output_dir, "train.csv"), output=processed_data["train"])
    write_step_output(output_path=os.path.join(output_dir, "eval.csv"), output=processed_data["eval"])

if __name__ == "__main__":
    main()

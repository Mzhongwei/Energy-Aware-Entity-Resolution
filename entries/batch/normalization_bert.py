import argparse
import os

import pandas as pd

from utils.config_io import load_config
from pipeline.normalization import sequence_generating_m1
from utils.pipeline_io import write_step_output
"""
mode: batch + bert
"""

def main():
    parser = argparse.ArgumentParser(description="Batch entry for BERT normalization.")
    parser.add_argument("--config", default="/app/config/examples/config-bert.yaml")
    parser.add_argument("--mode", default="bert-inference")
    parser.add_argument("--output", default="-")
    args = parser.parse_args()
    config = load_config(args.config)
    mode = args.mode
    output_path = args.output
    os.environ["EAER_CONFIG_PATH"] = args.config

   
    if "training" in mode:
        raw_data = {
            "train": pd.read_csv(config.get("trainset_path")),
            "eval": pd.read_csv(config.get("evalset_path")),
        }
    elif "b_evaluation" in mode:
        raw_data = {
            "test": pd.read_csv(config.get("testset_path")),
        }
    elif "inference" in mode:
        raw_data = {
            "test": pd.read_csv(config.get("testset_path")),
        }
    for key, df in raw_data.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            raw_data[key] = sequence_generating_m1(df)
    processed_data = raw_data
    write_step_output(output_path=output_path, output=processed_data)

if __name__ == "__main__":
    main()
import argparse
import sys
from pathlib import Path


from utils.pipeline_io import load_processed_data
from utils.config_io import load_config
from utils.pipeline_io import serialize_for_json, write_step_output

"""
To complete
"""

def run(config: dict, processed_data):
    pass


def main():
    parser = argparse.ArgumentParser(description="Batch entry for BERT evaluation.")
    parser.add_argument("--config", default="/app/config/examples/config-bert.yaml")
    parser.add_argument("--mode", default="bert-evaluation")
    parser.add_argument("--processed_data_path", required=True)
    parser.add_argument("--output", default="-")
    args = parser.parse_args()

    config = load_config(args.config)
    config["mode"] = args.mode
    processed_data = load_processed_data(args.processed_data_path)

    if not isinstance(processed_data, dict) or "test" not in processed_data:
        raise ValueError("BERT evaluation expects processed_data with a 'test' dataset.")

    output = run(config, processed_data)
    write_step_output(args.output, output, serializer=serialize_for_json)


if __name__ == "__main__":
    main()

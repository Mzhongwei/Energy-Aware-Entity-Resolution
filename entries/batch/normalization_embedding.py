import argparse

from utils.config_io import load_config
from pipeline.normalization import index_normalization
from utils.pipeline_io import write_step_output
"""
mode: batch + embedding
"""

def main():
    parser = argparse.ArgumentParser(description="Batch entry for embedding normalization.")
    parser.add_argument("--config", default="/app/config/examples/config-embedding.yaml")
    parser.add_argument("--mode", default="embedding-evaluation")
    parser.add_argument("--output", default="-")
    args = parser.parse_args()
    config = load_config(args.config)
    output_path = args.output
    if "training" in args.mode:
        raw_data_path = config.get("data_source_A")
        output = index_normalization(config=config, raw_data=None, raw_data_path=raw_data_path, is_training=True)
    else:
        raw_data_path = config.get("data_source_B")
        output = index_normalization(config=config, raw_data_path=raw_data_path, is_training=False)
    write_step_output(output_path=output_path, output=output)

if __name__ == "__main__":
    main()

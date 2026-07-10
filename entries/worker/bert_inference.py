import argparse
import os
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[2]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from utils.pipeline_io import load_config
from utils.pipeline_io import serialize_for_json, write_step_output

"""
disabled: k8s BERT inference worker interface is not implemented yet.
"""

def run(config: dict):
    pass


def main():
    parser = argparse.ArgumentParser(description="Worker entry placeholder for BERT inference.")
    parser.add_argument("--config", default="/app/config/examples/config-bert.yaml")
    parser.add_argument("--mode", default="bert-inference")
    parser.add_argument("--output", default="-")
    args = parser.parse_args()

    config = load_config(args.config)
    config["mode"] = args.mode
    output = run(config)
    output_directory = os.path.dirname(args.output) or "."
    output_base = os.path.basename(args.output)
    file_name, extension = os.path.splitext(output_base)
    write_step_output(output_directory, file_name, extension.lstrip("."), output, serializer=serialize_for_json)


if __name__ == "__main__":
    main()

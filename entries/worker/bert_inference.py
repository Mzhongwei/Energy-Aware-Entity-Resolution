import argparse
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[2]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from utils.config_io import load_config
from utils.pipeline_io import serialize_for_json, write_step_output


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
    write_step_output(args.output, output, serializer=serialize_for_json)


if __name__ == "__main__":
    main()

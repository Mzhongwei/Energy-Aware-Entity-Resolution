import argparse
import sys

from utils.pipeline_io import BufferIO, StreamStage, is_empty_payload, load_config
from pipeline.normalization import index_normalization
"""
mode: incremental + embedding
input: raw data[buffer]
output: processed data for feature tasks[buffer], processed data for graph tasks[buffer]
We don't accept training in incremental embedding mode.
"""

INPUT_DATA_TYPE = "raw_data"
OUTPUTS = {"graph": "processed_data_graph", "feature": "processed_data_feature"}


def main():
    parser = argparse.ArgumentParser(description="Worker entry for embedding normalization.")
    parser.add_argument("--config", default="/app/config/examples/config-embedding.yaml")
    parser.add_argument("--workload", default="default")
    args = parser.parse_args()

    config = load_config(args.config)
    io = BufferIO(args.workload, INPUT_DATA_TYPE, OUTPUTS, config)
    stage = StreamStage("normalization", io, is_empty=is_empty_payload)
    print(f"[normalization] get raw buffer from {io.input_dir}", file=sys.stderr)

    def process(window):
        raw_data = window.take()
        print(
            f"[normalization_distribution] window_index={window.index} "
            f"raw_type={type(raw_data).__name__} rows={len(raw_data)}",
            file=sys.stderr,
        )
        returned = index_normalization(config=config, raw_data=raw_data, raw_data_path=None, is_training=False)
        del raw_data
        if returned is not None:
            stage.send(window.index, returned, "graph", "csv")
            stage.send(window.index, returned, "feature", "csv")
        del returned

    stage.run(process)


if __name__ == "__main__":
    main()

import argparse

from utils.pipeline_io import BufferIO, StreamStage, is_empty_payload, load_config
from pipeline.cg_feature_extraction import compute_features

"""
task: construct feature
mode: incremental + embedding
input: processed data[buffer]
output: feature[buffer]
"""

INPUT_DATA_TYPE = "processed_data_feature"
OUTPUT_DATA_TYPE = "cg_feature"


def main():
    parser = argparse.ArgumentParser(description="Worker entry for CG feature extraction.")
    parser.add_argument("--config", default="/app/config/examples/config-embedding.yaml")
    parser.add_argument("--workload", default="default")
    args = parser.parse_args()

    config = load_config(args.config)
    stage = StreamStage(
        "cg_feature_extraction",
        BufferIO(args.workload, INPUT_DATA_TYPE, OUTPUT_DATA_TYPE, config),
        is_empty=is_empty_payload,
    )
    method = config.get("candidate_generation", {}).get("method", "fullindexing")

    def process(window):
        processed_data = window.take()
        cg_feature = compute_features(processed_data, method, config)
        del processed_data
        if cg_feature is not None:
            stage.send(window.index, cg_feature)
        del cg_feature

    stage.run(process)


if __name__ == "__main__":
    main()

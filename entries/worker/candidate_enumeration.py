import argparse

from utils.pipeline_io import BufferIO, StreamStage, get_model_directory, is_empty_payload, load_config
from pipeline.candidate_enumeration import enumerate_candidates, fetch_candidates
from models.cg_index import CGIndex

"""
task: candidate enumeration
mode: incremental + embedding
input: feature[buffer]
output: candidate[buffer], feature index[model]
description: now we accept only the two datasets comparisons but not deduplication. the pre-training of dataset A is required.
"""

INPUT_DATA_TYPE = "cg_feature"
OUTPUT_DATA_TYPE = "candidate_pairs"


def main():
    parser = argparse.ArgumentParser(description="Worker entry for feature index construction and candidate enumeration.")
    parser.add_argument("--config", default="/app/config/examples/config-embedding.yaml")
    parser.add_argument("--workload", default="default")
    args = parser.parse_args()

    config = load_config(args.config)
    stage = StreamStage(
        "candidate_enumeration",
        BufferIO(args.workload, INPUT_DATA_TYPE, OUTPUT_DATA_TYPE, config),
        is_empty=is_empty_payload,
    )

    def process(window):
        cg_feature = window.take()
        if config.get("candidate_generation", {}).get("status", True):
            index = CGIndex.from_disk(config, get_model_directory(config, "index"))
            candidate_pairs = enumerate_candidates(cg_feature, index)
            del index
        else:
            data_pairs_file = config.get("candidate_generation", {}).get("data_pairs_fixed", "")
            if not data_pairs_file:
                raise FileNotFoundError(
                    "candidate_generation.data_pairs_fixed is required when candidate generation is disabled"
                )
            print(f"[candidate_enumeration] fetch from file {data_pairs_file}")
            candidate_pairs = fetch_candidates(data_pairs_file)
        del cg_feature
        stage.send(window.index, candidate_pairs)
        del candidate_pairs

    stage.run(process)


if __name__ == "__main__":
    main()

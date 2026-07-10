import argparse
import os

from utils.pipeline_io import load_config, get_model_directory, get_transfer_data_directory, load_processed_data
from models.embedding_model import EmbeddingModel
from pipeline.decision_making import decide_matches

"""
task: decision making
mode: batch + embedding
input: matching pairs [transfer], embedding model [model]
output: predicted matching graph [model]
"""

INPUT_DATA_TYPE = "embedding"
INPUT_FILE_NAME = "matching_pairs.json"
MODEL_FILE_NAME = "embedding.emb"
PREDICTED_MATCH_FILE_NAME = "predicted_matching.graphml"

def main():
    parser = argparse.ArgumentParser(description="Batch entry for prediction decision making.")
    parser.add_argument("--config", default="/app/config/examples/config-embedding.yaml")
    parser.add_argument("--workload", default="default")
    args = parser.parse_args()

    config = load_config(args.config)
    input_directory = get_transfer_data_directory(args.workload, INPUT_DATA_TYPE)
    matching_pairs = load_processed_data(os.path.join(input_directory, INPUT_FILE_NAME))

    model_path = os.path.join(get_model_directory(config, "embedding"), MODEL_FILE_NAME)
    model = EmbeddingModel.load(model_path)

    output_format = config.get("decision_making", {}).get("output_format", "graphml")
    _, predicted_graph = decide_matches(
        mutualtop_pairs=matching_pairs,
        previous_pairs=None,
        model=model,
        output_format=output_format,
    )

    predicted_match_path = os.path.join(get_model_directory(config, "predicted_match"), PREDICTED_MATCH_FILE_NAME)
    predicted_graph.export_graphml(predicted_match_path)

if __name__ == "__main__":
    main()

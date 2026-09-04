import argparse
import os

from utils.pipeline_io import load_config, get_model_directory, get_transfer_data_directory, load_processed_data, write_step_output
from models.embedding_model import EmbeddingModel
from pipeline.calculating_similarity import get_mutual_top_k, score_mutual_topk_candidate_pairs

"""
task: similarity calculation
mode: batch + embedding
input: candidate pairs [transfer], embedding model [model]
output: matching pairs [transfer]
"""

INPUT_DATA_TYPE = "embedding"
INPUT_FILE_NAME = "candidate_pairs.json"
MODEL_FILE_NAME = "embedding.emb"
OUTPUT_DATA_TYPE = "embedding"
OUTPUT_FILE_NAME = "matching_pairs"
OUTPUT_EXTENSION = "json"

def main():
    parser = argparse.ArgumentParser(description="Batch entry for prediction similarity calculation.")
    parser.add_argument("--config", default="/app/config/examples/config-embedding.yaml")
    parser.add_argument("--workload", default="default")
    args = parser.parse_args()

    config = load_config(args.config)
    input_directory = get_transfer_data_directory(args.workload, INPUT_DATA_TYPE)
    candidate_pairs = load_processed_data(os.path.join(input_directory, INPUT_FILE_NAME))

    model_path = os.path.join(get_model_directory(config, "embedding"), MODEL_FILE_NAME)
    model = EmbeddingModel.load(model_path)

    task_config = config.get("calculating_similarity", {}) or {}
    top_k = get_mutual_top_k(config)
    batch_threshold = int(task_config.get("batch_threshold", 2048))
    chunk_size = int(task_config.get("chunk_size", 4096))
    matching_pairs = score_mutual_topk_candidate_pairs(
        model,
        candidate_pairs,
        top_k=top_k,
        batch_threshold=batch_threshold,
        chunk_size=chunk_size,
    ) if candidate_pairs else []

    write_step_output(
        directory=get_transfer_data_directory(args.workload, OUTPUT_DATA_TYPE),
        file_name=OUTPUT_FILE_NAME,
        extension=OUTPUT_EXTENSION,
        output=matching_pairs,
    )

if __name__ == "__main__":
    main()

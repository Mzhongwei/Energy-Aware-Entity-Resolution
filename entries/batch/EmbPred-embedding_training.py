import argparse
import os

from utils.pipeline_io import load_config, get_model_directory, get_transfer_data_directory, load_processed_data
from pipeline.embedding_training import load_or_create_model, train_embeddings

"""
task: embedding model training
mode: batch + embedding
input: walk sequences [transfer]
output: embedding model [model]
description: continues training the batch-trained embedding model on prediction-side sequences.
"""

INPUT_DATA_TYPE = "embedding"
INPUT_FILE_NAME = "prediction_sequences.json"
MODEL_FILE_NAME = "embedding.emb"

def main():
    parser = argparse.ArgumentParser(description="Batch entry for prediction embedding training.")
    parser.add_argument("--config", default="/app/config/examples/config-embedding.yaml")
    parser.add_argument("--workload", default="default")
    args = parser.parse_args()

    config = load_config(args.config)
    input_directory = get_transfer_data_directory(args.workload, INPUT_DATA_TYPE)
    sequences = load_processed_data(os.path.join(input_directory, INPUT_FILE_NAME))

    model_path = os.path.join(get_model_directory(config, "embedding"), MODEL_FILE_NAME)
    model = load_or_create_model(config, model_path)
    model = train_embeddings(config, model, sequences)
    model.save(model_path)

if __name__ == "__main__":
    main()

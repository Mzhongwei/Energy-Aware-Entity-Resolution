import argparse
import os

from utils.pipeline_io import load_config, get_model_directory, get_transfer_data_directory, load_processed_data, write_step_output
from pipeline.graph_construction import load_or_create_graph, persist_graph, serialize_dyn_roots

"""
task: representation graph construction
mode: batch + embedding, training for incremental jobs
input: processed data [transfer]
output: representation graph [model], dyn_roots [transfer]
"""

INPUT_DATA_TYPE = "embedding"
INPUT_FILE_NAME = "training_processed_data.csv"
GRAPH_FILE_NAME = "graph.graphml"
OUTPUT_DATA_TYPE = "embedding"
OUTPUT_FILE_NAME = "training_dyn_roots"
OUTPUT_EXTENSION = "json"

def main():
    parser = argparse.ArgumentParser(description="Batch entry for training graph construction.")
    parser.add_argument("--config", default="/app/config/examples/config-embedding.yaml")
    parser.add_argument("--workload", default="default")
    args = parser.parse_args()

    config = load_config(args.config)
    input_directory = get_transfer_data_directory(args.workload, INPUT_DATA_TYPE)
    processed_data = load_processed_data(os.path.join(input_directory, INPUT_FILE_NAME))

    graph_path = os.path.join(get_model_directory(config, "graph"), GRAPH_FILE_NAME)
    graph = load_or_create_graph(config, graph_path)
    graph.build_relation(processed_data)
    persist_graph(graph, graph_path)

    write_step_output(
        directory=get_transfer_data_directory(args.workload, OUTPUT_DATA_TYPE),
        file_name=OUTPUT_FILE_NAME,
        extension=OUTPUT_EXTENSION,
        output=serialize_dyn_roots(graph),
    )

if __name__ == "__main__":
    main()

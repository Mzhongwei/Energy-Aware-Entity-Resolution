import argparse
import os

from utils.pipeline_io import load_config, get_model_directory, get_transfer_data_directory, load_processed_data, write_step_output
from pipeline.graph_construction import load_or_create_graph, restore_dyn_roots
from pipeline.random_walk import dynrandom_walks_generation

"""
task: random walk sequence generation
mode: batch + embedding
input: representation graph [model], dyn_roots [transfer]
output: walk sequences [transfer]
"""

GRAPH_FILE_NAME = "graph.graphml"
INPUT_DATA_TYPE = "embedding"
INPUT_FILE_NAME = "prediction_dyn_roots.json"
OUTPUT_DATA_TYPE = "embedding"
OUTPUT_FILE_NAME = "prediction_sequences"
OUTPUT_EXTENSION = "json"

def main():
    parser = argparse.ArgumentParser(description="Batch entry for prediction random walk.")
    parser.add_argument("--config", default="/app/config/examples/config-embedding.yaml")
    parser.add_argument("--workload", default="default")
    args = parser.parse_args()

    config = load_config(args.config)
    graph_path = os.path.join(get_model_directory(config, "graph"), GRAPH_FILE_NAME)
    graph = load_or_create_graph(config, graph_path)

    input_directory = get_transfer_data_directory(args.workload, INPUT_DATA_TYPE)
    dyn_roots = load_processed_data(os.path.join(input_directory, INPUT_FILE_NAME))
    restore_dyn_roots(graph, dyn_roots)

    sequences = dynrandom_walks_generation(config, graph) if graph.dyn_roots else []

    write_step_output(
        directory=get_transfer_data_directory(args.workload, OUTPUT_DATA_TYPE),
        file_name=OUTPUT_FILE_NAME,
        extension=OUTPUT_EXTENSION,
        output=sequences,
    )

if __name__ == "__main__":
    main()

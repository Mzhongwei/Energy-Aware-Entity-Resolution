import argparse
import os

from utils.pipeline_io import load_config, get_model_directory, get_transfer_data_directory, write_step_output
from pipeline.evaluation import compare_ground_truth

"""
task: evaluation
mode: batch + embedding
input: predicted matching graph [model], ground truth [config]
output: evaluation report [transfer]
"""

PREDICTED_MATCH_FILE_NAME = "predicted_matching.graphml"
OUTPUT_DATA_TYPE = "report"
OUTPUT_FILE_NAME = "evaluation_report"
OUTPUT_EXTENSION = "json"

def main():
    parser = argparse.ArgumentParser(description="Batch entry for prediction evaluation.")
    parser.add_argument("--config", default="/app/config/examples/config-embedding.yaml")
    parser.add_argument("--workload", default="default")
    args = parser.parse_args()

    config = load_config(args.config)
    config["output_format"] = config.get("decision_making", {}).get("output_format", "graphml")

    predicted_match_path = os.path.join(get_model_directory(config, "predicted_match"), PREDICTED_MATCH_FILE_NAME)
    result = compare_ground_truth(config, similarity_file=predicted_match_path)

    write_step_output(
        directory=get_transfer_data_directory(args.workload, OUTPUT_DATA_TYPE),
        file_name=OUTPUT_FILE_NAME,
        extension=OUTPUT_EXTENSION,
        output=result,
    )

if __name__ == "__main__":
    main()
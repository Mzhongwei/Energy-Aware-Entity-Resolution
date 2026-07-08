import argparse
import os

from pipeline.bert_evaluation import evaluate_from_saved_model
from utils.config_io import load_config
from utils.pipeline_io import load_processed_data, write_step_output

TRANSFER_DATA_DIRECTORY = "batch/bert/"
INPUT_DIRECTORY = "b_evaluation_processed_data"
REPORT_DIRECTORY = "reports"
REPORT_FILE_NAME = "bert_b_evaluation_result.json"

def main():
    parser = argparse.ArgumentParser(description="Batch entry for BERT b_evaluation.")
    parser.add_argument("--config", default="/app/config/examples/config-bert.yaml")
    parser.add_argument("--run_dir", default="/app/data/runs/default")
    args = parser.parse_args()

    config = load_config(args.config)
    input_dir = os.path.join(args.run_dir, TRANSFER_DATA_DIRECTORY, INPUT_DIRECTORY)
    processed_data = {
        "test": load_processed_data(os.path.join(input_dir, "test.csv")),
    }

    if not isinstance(processed_data, dict) or "test" not in processed_data:
        raise ValueError("BERT b_evaluation expects processed_data with a 'test' dataset.")

    output = evaluate_from_saved_model(processed_data, config)
    report_path = os.path.join(args.run_dir, REPORT_DIRECTORY, REPORT_FILE_NAME)
    write_step_output(output_path=report_path, output=output)


if __name__ == "__main__":
    main()

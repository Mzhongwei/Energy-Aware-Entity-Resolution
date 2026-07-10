import argparse
import os

from pipeline.bert_evaluation import evaluate_from_saved_model
from utils.pipeline_io import load_config
from utils.pipeline_io import get_transfer_data_directory, load_processed_data, write_step_output

INPUT_DATA_TYPE = "bert/b_evaluation_processed_data"
REPORT_DATA_TYPE = "reports"
REPORT_FILE_NAME = "bert_b_evaluation_result"
REPORT_EXTENSION = "json"

def main():
    parser = argparse.ArgumentParser(description="Batch entry for BERT b_evaluation.")
    parser.add_argument("--config", default="/app/config/examples/config-bert.yaml")
    parser.add_argument("--workload", default="default")
    args = parser.parse_args()

    config = load_config(args.config)
    input_directory = get_transfer_data_directory(args.workload, INPUT_DATA_TYPE)
    processed_data = {
        "test": load_processed_data(
            os.path.join(input_directory, "test.csv")
        ),
    }

    if not isinstance(processed_data, dict) or "test" not in processed_data:
        raise ValueError("BERT b_evaluation expects processed_data with a 'test' dataset.")

    output = evaluate_from_saved_model(processed_data, config)
    write_step_output(
        directory=get_transfer_data_directory(args.workload, REPORT_DATA_TYPE),
        file_name=REPORT_FILE_NAME,
        extension=REPORT_EXTENSION,
        output=output,
    )


if __name__ == "__main__":
    main()

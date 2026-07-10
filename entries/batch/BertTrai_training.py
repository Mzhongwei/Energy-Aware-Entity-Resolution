import argparse
import os
    
from pipeline.bert_training import train_model
from utils.pipeline_io import get_model_directory, get_transfer_data_directory, load_processed_data
from utils.pipeline_io import load_config

INPUT_DATA_TYPE = "bert/training_processed_data"

def main():
    parser = argparse.ArgumentParser(description="Batch entry for BERT training.")
    parser.add_argument("--config", default="/app/config/examples/config-bert.yaml")
    parser.add_argument("--workload", default="default")
    args = parser.parse_args()

    config = load_config(args.config)
    input_directory = get_transfer_data_directory(args.workload, INPUT_DATA_TYPE)
    processed_data = {
        "train": load_processed_data(
            os.path.join(input_directory, "train.csv")
        ),
        "eval": load_processed_data(
            os.path.join(input_directory, "eval.csv")
        ),
    }

    if not isinstance(processed_data, dict) or "train" not in processed_data or "eval" not in processed_data:
        raise ValueError("BERT training expects processed_data with 'train' and 'eval' datasets.")

    trainer, tokenizer = train_model(config, processed_data)

    save_dir = get_model_directory(config, "bert")
    os.makedirs(save_dir, exist_ok=True)
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)


if __name__ == "__main__":
    main()

import argparse
import os
    
from pipeline.bert_training import train_model
from utils.pipeline_io import load_processed_data
from utils.config_io import load_config

TRANSFER_DATA_DIRECTORY = "batch/bert/"
INPUT_DIRECTORY = "training_processed_data"

def main():
    parser = argparse.ArgumentParser(description="Batch entry for BERT training.")
    parser.add_argument("--config", default="/app/config/examples/config-bert.yaml")
    parser.add_argument("--run_dir", default="/app/data/runs/default")
    args = parser.parse_args()

    config = load_config(args.config)
    input_dir = os.path.join(args.run_dir, TRANSFER_DATA_DIRECTORY, INPUT_DIRECTORY)
    processed_data = {
        "train": load_processed_data(os.path.join(input_dir, "train.csv")),
        "eval": load_processed_data(os.path.join(input_dir, "eval.csv")),
    }

    if not isinstance(processed_data, dict) or "train" not in processed_data or "eval" not in processed_data:
        raise ValueError("BERT training expects processed_data with 'train' and 'eval' datasets.")

    trainer, tokenizer = train_model(config, processed_data)

    version_name = config.get("version_name", "test")
    save_dir = os.path.join(config.get('state_management').get('bert-dir'), version_name)
    os.makedirs(save_dir, exist_ok=True)
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)


if __name__ == "__main__":
    main()

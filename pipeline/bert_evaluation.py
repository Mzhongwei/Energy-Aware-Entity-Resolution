from datasets import Dataset, DatasetDict
import numpy as np
import evaluate
import os
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from .bert_utils import build_compute_metrics, build_tokenize_fn


def evaluate_from_saved_model(processed_data, config):
    state_config = config.get("state_management", {}) if isinstance(config, dict) else {}
    bert_dir = state_config.get("bert-dir", "data/bert")
    if not os.path.isabs(bert_dir):
        bert_dir = os.path.join("/app", bert_dir)
    save_dir = os.path.join(bert_dir, config.get("version_name", "test"))

    if not os.path.isdir(save_dir):
        raise FileNotFoundError(f"Local BERT model directory not found: {save_dir}")

    tokenizer = AutoTokenizer.from_pretrained(save_dir)
    model = AutoModelForSequenceClassification.from_pretrained(save_dir)

    dataset = DatasetDict({
        "test": Dataset.from_pandas(processed_data['test']),
    })
    dataset = dataset.map(build_tokenize_fn(tokenizer), batched=True)

    def tokenize_fn(batch):
        return tokenizer(
            batch["text1"],
            batch["text2"],
            padding="max_length",
            truncation=True,
            max_length=128,
        )

    dataset = dataset.map(tokenize_fn, batched=True)

    args = TrainingArguments(
        output_dir=f"{save_dir}/eval_tmp",
        per_device_eval_batch_size=16,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        tokenizer=tokenizer,
        compute_metrics=build_compute_metrics(),
    )

    results = trainer.evaluate(eval_dataset=dataset["test"])
    results['dataset']= config['testset_path']
    print(f'[RESULT] {results}')
    return results
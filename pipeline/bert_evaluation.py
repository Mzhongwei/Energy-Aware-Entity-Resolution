from datasets import Dataset, DatasetDict
import numpy as np
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from .bert_utils import build_compute_metrics, build_tokenize_fn


def evaluate_from_saved_model(processed_data, config):
    save_dir = config.get("save_dir", "data/bert_model")
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
    print(results)
    return results
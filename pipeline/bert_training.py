from models.bert_model import Model
import torch
import random
import numpy as np

from transformers import TrainingArguments, Trainer
from datasets import Dataset, DatasetDict

from pipeline.bert_utils import build_compute_metrics, build_tokenize_fn

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_model(configuration, processed_data):
    """
    Docstring for train_model
    
    :param configuration: Description
    :param processed_data: {"test": df.DataFrame, "eval": df.DataFrame }
    """
    configuration = configuration.get('bert_training') or {}
    set_seed(configuration.get("seed", 42))

    model_choice = configuration.get("model", "bert")
    num_labels = configuration.get("num_labels", 2)

    model_llm = Model(model_choice, num_labels, freeze_layers=0)
    model = model_llm.get_model()
    tokenizer = model_llm.get_tokenizer()

    train_frame = processed_data.get('train')
    eval_frame = processed_data.get('eval')
    if train_frame is None or eval_frame is None:
        raise ValueError("processed_data must include non-empty 'train' and 'eval' datasets.")
    if len(train_frame) == 0 or len(eval_frame) == 0:
        raise ValueError(
            "processed_data contains empty training or evaluation data. "
            "Check that Data_example/bert/*.csv is present in the mounted PVC and the config paths are correct."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    dataset = DatasetDict({
        "train": Dataset.from_pandas(train_frame),
        "eval": Dataset.from_pandas(eval_frame)
    })

    dataset = dataset.map(build_tokenize_fn(tokenizer), batched=True)

    training_args = TrainingArguments(
        output_dir=f'pipeline/{configuration.get("exp_name", f"{model_choice}-test")}',
        learning_rate=configuration.get("learning_rate", 2e-5),
        per_device_train_batch_size=configuration.get("training_batch_size", 16),
        per_device_eval_batch_size=configuration.get("eval_batch_size", 16),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=configuration.get("epochs", 2),
        push_to_hub=False,
        fp16=torch.cuda.is_available(),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none"   # delete this line if carboncode
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        tokenizer=tokenizer,
        compute_metrics=build_compute_metrics(),
    )

    trainer.train()
    return trainer, tokenizer

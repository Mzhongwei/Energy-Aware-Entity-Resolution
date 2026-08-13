import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def build_tokenize_fn(tokenizer):
    def tokenize_dataset(dataset):
        return tokenizer(
            dataset["text1"],
            dataset["text2"],
            padding="max_length",
            truncation=True,
            max_length=128
        )
    return tokenize_dataset


def build_compute_metrics():
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        return {
            "accuracy": accuracy_score(labels, predictions),
            "f1": f1_score(labels, predictions, average="weighted", zero_division=0),
            "precision": precision_score(
                labels, predictions, average="weighted", zero_division=0
            ),
            "recall": recall_score(
                labels, predictions, average="weighted", zero_division=0
            ),
        }

    return compute_metrics

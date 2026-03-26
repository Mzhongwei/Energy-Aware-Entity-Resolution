import evaluate
import numpy as np

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
    metric_acc = evaluate.load("accuracy")
    metric_f1 = evaluate.load("f1")
    metric_precision = evaluate.load("precision")
    metric_recall = evaluate.load("recall")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        return {
            "accuracy": metric_acc.compute(predictions=predictions, references=labels)["accuracy"],
            "f1": metric_f1.compute(predictions=predictions, references=labels, average="weighted")["f1"],
            "precision": metric_precision.compute(predictions=predictions, references=labels, average="weighted")["precision"],
            "recall": metric_recall.compute(predictions=predictions, references=labels, average="weighted")["recall"],
        }

    return compute_metrics
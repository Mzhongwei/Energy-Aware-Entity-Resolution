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
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        labels = np.asarray(labels)
        classes, supports = np.unique(labels, return_counts=True)
        total = supports.sum()
        precision = recall = f1 = 0.0

        for label, support in zip(classes, supports):
            true_positive = np.sum((predictions == label) & (labels == label))
            predicted = np.sum(predictions == label)
            class_precision = true_positive / predicted if predicted else 0.0
            class_recall = true_positive / support
            denominator = class_precision + class_recall
            class_f1 = 2 * class_precision * class_recall / denominator if denominator else 0.0
            weight = support / total
            precision += weight * class_precision
            recall += weight * class_recall
            f1 += weight * class_f1

        return {
            "accuracy": float(np.mean(predictions == labels)),
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
        }

    return compute_metrics

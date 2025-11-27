from llm.data_processing import sequence_grnerating
from llm.or_dataset import PairDataset, tokenize_dataset
from llm.or_script_model import export_torchscript
from llm.or_train import train
from llm.or_model import BertClassifier, get_tokenizer
import torch


from datasets import DatasetDict, Dataset
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

def evaluate(model, dataset, device, batch_size=16, desc="test"):
    loader = DataLoader(dataset, batch_size=batch_size)
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating [{desc}]"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask, token_type_ids)
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="weighted")
    recall = recall_score(all_labels, all_preds, average="weighted")
    f1 = f1_score(all_labels, all_preds, average="weighted")

    print(f"[{desc}] metrics: acc={acc:.4f}, precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}")
    return {"acc": acc, "precision": precision, "recall": recall, "f1": f1}


def start_training(configuration):
    model_name = "bert-base-uncased"
    tokenizer = get_tokenizer(model_name)

    train_df = sequence_grnerating(configuration['trainset_path'])
    valid_df = sequence_grnerating(configuration['validset_path'])
    test_df = sequence_grnerating(configuration['testset_path'])
    dataset = DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "eval": Dataset.from_pandas(valid_df),
        "test": Dataset.from_pandas(test_df),
    })

    dataset = dataset.map(lambda x: tokenize_dataset(x, tokenizer), batched=True)

    # 让 Hugging Face dataset 直接变成 torch Dataset
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "token_type_ids", "labels"])

    train_dataset = dataset["train"]
    valid_dataset = dataset["eval"]
    test_dataset = dataset["test"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertClassifier(model_name=model_name, num_labels=2)

    model = train(model, train_dataset, valid_dataset, device, epochs=2)

    # === 测试 ===
    evaluate(model, test_dataset, device, batch_size=16, desc="test")

    # 导出 TorchScript
    export_torchscript(model, tokenizer)



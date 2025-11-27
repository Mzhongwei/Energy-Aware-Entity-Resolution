import torch
from torch.utils.data import Dataset
from datasets import DatasetDict, Dataset

class PairDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.encodings["input_ids"][idx]),
            "attention_mask": torch.tensor(self.encodings["attention_mask"][idx]),
            "token_type_ids": torch.tensor(self.encodings["token_type_ids"][idx]),
            "labels": torch.tensor(self.labels[idx])
        }


def tokenize_dataset(dataset, tokenizer, max_len=128):
    return tokenizer(
        dataset["text1"],
        dataset["text2"],
        padding="max_length",
        truncation=True,
        max_length=max_len
    )

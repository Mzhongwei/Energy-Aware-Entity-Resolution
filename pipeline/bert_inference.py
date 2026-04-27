import os

import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

class InferenceService:
    def __init__(self, save_dir=None, max_length=128):
        if save_dir is None:
            save_dir = os.path.join("data", "bert", "test")
        save_dir = os.path.abspath(save_dir)
        _validate_local_checkpoint(save_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(save_dir, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(save_dir, local_files_only=True)
        self.model.to(self.device)
        self.model.eval()
        self.max_length = max_length


def _validate_local_checkpoint(save_dir: str) -> None:
    if not os.path.isdir(save_dir):
        raise FileNotFoundError(
            "Local BERT model directory not found: {path}. "
            "Run training first or mount the model volume in this container.".format(path=save_dir)
        )

    model_weight_files = ("pytorch_model.bin", "model.safetensors")
    tokenizer_files = ("tokenizer_config.json", "tokenizer.json", "vocab.txt", "spiece.model")
    required_common = ("config.json",)

    missing_common = [name for name in required_common if not os.path.isfile(os.path.join(save_dir, name))]
    has_model_weights = any(os.path.isfile(os.path.join(save_dir, name)) for name in model_weight_files)
    has_tokenizer = any(os.path.isfile(os.path.join(save_dir, name)) for name in tokenizer_files)

    if missing_common or not has_model_weights or not has_tokenizer:
        details = []
        if missing_common:
            details.append("missing common files: {files}".format(files=", ".join(missing_common)))
        if not has_model_weights:
            details.append("missing model weights: one of {files}".format(files=", ".join(model_weight_files)))
        if not has_tokenizer:
            details.append("missing tokenizer files: one of {files}".format(files=", ".join(tokenizer_files)))

        raise FileNotFoundError(
            "Incomplete local BERT checkpoint at {path}: {details}".format(
                path=save_dir,
                details="; ".join(details),
            )
        )

    def predict(self, text1, text2):
        inputs = self.tokenizer(
            text1,
            text2,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[0].tolist()
            pred_id = int(torch.argmax(logits, dim=-1).item())

        return {
            "label_id": pred_id,
            "probabilities": probs,
            "similarity_degree": probs[1] if len(probs) > 1 else probs[0],
        }

def process_inference(processed_data, state_manager=None, save_dir=None):
    inference_service = InferenceService(save_dir=save_dir)

    if isinstance(processed_data, dict):
        data_frame = processed_data.get("test")
        if data_frame is None:
            data_frame = processed_data.get("data")
    else:
        data_frame = processed_data

    if isinstance(data_frame, pd.DataFrame):
        rows = data_frame.to_dict(orient="records")
    else:
        rows = list(data_frame or [])

    predicted_rows = []
    for row in rows:
        prediction = inference_service.predict(row["text1"], row["text2"])
        row["labels"] = prediction.get("label_id")
        if state_manager is not None and row["labels"] == 1:
            state_manager.update("predicted_pairs", state_manager.get("predicted_pairs"))
        predicted_rows.append(row)

    return predicted_rows
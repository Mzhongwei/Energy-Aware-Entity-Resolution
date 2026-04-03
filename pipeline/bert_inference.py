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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(save_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(save_dir)
        self.model.to(self.device)
        self.model.eval()
        self.max_length = max_length

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
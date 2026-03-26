import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

class InferenceService:
    def __init__(self, save_dir, max_length=128):
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

def process_inference(processed_data, state_manager): 
    #predicted pairs with ID
    inferenceService = InferenceService()
    for row in processed_data:
        row["labels"] = inferenceService.predict(row["text1"], row["text2"]).get('label_id')
        if row["labels"] == 1:
            state_manager.update("predicted_pairs", state_manager["predicted_pairs"] )
    return processed_data
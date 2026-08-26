from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model


class Model:
    def __init__(
        self,
        model_name: str = "bert",
        num_labels: int = 2,
        freeze_layers: int = 0,
    ):
        self.model_name = model_name.lower()
        if self.model_name == "bert":
            print(f"use Model {self.model_name}, labels number: {num_labels}")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                "bert-base-uncased",
                num_labels=num_labels,
                local_files_only=True,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                "bert-base-uncased",
                local_files_only=True,
            )
        elif self.model_name == "distilbert":
            print(f"use Model {self.model_name}")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                "distilbert-base-uncased",
                num_labels=num_labels,
            )
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def get_model(self):
        return self.model

    def get_tokenizer(self):
        return self.tokenizer

    def freeze_transformer_layers(model, base_model_name, freeze_until=0):
        backbone = getattr(model, base_model_name, None)
        if backbone is None:
            raise ValueError(f"模型里找不到 {base_model_name}, 可选: bert/roberta/distilbert/transformer")

        try:
            layers = backbone.encoder.layer
        except AttributeError:
            try:
                layers = backbone.transformer.layer
            except AttributeError:
                raise ValueError("未识别的 transformer 层结构")

        for idx, layer in enumerate(layers):
            if idx < freeze_until:
                for param in layer.parameters():
                    param.requires_grad = False

        print(f"✅ 已冻结前 {freeze_until} 层 {base_model_name} 的参数")

    def set_peft(self):
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )
        model = get_peft_model(model, peft_config)
        return model

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
class Model:
    def __init__(
            self,
            model_name: str = "bert",
            num_labels: int = 2,
            freeze_layers: int = 0
            ):
        self.model_name = model_name.lower()
        if self.model_name == "bert":
            print(f"use Model {self.model_name}, labels number: {num_labels}")
            self.model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        elif self.model_name == "distilbert":
            print(f"use Model {self.model_name}")
            self.model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_labels)
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    def get_model(self):
        return self.model
    
    def get_tokenizer(self):
        return self.tokenizer

    def freeze_transformer_layers(model, base_model_name, freeze_until=0):
        """
        冻结 transformer 模型的前 freeze_until 层参数
        base_model_name: 主干模型在 HuggingFace 中的名字（bert, roberta, distilbert, transformer...）
        freeze_until: 要冻结的层数（从前往后）
        """
        # 1. 找到 backbone
        backbone = getattr(model, base_model_name, None)
        if backbone is None:
            raise ValueError(f"模型里找不到 {base_model_name}, 可选: bert/roberta/distilbert/transformer")

        # 2. 遍历 transformer 层
        try:
            layers = backbone.encoder.layer  # BERT / RoBERTa
        except AttributeError:
            try:
                layers = backbone.transformer.layer  # DistilBERT / XLNet
            except AttributeError:
                raise ValueError("未识别的 transformer 层结构")

        # 3. 冻结前 freeze_until 层
        for idx, layer in enumerate(layers):
            if idx < freeze_until:
                for param in layer.parameters():
                    param.requires_grad = False

        print(f"✅ 已冻结前 {freeze_until} 层 {base_model_name} 的参数")

        
    def set_peft(self):
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=8, lora_alpha=32, lora_dropout=0.1
        )

        model = get_peft_model(model, peft_config)
        return model
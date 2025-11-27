from llm.data_processing import preparing_testing, preparing_training
from llm.model import Model
import torch, random, numpy as np
from transformers import TrainingArguments
from transformers import Trainer
import numpy as np
import evaluate

def start_training(configuration):

    ##
    # initialize the random seed (to ensure reproducibility)
    SEED = configuration.get("seed", 42)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # For complete reproduction
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    ## 
    # load model and tokenizer
    model_choice = configuration.get("model", "bert")
    num_labels=configuration.get("num_labels", 2)
    model_llm = Model(model_choice, num_labels, freeze_layers = 0)

    model = model_llm.get_model()
    tokenizer = model_llm.get_tokenizer()

    ##
    # choose device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    ##
    # prepare datasets
    dataset = preparing_training(configuration["trainset_path"], configuration["validset_path"])

    ## 
    # tokenize
    def tokenize_dataset(dataset):
        return  tokenizer(
                    dataset["text1"], 
                    dataset["text2"],   # 第二个句子
                    padding="max_length", 
                    truncation=True,
                    max_length=128
                )
    dataset = dataset.map(tokenize_dataset, batched=True)
    # dataset = dataset.rename_column("label", "labels")

    ##
    # metrics. These metrics will be output after each training round
    metric_acc = evaluate.load("accuracy")
    metric_f1 = evaluate.load("f1")
    metric_precision = evaluate.load("precision")
    metric_recall = evaluate.load("recall")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        acc = metric_acc.compute(predictions=predictions, references=labels)
        f1 = metric_f1.compute(predictions=predictions, references=labels, average="weighted")
        precision = metric_precision.compute(predictions=predictions, references=labels, average="weighted")
        recall = metric_recall.compute(predictions=predictions, references=labels, average="weighted")

        return {
            "accuracy": acc["accuracy"],
            "f1": f1["f1"],
            "precision": precision["precision"],
            "recall": recall["recall"]
        }

    ##
    # set up TrainingArguments with the training features and hyperparameters
    training_args = TrainingArguments(
        output_dir=f'pipeline/{configuration.get("exp_name", f"{model_choice}-test")}',
        learning_rate=configuration.get("learning_rate", 2e-5),
        per_device_train_batch_size=configuration.get("training_batch_size", 16),
        per_device_eval_batch_size=configuration.get("eval_batch_size", 16),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=configuration.get("epochs", 2),
        push_to_hub=False,
        fp16=True,   # Enable mixed-precision training (significantly faster on NVIDIA GPUs with Tensor Cores)
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",  # "accuracy"
        greater_is_better=False, # True
    )

    ##
    # pass all these separate components to Trainer and call train() to start.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    ## 
    # test
    dataset = preparing_testing(configuration["testset_path"])
    dataset = dataset.map(tokenize_dataset, batched=True)

    results = trainer.evaluate(eval_dataset=dataset["test"])
    print(results)
    return trainer



   
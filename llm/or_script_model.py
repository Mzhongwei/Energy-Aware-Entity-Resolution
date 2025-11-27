import torch

def export_torchscript(model, tokenizer, save_path="bert_classifier_ts.pt"):
    model.eval()
    device = next(model.parameters()).device   # 获取模型所在设备

    example_inputs = tokenizer(
        "Hello world", "This is a test",
        return_tensors="pt", max_length=128, padding="max_length", truncation=True
    )

    # 把输入移到和模型同样的 device
    example_inputs = {k: v.to(device) for k, v in example_inputs.items()}

    try:
        scripted = torch.jit.script(model)
        print("✅ TorchScript scripting 成功")
    except Exception as e:
        print("⚠ scripting 失败，回退到 trace:", e)
        scripted = torch.jit.trace(
            model,
            (example_inputs["input_ids"], example_inputs["attention_mask"], example_inputs["token_type_ids"])
        )

    scripted.save(save_path)
    print(f"✅ TorchScript 模型已保存: {save_path}")

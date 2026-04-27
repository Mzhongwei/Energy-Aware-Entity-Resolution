import argparse
import json
import os

import pandas as pd
from ruamel.yaml import YAML

from pipeline.bert_evaluation import evaluate_from_saved_model
from pipeline.bert_inference import InferenceService
from pipeline.bert_training import train_model

CONFIG_PATH = os.environ.get("EAER_CONFIG_PATH", "/app/config/examples/config-bert.yaml")


def load_config(config_path: str = CONFIG_PATH):
    if not os.path.exists(config_path):
        return {}
    yaml = YAML(typ="safe")
    with open(config_path, "r", encoding="utf-8") as file_handle:
        loaded = yaml.load(file_handle) or {}
    return loaded if isinstance(loaded, dict) else {}


def _bert_save_dir(config):
    state_config = config.get("state_management", {}) if isinstance(config, dict) else {}
    bert_dir = state_config.get("bert-dir", "data/bert")
    if not os.path.isabs(bert_dir):
        bert_dir = os.path.join("/app", bert_dir)
    version_name = config.get("version_name", "test") if isinstance(config, dict) else "test"
    return os.path.join(bert_dir, version_name)


def _parse_json_payload(content: str):
    stripped = content.strip()
    if not stripped:
        return None

    # Argo parameter files can include log lines; parse the last valid JSON line first.
    for line in reversed([ln.strip() for ln in stripped.splitlines() if ln.strip()]):
        try:
            return deserialize_from_json(json.loads(line))
        except json.JSONDecodeError:
            continue

    try:
        return deserialize_from_json(json.loads(stripped))
    except json.JSONDecodeError:
        return None


def deserialize_from_json(obj):
    if isinstance(obj, dict):
        if obj.get("__dataframe__"):
            return pd.DataFrame(obj.get("data", []))
        return {k: deserialize_from_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [deserialize_from_json(item) for item in obj]
    return obj


def bert_training(config, processed_data):
    trainer, tokenizer = train_model(config, processed_data)
    save_dir = _bert_save_dir(config)
    os.makedirs(save_dir, exist_ok=True)
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    return {"status": "trained", "save_dir": save_dir}


def bert_inference(config, processed_data):
    save_dir = _bert_save_dir(config)
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

    predictions = []
    print("[bert_inference] starting inference on {num_rows} rows".format(num_rows=len(rows)))
    for row in rows:
        prediction = inference_service.predict(row["text1"], row["text2"])
        row["labels"] = prediction.get("label_id")
        predictions.append(row)

    print("[bert_inference] completed, predictions: {predictions}".format(predictions=predictions))
    return predictions


def bert_evaluation(config, processed_data):
    evaluation_results = evaluate_from_saved_model(processed_data, config)
    print("[bert_evaluation] completed, evaluation_results: {evaluation_results}".format(evaluation_results=evaluation_results))
    return evaluation_results

def serialize_for_json(obj):
    if isinstance(obj, pd.DataFrame):
        return {"__dataframe__": True, "data": obj.to_dict(orient="records")}
    if isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [serialize_for_json(item) for item in obj]
    return obj


def load_processed_data(processed_data_value: str):
    if processed_data_value.startswith("@"):
        argfile_path = processed_data_value[1:]
        if os.path.isfile(argfile_path):
            processed_data_value = argfile_path

    if os.path.isfile(processed_data_value) and processed_data_value.lower().endswith(".csv"):
        return pd.read_csv(processed_data_value)

    if os.path.isfile(processed_data_value):
        with open(processed_data_value, "r", encoding="utf-8") as f:
            content = f.read().strip()
            parsed = _parse_json_payload(content)
            if parsed is not None:
                return parsed
            return content

    parsed = _parse_json_payload(processed_data_value)
    if parsed is not None:
        return parsed
    else:
        return processed_data_value


def run_argo_once(mode: str, processed_data_value: str, output_path: str = "-"):
    config = load_config()
    config["mode"] = mode
    processed_data = load_processed_data(processed_data_value)
    print(f"[{mode}] starting")

    if "training" in mode:
        if not isinstance(processed_data, dict) or "train" not in processed_data or "eval" not in processed_data:
            raise ValueError("For training mode, processed_data must be a dict with 'train' and 'eval' DataFrames.")
        output = bert_training(config, processed_data)
    elif "inference" in mode:
        print("[run_argo_once] starting inference")
        output = bert_inference(config, processed_data)
    elif "evaluation" in mode:
        if not isinstance(processed_data, dict) or "test" not in processed_data:
            raise ValueError("For evaluation mode, processed_data must be a dict with a 'test' DataFrame.")
        output = bert_evaluation(config, processed_data)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    serialized_output = json.dumps(serialize_for_json(output))
    if output_path and output_path != "-":
        with open(output_path, "w", encoding="utf-8") as file_handle:
            file_handle.write(serialized_output)
    else:
        print(serialized_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BERT distribution for Argo")
    parser.add_argument("--mode", default="training")
    parser.add_argument("--processed_data", default="")
    parser.add_argument("--output", default="-")
    args = parser.parse_args()
    run_argo_once(mode=args.mode, processed_data_value=args.processed_data, output_path=args.output)
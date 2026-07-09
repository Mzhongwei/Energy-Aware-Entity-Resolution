## disabled
import argparse
import json
import os

import pandas as pd
from ruamel.yaml import YAML

from pipeline.bert_evaluation import evaluate_from_saved_model
from pipeline.bert_inference import process_inference
from pipeline.bert_training import train_model
from utils.pipeline_io import deserialize_from_json, parse_json_payload, serialize_for_json, write_step_output, write_text


CONFIG_PATH = os.environ.get("EAER_CONFIG_PATH", "/app/config/examples/config-bert.yaml")
WORKFLOW_NAME = os.environ.get("WORKFLOW_NAME", "").strip()
STATE_CACHE_PATH = f"/app/data/{WORKFLOW_NAME}/state_cache.json" if WORKFLOW_NAME else "/app/data/state_cache.json"

STATE_CACHE = {}


def load_config(config_path: str = CONFIG_PATH):
    """Load the mounted pipeline YAML config into a dictionary.

    类别：IO / payload 解析类
    """
    if not os.path.exists(config_path):
        return {}
    yaml = YAML(typ="safe")
    with open(config_path, "r", encoding="utf-8") as file_handle:
        loaded = yaml.load(file_handle) or {}
    return loaded if isinstance(loaded, dict) else {}


def _bert_save_dir(config):
    """Resolve the directory containing the saved BERT model artifacts.

    类别：数据管理类
    """
    state_config = config.get("state_management", {}) if isinstance(config, dict) else {}
    bert_dir = state_config.get("bert-dir", "data/bert")
    if not os.path.isabs(bert_dir):
        bert_dir = os.path.join("/app", bert_dir)
    version_name = config.get("version_name", "test") if isinstance(config, dict) else "test"
    return os.path.join(bert_dir, version_name)


def _parse_json_payload(content: str):
    """Parse JSON payloads written by upstream steps, including serialized DataFrames.

    类别：IO / payload 解析类
    """
    return parse_json_payload(content, deserializer=deserialize_from_json)


def _state_config(config):
    """Extract state-management configuration from the loaded pipeline config.

    类别：数据管理类
    """
    if not isinstance(config, dict):
        return {}
    state_config = config.get("state_management", {}) or config.get("state_config", {}) or {}
    return state_config if isinstance(state_config, dict) else {}


def _write_text(path, value):
    """Write plain text or serialized structured data to a file path.

    类别：IO / payload 解析类
    """
    write_text(path, value, serializer=serialize_for_json, ensure_ascii=False)


def _load_state_cache():
    """Load this distribution state cache from disk if it exists.

    类别：数据管理类
    """
    if not os.path.exists(STATE_CACHE_PATH):
        return {}
    try:
        with open(STATE_CACHE_PATH, "r", encoding="utf-8") as file_handle:
            loaded = json.load(file_handle)
        loaded = deserialize_from_json(loaded)
        return loaded if isinstance(loaded, dict) else {}
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return {}


def _persist_state_cache():
    """Persist the in-memory distribution state cache to disk.

    类别：数据管理类
    """
    cache_dir = os.path.dirname(STATE_CACHE_PATH)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    with open(STATE_CACHE_PATH, "w", encoding="utf-8") as file_handle:
        json.dump(serialize_for_json(STATE_CACHE), file_handle)


def _bert_artifact_dir(config):
    """Resolve the directory used to persist BERT training artifacts.

    类别：数据管理类
    """
    state_config = _state_config(config)
    bert_dir = state_config.get("bert-dir", "data/bert")
    if not os.path.isabs(bert_dir):
        bert_dir = os.path.join("/app", bert_dir)
    version_name = config.get("version_name", "test") if isinstance(config, dict) else "test"
    return os.path.join(bert_dir, version_name)


def _predicted_artifact_path(config):
    """Resolve the configured predicted-matching artifact path.

    类别：数据管理类
    """
    state_config = _state_config(config)
    predicted_dir = state_config.get("predicted_match-dir", "data/predicted")
    predicted_name = state_config.get("predicted_match-name") or config.get("version_name", "test")
    if not os.path.isabs(predicted_dir):
        predicted_dir = os.path.join("/app", predicted_dir)
    os.makedirs(predicted_dir, exist_ok=True)
    return os.path.join(predicted_dir, f"{predicted_name}.txt")


def _evaluation_artifact_path(config):
    """Resolve where BERT evaluation results should be written.

    类别：数据管理类
    """
    state_config = _state_config(config)
    predicted_dir = state_config.get("predicted_match-dir", "data/predicted")
    predicted_name = state_config.get("predicted_match-name") or config.get("version_name", "test")
    if not os.path.isabs(predicted_dir):
        predicted_dir = os.path.join("/app", predicted_dir)
    os.makedirs(predicted_dir, exist_ok=True)
    return os.path.join(predicted_dir, f"{predicted_name}.result.txt")


def update(key, value):
    """Update local state and persist durable artifacts when required.

    类别：数据管理类
    """
    config = load_config()
    STATE_CACHE[key] = value

    if key == "bert_model":
        save_dir = _bert_artifact_dir(config)
        os.makedirs(save_dir, exist_ok=True)
        trainer = value.get("trainer") if isinstance(value, dict) else None
        tokenizer = value.get("tokenizer") if isinstance(value, dict) else None
        if trainer is not None and tokenizer is not None:
            trainer.save_model(save_dir)
            tokenizer.save_pretrained(save_dir)
        else:
            _write_text(os.path.join(save_dir, "bert_model.txt"), value)
        STATE_CACHE[key] = {"save_dir": save_dir}
        _persist_state_cache()
        return save_dir

    if key == "predicted_matching":
        _write_text(_predicted_artifact_path(config), value)
        _persist_state_cache()
        return value

    if key in {"result", "evaluation_result"}:
        _write_text(_evaluation_artifact_path(config), value)
        _persist_state_cache()
        return value

    _persist_state_cache()
    return value


STATE_CACHE.update(_load_state_cache())

def bert_training(config, processed_data):
    """Train the BERT model and store trainer/tokenizer state for later steps.

    类别：业务包装类
    """
    trainer, tokenizer = train_model(config, processed_data)
    save_dir = update("bert_model", {"trainer": trainer, "tokenizer": tokenizer})
    return {"status": "trained", "save_dir": save_dir}

def bert_inference(config, processed_data):
    """Run BERT inference from a saved model and persist predicted pairs.

    类别：业务包装类
    """
    predicted_pairs = process_inference(processed_data, save_dir=_bert_save_dir(config))
    update("predicted_matching", predicted_pairs)
    print("[bert_inference] completed, predicted_pairs count: {count}".format(count=len(predicted_pairs)))
    return predicted_pairs

def bert_evaluation(config, processed_data):
    """Evaluate BERT predictions against the configured evaluation data.

    类别：业务包装类
    """
    evaluation_results = evaluate_from_saved_model(processed_data, config)
    update("evaluation_result", evaluation_results)
    print("[bert_evaluation] completed, evaluation_results: {evaluation_results}".format(evaluation_results=evaluation_results))
    return evaluation_results

def load_processed_data(processed_data_value: str):
    """Load processed data from a file path or serialized inline payload.

    类别：IO / payload 解析类
    """
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
    """Execute the BERT distribution once for the selected batch mode.

    类别：Pod / Argo 入口类
    """
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

    write_step_output(
        os.path.dirname(output_path) or ".",
        os.path.splitext(os.path.basename(output_path))[0],
        os.path.splitext(os.path.basename(output_path))[1].lstrip("."),
        output,
        serializer=serialize_for_json,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BERT distribution for Argo")
    parser.add_argument("--mode", default="training")
    parser.add_argument("--processed_data", default="")
    parser.add_argument("--output", default="-")
    parser.add_argument("--function", default="")
    args = parser.parse_args()
    run_argo_once(mode=args.mode, processed_data_value=args.processed_data, output_path=args.output)

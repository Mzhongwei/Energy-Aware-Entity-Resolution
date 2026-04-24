import argparse
import json
import os
import sys

import pandas as pd
from pandas import DataFrame
from ruamel.yaml import YAML

from kafka_chain import kafka_chain_enabled, run_kafka_stage
from pipeline.cg_feature_extraction import compute_features

CONFIG_PATH = os.environ.get("EAER_CONFIG_PATH", "/app/config/examples/config-embedding.yaml")


def load_config(config_path: str = CONFIG_PATH):
    if not os.path.exists(config_path):
        return {}
    yaml = YAML(typ="safe")
    with open(config_path, "r", encoding="utf-8") as file_handle:
        loaded = yaml.load(file_handle) or {}
    return loaded if isinstance(loaded, dict) else {}


def safe_read_csv(path):
    if not path:
        return pd.DataFrame()
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def serialize_for_json(obj):
    if isinstance(obj, pd.DataFrame):
        return {"__dataframe__": True, "data": obj.to_dict(orient="records")}
    if isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [serialize_for_json(item) for item in obj]
    return obj


def deserialize_from_json(obj):
    if isinstance(obj, dict):
        if obj.get("__dataframe__"):
            return pd.DataFrame(obj.get("data", []))
        return {k: deserialize_from_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [deserialize_from_json(item) for item in obj]
    return obj


def _parse_json_payload(content: str):
    stripped = content.strip()
    if not stripped:
        return None
    
    # Argo may prepend logs before the JSON payload, so parse the last valid JSON line first.
    for line in reversed([ln.strip() for ln in stripped.splitlines() if ln.strip()]):
        try:
            return deserialize_from_json(json.loads(line))
        except json.JSONDecodeError:
            continue

    try:
        return deserialize_from_json(json.loads(stripped))
    except json.JSONDecodeError:
        return None


def _coerce_processed_data_to_df(processed_data) -> DataFrame:
    if isinstance(processed_data, pd.DataFrame):
        return processed_data

    if isinstance(processed_data, dict):
        payload = processed_data.get("payload") if "payload" in processed_data else processed_data

        if isinstance(payload, pd.DataFrame):
            return payload

        data_value = payload.get("data") if isinstance(payload, dict) else None
        if isinstance(data_value, pd.DataFrame):
            return data_value
        if isinstance(data_value, list):
            return pd.DataFrame(data_value)

        if isinstance(payload, list):
            return pd.DataFrame(payload)

    if isinstance(processed_data, list):
        return pd.DataFrame(processed_data)

    return pd.DataFrame()


def cg_feature_extraction(config, processed_data):
    print("[cg_feature_extraction]", file=sys.stderr)
    processed_df = _coerce_processed_data_to_df(processed_data)
    if not processed_df.empty:
        method = config.get("candidate_generation", {}).get("method", "fullindexing")
        computed = compute_features(processed_df, method, config)
        print(f'[CG FEATURE EXTRACTION] computed features: {computed}', file=sys.stderr)
        return computed
    else:
        print(f'error : Invalid processed_data format type={type(processed_data)}', file=sys.stderr)
    return None


def load_processed_data(processed_data_value: str) -> dict | DataFrame:
    if processed_data_value.startswith("@"):
        argfile_path = processed_data_value[1:]
        if os.path.isfile(argfile_path):
            processed_data_value = argfile_path

    if os.path.isfile(processed_data_value) and processed_data_value.lower().endswith(".csv"):
        return pd.read_csv(processed_data_value)

    if os.path.isfile(processed_data_value):
        with open(processed_data_value, "r", encoding="utf-8") as f:
            content = f.read()
        parsed = _parse_json_payload(content)
        if parsed is not None:
            return parsed
        try:
            return pd.read_csv(processed_data_value)
        except Exception:
            pass
        return content

    parsed = _parse_json_payload(processed_data_value)
    if parsed is not None:
        return parsed
    return processed_data_value

def run_argo_once(mode: str, processed_data_value: str, output_path: str = "-"):
    config = load_config()
    config["mode"] = mode

    if "inference" in mode and "training" not in mode and kafka_chain_enabled(config, "cg_feature_extraction"):
        print(f"[INFO] Running Kafka chain for CG feature extraction in mode '{mode}'...")
        returned = run_kafka_stage(config, "cg_feature_extraction", lambda payload, message, kafka_config: cg_feature_extraction(kafka_config, payload))
        if returned is not None:
            print(json.dumps(serialize_for_json(returned)))
        else:
            print(json.dumps(serialize_for_json({})))
        return

    output = cg_feature_extraction(config, load_processed_data(processed_data_value))
    payload = json.dumps(serialize_for_json(output))
    if output_path and output_path != "-":
        with open(output_path, "w", encoding="utf-8") as file_handle:
            file_handle.write(payload)
    else:
        print(payload)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CG feature distribution for Argo")
    parser.add_argument("--mode", default="default")
    parser.add_argument("--processed_data", default="")
    parser.add_argument("--output", default="-")
    args = parser.parse_args()
    run_argo_once(mode=args.mode, processed_data_value=args.processed_data, output_path=args.output)
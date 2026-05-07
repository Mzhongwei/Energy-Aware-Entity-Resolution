import argparse
import json
import os
import sys

import pandas as pd
from pandas import DataFrame
from ruamel.yaml import YAML

from pipeline.cg_feature_extraction import compute_features
from utils.buffers import delete_earliest_buffer_file, get_earliest_window_index, load_earliest_buffer, write_buffer, write_eos, wait_for_buffer
from utils.codecarbon import ccdecorator

CONFIG_PATH = os.environ.get("EAER_CONFIG_PATH", "/app/config/examples/config-embedding.yaml")
BUFFER_PATH = "/app/data/buffers/"

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

def _exit(output_path=None, output=None, output_buffer_path=None):
    if output_buffer_path:
        write_eos(output_buffer_path + "_construction", reason=f"timeout_no_initial_buffer")
        write_eos(output_buffer_path + "_candidate", reason=f"timeout_no_initial_buffer")
    payload = json.dumps(serialize_for_json(output))
    if output_path and output_path != "-":
        with open(output_path, "w", encoding="utf-8") as file_handle:
            file_handle.write(payload)
    else:
        print(payload)

@ccdecorator
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

def run_argo_batch(mode: str, processed_data_value: str, output_path: str = "-"):
    config = load_config()
    config["mode"] = mode

    output = cg_feature_extraction(config, load_processed_data(processed_data_value))
    _exit(output_path=output_path, output=output)
    return

def run_argo_incremental(output_path: str = "-"):
    config = load_config()

    load_buffer_path = BUFFER_PATH + "processed_data_feature"
    output_buffer_path = BUFFER_PATH + "cg_feature"

    first_ready = wait_for_buffer(load_buffer_path, timeout_seconds=120)
    if first_ready is None:
        print("[INFO] No incoming buffer within startup timeout; writing EOS and exiting.", file=sys.stderr)
        _exit(output_path, output_buffer_path)
        return
    
    raw_data = load_earliest_buffer(load_buffer_path)
    while raw_data is not None:
        if raw_data.empty:
            next_ready = wait_for_buffer(load_buffer_path, timeout_seconds=30)
            raw_data = load_earliest_buffer(load_buffer_path) if next_ready is not None else None
            continue

        output = cg_feature_extraction(config, raw_data)
        if output is not None:
            window_index = get_earliest_window_index(load_buffer_path)
            write_buffer(output, output_buffer_path + "_construction", window_index, extension="json")
            write_buffer(output, output_buffer_path + "_candidate", window_index, extension="json")

        delete_earliest_buffer_file(load_buffer_path)
        next_ready = wait_for_buffer(load_buffer_path, timeout_seconds=30)
        raw_data = load_earliest_buffer(load_buffer_path) if next_ready is not None else None
    
    _exit(output_path, output_buffer_path)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CG feature distribution for Argo")
    parser.add_argument("--mode", default="default")
    parser.add_argument("--processed_data", default="")
    parser.add_argument("--output", default="-")
    args = parser.parse_args()
    if "embedding" in args.mode and "inference" in args.mode and "training" not in args.mode:
        run_argo_incremental(output_path=args.output)
    else:
        run_argo_batch(mode=args.mode, processed_data_value=args.processed_data, output_path=args.output)
import argparse
import json
import os
import signal
import sys

import pandas as pd
from pandas import DataFrame
from ruamel.yaml import YAML

from pipeline.cg_feature_extraction import compute_features
from utils.pipeline_io import delete_earliest_buffer_file, get_earliest_window_index, load_earliest_buffer, write_buffer, write_eos, wait_for_buffer
from utils.pipeline_io import deserialize_from_json, parse_json_payload, serialize_for_json, write_step_output

CONFIG_PATH = os.environ.get("EAER_CONFIG_PATH", "/app/config/examples/config-embedding.yaml")
BUFFER_PATH = "/app/data/buffers/"


stop_requested = False

def handle_sigterm(signum, frame):
    """Record termination requests so long-running buffer loops can stop cleanly.

    类别：Pod / Argo 入口类
    """
    global stop_requested
    stop_requested = True


signal.signal(signal.SIGTERM, handle_sigterm)
signal.signal(signal.SIGINT, handle_sigterm)

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


def safe_read_csv(path):
    """Read a CSV file when available, otherwise return an empty DataFrame.

    类别：IO / payload 解析类
    """
    if not path:
        return pd.DataFrame()
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def _parse_json_payload(content: str):
    """Parse JSON payloads written by upstream steps, including serialized DataFrames.

    类别：IO / payload 解析类
    """
    return parse_json_payload(content, deserializer=deserialize_from_json)


def _coerce_processed_data_to_df(processed_data) -> DataFrame:
    """Normalize processed-data payload variants into a pandas DataFrame.

    类别：IO / payload 解析类
    """
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
    """Write final output or EOS markers before the distribution process exits.

    类别：Pod / Argo 入口类
    """
    if output_buffer_path:
        write_eos(output_buffer_path + "_construction", reason=f"timeout_no_initial_buffer")
        write_eos(output_buffer_path + "_candidate", reason=f"timeout_no_initial_buffer")
    write_step_output(
        os.path.dirname(output_path) or ".",
        os.path.splitext(os.path.basename(output_path))[0],
        os.path.splitext(os.path.basename(output_path))[1].lstrip("."),
        output,
        serializer=serialize_for_json,
    )

def cg_feature_extraction(config, processed_data):
    """Compute candidate-generation blocking features from processed records.

    类别：业务包装类
    """
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
    """Dispatch a batch Argo invocation to the requested business function.

    类别：Pod / Argo 入口类
    """
    config = load_config()
    config["mode"] = mode

    output = cg_feature_extraction(config, load_processed_data(processed_data_value))
    _exit(output_path=output_path, output=output)
    return

def run_argo_incremental(output_path: str = "-"):
    """Dispatch an incremental worker loop to the requested business function.

    类别：Pod / Argo 入口类
    """
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
            if stop_requested:
                sys.exit(0)
            next_ready = wait_for_buffer(load_buffer_path, timeout_seconds=30)
            raw_data = load_earliest_buffer(load_buffer_path) if next_ready is not None else None
            continue

        output = cg_feature_extraction(config, raw_data)
        if output is not None:
            window_index = get_earliest_window_index(load_buffer_path)
            write_buffer(output, output_buffer_path + "_construction", window_index, extension="json")
            write_buffer(output, output_buffer_path + "_candidate", window_index, extension="json")

        delete_earliest_buffer_file(load_buffer_path)
        if stop_requested:
            sys.exit(0)
        next_ready = wait_for_buffer(load_buffer_path, timeout_seconds=30)
        raw_data = load_earliest_buffer(load_buffer_path) if next_ready is not None else None
    
    _exit(output_path, output_buffer_path)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CG feature distribution for Argo")
    parser.add_argument("--mode", default="default")
    parser.add_argument("--processed_data", default="")
    parser.add_argument("--output", default="-")
    parser.add_argument("--function", default="")
    args = parser.parse_args()
    if "embedding" in args.mode and "inference" in args.mode and "training" not in args.mode:
        run_argo_incremental(output_path=args.output)
    else:
        run_argo_batch(mode=args.mode, processed_data_value=args.processed_data, output_path=args.output)

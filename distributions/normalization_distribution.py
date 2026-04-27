import argparse
import json
import os
import sys
from time import time

import pandas as pd
from pandas.errors import EmptyDataError
from pandas import DataFrame
from ruamel.yaml import YAML
from utils.buffers import load_latest_buffer, _write_buffer, _wait_for_buffer, _write_eos

from pipeline.normalization import index_normalization, sequence_generating_m1

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


def _as_bool(value, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "y"}
    return default


def _counter_path_for_version(version_name: str) -> str:
    save_dir = os.path.join("data", "ids")
    os.makedirs(save_dir, exist_ok=True)
    return os.path.join(save_dir, f"{version_name}.txt")


def _maybe_reset_rid_counter(config: dict) -> str:
    norm_cfg = config.get("normalization", {}) if isinstance(config, dict) else {}
    norm_cfg = norm_cfg if isinstance(norm_cfg, dict) else {}
    reset_counter = _as_bool(norm_cfg.get("reset_counter_on_start", False), default=False)
    version_name = str(config.get("version_name", "test"))
    counter_path = _counter_path_for_version(version_name)

    if reset_counter:
        with open(counter_path, "w", encoding="utf-8") as file_handle:
            file_handle.write("0")
        print(
            f"[normalization] reset_counter_on_start enabled; counter reset to 0 path={counter_path} version={version_name}",
            file=sys.stderr,
        )
    else:
        print(
            f"[normalization] reset_counter_on_start disabled; keeping existing counter path={counter_path} version={version_name}",
            file=sys.stderr,
        )

    return counter_path


def _resolve_embedding_raw_df(config: dict, raw_data: dict | DataFrame):
    if isinstance(raw_data, dict):
        if isinstance(raw_data.get("data"), pd.DataFrame):
            return raw_data["data"]
        payload_df = next((value for value in raw_data.values() if isinstance(value, pd.DataFrame)), pd.DataFrame())
        if not payload_df.empty:
            return payload_df

    if isinstance(raw_data, pd.DataFrame) and not raw_data.empty:
        return raw_data

    source_a = config.get("data_source_A")
    source_b = config.get("data_source_B")

    # For ER evaluation datasets (e.g. Fodors/Zagat), ground-truth ids assume
    # A and B are indexed in one continuous sequence.
    if source_a and source_b:
        df_a = safe_read_csv(source_a)
        df_b = safe_read_csv(source_b)
        if not df_a.empty and not df_b.empty:
            return pd.concat([df_a, df_b], ignore_index=True)

    if isinstance(raw_data, dict):
        if isinstance(raw_data.get("data"), pd.DataFrame):
            return raw_data["data"]
        return next((value for value in raw_data.values() if isinstance(value, pd.DataFrame)), pd.DataFrame())

    if isinstance(raw_data, pd.DataFrame):
        return raw_data

    return pd.DataFrame()


def serialize_for_json(obj):
    if isinstance(obj, pd.DataFrame):
        return {"__dataframe__": True, "data": obj.to_dict(orient="records")}
    if isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [serialize_for_json(item) for item in obj]
    return obj

def normalization(config: dict, raw_data: dict | DataFrame, is_training: bool = False):
    print("[normalization]", file=sys.stderr)
    if 'embedding' in config['mode']:
        # in incremental mode, we index records and normalize

        raw_data_path = config.get("data_source_A")
        raw_df = _resolve_embedding_raw_df(config, raw_data)
        _maybe_reset_rid_counter(config)
        print(
            "[normalization] embedding_input_rows={rows} source_A={source_a} source_B={source_b}".format(
                rows=len(raw_df),
                source_a=config.get("data_source_A"),
                source_b=config.get("data_source_B"),
            ),
            file=sys.stderr,
        )
        processed_data = index_normalization(config, raw_df, raw_data_path, is_training)
    else:
        if "bert" in config.get("mode", ""):
            if "training" in config["mode"]:
                raw_data = {
                    "train": safe_read_csv(config.get("trainset_path")),
                    "eval": safe_read_csv(config.get("evalset_path")),
                }
            else:
                raw_data = {
                    "test": safe_read_csv(config.get("testset_path")),
                }

        for key, df in raw_data.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                raw_data[key] = sequence_generating_m1(df)
        processed_data = raw_data
    print("[normalization] completed, processed_data: {processed_data}".format(processed_data=processed_data), file=sys.stderr)
    return processed_data


def load_raw_data(raw_data_value: str) -> dict | DataFrame:
    if os.path.isfile(raw_data_value) and raw_data_value.lower().endswith(".csv"):
        return {"data": pd.read_csv(raw_data_value)}
    return {"data": pd.DataFrame([{"value": raw_data_value}])}


def run_argo_once(
    mode: str,
    raw_data_value: str,
    data_source_a: str = "",
    data_source_b: str = "",
):
    config = load_config()
    config["mode"] = mode
    if isinstance(data_source_a, str) and data_source_a.strip():
        config["data_source_A"] = data_source_a.strip()
    if isinstance(data_source_b, str) and data_source_b.strip():
        config["data_source_B"] = data_source_b.strip()
    is_training = "training" in mode
    print(f"[DEBUG] mode = {mode}", file=sys.stderr)

    if "embedding" in mode and "inference" in mode and "training" not in mode:
        print(f"[INFO] Running buffer chain for normalization in mode '{mode}'...", file=sys.stderr)
        load_buffer_path = BUFFER_PATH + "rawdata"
        first_ready = _wait_for_buffer(load_buffer_path, timeout_seconds=120)
        if first_ready is None:
            print("[INFO] No incoming raw buffer within startup timeout; writing EOS and exiting.", file=sys.stderr)
            _write_eos(BUFFER_PATH + "normalized", reason=f"normalization_timeout_no_initial_buffer_for_mode_{mode}")
            return
        raw_data = load_latest_buffer(load_buffer_path)
        while raw_data is not None:
            if raw_data.empty:
                print(f"[INFO] Loaded empty buffer; waiting for next buffer...", file=sys.stderr)
                next_ready = _wait_for_buffer(load_buffer_path, timeout_seconds=30)
                raw_data = load_latest_buffer(load_buffer_path) if next_ready is not None else None
                continue
            returned = normalization(config=config, raw_data=raw_data, is_training=is_training)
            if returned is not None:
                _write_buffer([{"value": json.dumps(returned, default=str)}], BUFFER_PATH + "normalized")
            else:
                print(json.dumps(serialize_for_json({})))
            next_ready = _wait_for_buffer(load_buffer_path, timeout_seconds=30)
            raw_data = load_latest_buffer(load_buffer_path) if next_ready is not None else None
        _write_eos(BUFFER_PATH + "normalized", reason=f"normalization_completed_for_mode_{mode}")
        return
    output = normalization(config=config, raw_data=load_raw_data(raw_data_value), is_training=is_training)
    print(json.dumps(serialize_for_json(output)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalization distribution for Argo")
    parser.add_argument("--mode", default="default")
    parser.add_argument("--raw_data", default="")
    parser.add_argument("--data_source_A", default="")
    parser.add_argument("--data_source_B", default="")
    args = parser.parse_args()
    run_argo_once(
        mode=args.mode,
        raw_data_value=args.raw_data,
        data_source_a=args.data_source_A,
        data_source_b=args.data_source_B,
    )
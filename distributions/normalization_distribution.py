import argparse
import json
import os
import sys

import pandas as pd
from pandas import DataFrame
from ruamel.yaml import YAML

from pipeline.normalization import index_normalization, sequence_generating_m1

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


def _resolve_embedding_raw_df(config: dict, raw_data: dict | DataFrame):
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


def normalization(config: dict, raw_data: dict | DataFrame):
    print("[normalization]", file=sys.stderr)
    if 'embedding' in config['mode']:
        # in incremental mode, we index records and normalize
 
        # # data example
        # raw_data = pd.DataFrame(Traceback (most recent call last):File "/app/distributions/normalization_distribution.py", line 92, in <module>run_argo_once(mode=args.mode, raw_data_value=args.raw_data)File "/app/distributions/normalization_distribution.py", line 83, in run_argo_onceoutput = normalization(config=config, raw_data=load_raw_data(raw_data_value))File "/app/distributions/normalization_distribution.py", line 54, in normalizationprocessed_data = index_normalization(config, raw_data, raw_data_path)File "/app/pipeline/normalization.py", line 84, in index_normalizationif "rid" not in raw_data.columns:AttributeError: 'dict' object has no attribute 'columns'
        #     data = {
        #         "name": ["kkk", "ttt", ["hhh", "JJJ"]],
        #         "adress": ["d ? rue", "yes addre", "ad . r"]
        #     }
        # )
        raw_data_path = config.get("data_source_A")
        raw_df = _resolve_embedding_raw_df(config, raw_data)
        print(
            "[normalization] embedding_input_rows={rows} source_A={source_a} source_B={source_b}".format(
                rows=len(raw_df),
                source_a=config.get("data_source_A"),
                source_b=config.get("data_source_B"),
            ),
            file=sys.stderr,
        )
        processed_data = index_normalization(config, raw_df, raw_data_path)
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
    output = normalization(config=config, raw_data=load_raw_data(raw_data_value))
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
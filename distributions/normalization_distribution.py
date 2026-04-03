import argparse
import json
import os

import pandas as pd
from pandas import DataFrame
from ruamel.yaml import YAML

from pipeline.normalization import index_normalization, sequence_generating_m1

CONFIG_PATH = os.environ.get("EAER_CONFIG_PATH", "/app/config/examples/config-bert.yaml")


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


def normalization(config: dict, raw_data: dict | DataFrame):
    print("[normalization]")
    if 'embedding' in config['mode']:
        # in incremental mode, we index records and normalize
 
        # # data example
        # raw_data = pd.DataFrame(
        #     data = {
        #         "name": ["kkk", "ttt", ["hhh", "JJJ"]],
        #         "adress": ["d ? rue", "yes addre", "ad . r"]
        #     }
        # )
        raw_data_path = config.get("data_source_A")
        processed_data = index_normalization(config, raw_data, raw_data_path)
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
    return processed_data


def load_raw_data(raw_data_value: str) -> dict | DataFrame:
    if os.path.isfile(raw_data_value) and raw_data_value.lower().endswith(".csv"):
        return {"data": pd.read_csv(raw_data_value)}
    return {"data": pd.DataFrame([{"value": raw_data_value}])}


def run_argo_once(mode: str, raw_data_value: str):
    config = load_config()
    config["mode"] = mode
    output = normalization(config=config, raw_data=load_raw_data(raw_data_value))
    print(json.dumps(serialize_for_json(output)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalization distribution for Argo")
    parser.add_argument("--mode", default="default")
    parser.add_argument("--raw_data", default="")
    args = parser.parse_args()
    run_argo_once(mode=args.mode, raw_data_value=args.raw_data)
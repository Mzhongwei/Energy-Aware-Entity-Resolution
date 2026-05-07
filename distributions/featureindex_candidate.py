import argparse
import json
import os
import sys

import pandas as pd
from ruamel.yaml import YAML

from pipeline.feature_index_construction import build_index as build_cg_index
from pipeline.feature_index_construction import create_cg_index
from pipeline.candidate_enumeration import enumerate_candidates
from models.cg_index import CGIndex
from utils.buffers import delete_earliest_buffer_file, delete_earliest_buffer_directory, get_earliest_window_index, get_cg_index_buffer_file, load_earliest_buffer, write_buffer, write_eos, wait_for_buffer
from utils.codecarbon import ccdecorator

CONFIG_PATH = os.environ.get("EAER_CONFIG_PATH", "/app/config/examples/config-embedding.yaml")
STATE_CACHE_PATH = "/app/data/state_cache.json"
BUFFER_PATH = "/app/data/buffers/"


def load_config(config_path: str = CONFIG_PATH):
    if not os.path.exists(config_path):
        return {}
    yaml = YAML(typ="safe")
    with open(config_path, "r", encoding="utf-8") as file_handle:
        loaded = yaml.load(file_handle) or {}
    return loaded if isinstance(loaded, dict) else {}


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


def _load_state_cache(path: str = STATE_CACHE_PATH):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as file_handle:
            loaded = json.load(file_handle)
        loaded = deserialize_from_json(loaded)
        return loaded if isinstance(loaded, dict) else {}
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return {}


def _persist_state_cache(path: str = STATE_CACHE_PATH):
    cache_dir = os.path.dirname(path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    serializable_state = {}
    for key, value in STATE_CACHE.items():
        try:
            json.dumps(serialize_for_json(value))
            serializable_state[key] = value
        except (TypeError, ValueError):
            continue

    with open(path, "w", encoding="utf-8") as file_handle:
        json.dump(serialize_for_json(serializable_state), file_handle)


STATE_CACHE = _load_state_cache()


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


def load_cg_feature(cg_feature_value: str):
    if cg_feature_value.startswith("@"):
        argfile_path = cg_feature_value[1:]
        if os.path.isfile(argfile_path):
            cg_feature_value = argfile_path

    if os.path.isfile(cg_feature_value):
        with open(cg_feature_value, "r", encoding="utf-8") as file_handle:
            content = file_handle.read()
        parsed = _parse_json_payload(content)
        if parsed is not None:
            return parsed
        return content

    parsed = _parse_json_payload(cg_feature_value)
    if parsed is not None:
        return parsed
    return cg_feature_value


def _write_text(path: str, value):
    with open(path, "w", encoding="utf-8") as file_handle:
        if isinstance(value, str):
            file_handle.write(value)
        else:
            file_handle.write(json.dumps(serialize_for_json(value)))

def get(key):
    return STATE_CACHE.get(key)

def _exit(output_path=None, output=None, output_buffer_path=None):
    if output_buffer_path:
        write_eos(output_buffer_path, reason=f"timeout_no_initial_buffer")
    payload = json.dumps(serialize_for_json(output))
    if output_path and output_path != "-":
        with open(output_path, "w", encoding="utf-8") as file_handle:
            file_handle.write(payload)
    else:
        print(payload)

def _cg_index_save_dir(config: dict):
    state_config = {}
    if isinstance(config, dict):
        state_config = config.get("state_management", {}) or config.get("state_config", {}) or {}
    if not isinstance(state_config, dict):
        state_config = {}

    index_dir = state_config.get("feature_index-dir", "data/index")
    index_name = state_config.get("feature_index-name") or config.get("version_name", "test")
    return os.path.join(index_dir, index_name)


def ensure_cg_feature_index(config: dict, force_rebuild: bool = False):
    current = get("cg_feature_index")
    if current is not None and hasattr(current, "build") and not force_rebuild:
        return current

    save_dir = _cg_index_save_dir(config)
    return build_cg_feature(save_dir, config)
    

def build_cg_feature(save_dir: str, config: dict):
    manifest_path = os.path.join(save_dir, "manifest.json")
    if os.path.exists(manifest_path):
        index = CGIndex.from_disk(config, save_dir)
        update("cg_feature_index", index)
        return index

    method = str(config.get("candidate_generation", {}).get("method", "fullindexing"))
    index = create_cg_index(method, config)
    update("cg_feature_index", index)
    return index

def update(key, value):
    STATE_CACHE[key] = value
    _persist_state_cache()

    if key != "cg_feature_index":
        return

    config = load_config()
    save_dir = _cg_index_save_dir(config)
    os.makedirs(save_dir, exist_ok=True)

    index = STATE_CACHE.get("cg_feature_index")
    if hasattr(index, "persist"):
        if hasattr(index, "index_dir"):
            index.index_dir = save_dir
        index.persist()
    else:
        _write_text(os.path.join(save_dir, "index.json"), index)
    

def batch_feature_index_construction(config, cg_feature):
    ensure_cg_feature_index(config)
    return feature_index_construction(config, cg_feature)

def incremental_feature_index_construction(config, cg_feature):
    ensure_cg_feature_index(config, True)
    return feature_index_construction(config, cg_feature)

@ccdecorator
def feature_index_construction(config, cg_feature):
    if not isinstance(cg_feature, list):
        raise ValueError("cg_feature must be a feature list.")

    index = get("cg_feature_index")
    if index is None:
        raise ValueError("cg_feature_index must be initialized before feature_index_construction.")
    if not hasattr(index, "build"):
        raise ValueError("cg_feature_index must be a CGIndex instance.")

    build_cg_index(cg_feature, index)
    update("cg_feature_index", index)
    return None

def batch_candidate_enumeration(config, cg_feature):
    ensure_cg_feature_index(config)
    return candidate_enumeration(config, cg_feature)

def incremental_candidate_enumeration(config, cg_feature, path):
    build_cg_feature(path, config)
    return candidate_enumeration(config, cg_feature)

@ccdecorator
def candidate_enumeration(config, cg_feature):
    print("[candidate_enumeration]")
    index = get("cg_feature_index")
    if isinstance(cg_feature, list) and cg_feature and index is not None and hasattr(index, "query"):
        enumerated = enumerate_candidates(cg_feature, index)
        print(f"Enumerated candidate groups: {len(enumerated)}")
        return enumerated
    else:
        print("error")
    return None

def run_argo_batch(mode: str, function: str, cg_feature: str, output_path: str = "-"):
    config = load_config()
    config["mode"] = mode
    config["function"] = function
    cg_feature_value = load_cg_feature(cg_feature)
    
    function_map = {
        "feature_index_construction": batch_feature_index_construction,
        "candidate_enumeration": batch_candidate_enumeration,
    }
    
    if function not in function_map:
        raise ValueError(f"Unsupported function: {function}")
    
    output = function_map[function](config, cg_feature_value)
    _exit(output=output, output_path=output_path)
    return

def run_argo_incremental(function: str, output_path: str = "-"):
    config = load_config()
    config["function"] = function

    if function == "feature_index_construction":
        load_buffer_path = BUFFER_PATH + "cg_feature_construction"
        output_buffer_path = BUFFER_PATH + "cg_feature_index"
        extension = "index"
    else:
        load_buffer_path = BUFFER_PATH + "cg_feature_candidate"
        output_buffer_path = BUFFER_PATH + "candidate_pairs"
        extension = "json"

    first_ready = wait_for_buffer(load_buffer_path, timeout_seconds=120)
    if first_ready is None:
        _exit(output_path=output_path,output_buffer_path=output_buffer_path)
        return
    
    data = load_earliest_buffer(load_buffer_path)
    while data is not None:
        # if data is empty list
        if not data:
            next_ready = wait_for_buffer(load_buffer_path, timeout_seconds=30)
            data = load_earliest_buffer(load_buffer_path) if next_ready is not None else None
            continue

        function_map = {
            "feature_index_construction": incremental_feature_index_construction,
            "candidate_enumeration": incremental_candidate_enumeration,
        }
        
        if function not in function_map:
            raise ValueError(f"Unsupported function: {function}")
        
        window_index = get_earliest_window_index(load_buffer_path)
        if function == "feature_index_construction":
            output = function_map[function](config, data)
            output = get("cg_feature_index")
        else:
            path_ready = wait_for_buffer(BUFFER_PATH + "cg_feature_index", timeout_seconds=30)
            path = get_cg_index_buffer_file(BUFFER_PATH + "cg_feature_index", window_index) if path_ready else None
            if not path:
                continue
            output = function_map[function](config, data, path)
            delete_earliest_buffer_directory(BUFFER_PATH + "cg_feature_index")

        write_buffer(output, output_buffer_path, window_index, extension=extension)

        delete_earliest_buffer_file(load_buffer_path)
        next_ready = wait_for_buffer(load_buffer_path, timeout_seconds=30)
        data = load_earliest_buffer(load_buffer_path) if next_ready is not None else None
    _exit(output_path=output_path, output_buffer_path=output_buffer_path)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CG feature distribution for Argo")
    parser.add_argument("--mode", default="default")
    parser.add_argument("--cg_feature", default="")
    parser.add_argument("--function", default="")
    parser.add_argument("--output", default="-")
    args = parser.parse_args()
    if "embedding" in args.mode and "inference" in args.mode and "training" not in args.mode:
        run_argo_incremental(function=args.function, output_path=args.output)
    else:
        run_argo_batch(mode=args.mode, function=args.function, cg_feature=args.cg_feature, output_path=args.output)
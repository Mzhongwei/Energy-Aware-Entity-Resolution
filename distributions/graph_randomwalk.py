import argparse
import json
from logging import config
import os
import sys
import time

import pandas as pd
from ruamel.yaml import YAML

from models.representation_graph import RepresentationGraph
from pipeline.graph_construction import dyn_graph_generation
from pipeline.random_walk import dynrandom_walks_generation
from utils.buffers import get_earliest_window_index, load_earliest_buffer, write_buffer, wait_for_buffer, write_eos, delete_earliest_buffer_file, get_manifest_file

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


def _resolve_cache_file(path: str) -> str:
    if os.path.isdir(path):
        return os.path.join(path, "state_cache.json")
    return path


def _load_state_cache(path: str = STATE_CACHE_PATH):
    path = _resolve_cache_file(path)
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
    path = _resolve_cache_file(path)
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


def load_processed_data(processed_data_value: str):
    if processed_data_value.startswith("@"):
        argfile_path = processed_data_value[1:]
        if os.path.isfile(argfile_path):
            processed_data_value = argfile_path

    if os.path.isfile(processed_data_value) and processed_data_value.lower().endswith(".csv"):
        return pd.read_csv(processed_data_value)

    if os.path.isfile(processed_data_value):
        with open(processed_data_value, "r", encoding="utf-8") as file_handle:
            content = file_handle.read()
        parsed = _parse_json_payload(content)
        if parsed is not None:
            return parsed
        return content

    parsed = _parse_json_payload(processed_data_value)
    if parsed is not None:
        return parsed
    return processed_data_value


def _write_text(path: str, value):
    with open(path, "w", encoding="utf-8") as file_handle:
        if isinstance(value, str):
            file_handle.write(value)
        else:
            file_handle.write(json.dumps(serialize_for_json(value)))


def _state_config(config: dict) -> dict:
    if not isinstance(config, dict):
        return {}
    cfg = config.get("state_management", {}) or config.get("state_config", {}) or {}
    return cfg if isinstance(cfg, dict) else {}


def _graph_artifact_path(state_config: dict, config: dict) -> str:
    graph_dir = state_config.get("graph-dir", "/app/data/graph")
    version_name = config.get("version_name", "test")
    graph_name = state_config.get("graph-name") or version_name
    os.makedirs(graph_dir, exist_ok=True)
    return os.path.join(graph_dir, f"{graph_name}.graphml")


def _graph_manifest_path(state_config: dict, config: dict) -> str:
    graph_dir = state_config.get("graph-dir", "/app/data/graph")
    version_name = config.get("version_name", "test")
    graph_name = state_config.get("graph-name") or version_name
    os.makedirs(graph_dir, exist_ok=True)
    return os.path.join(graph_dir, f"{graph_name}.manifest.json")


def _graph_config(config: dict) -> dict:
    if not isinstance(config, dict):
        return {}
    graph_cfg = config.get("graph_construction")
    if isinstance(graph_cfg, dict):
        return graph_cfg
    legacy_cfg = config.get("graph")
    if isinstance(legacy_cfg, dict):
        return legacy_cfg
    return {}


def _clean_graph_copy(graph):
    graph_copy = graph.copy()
    allowed_types = (str, int, float, bool)
    for v in graph_copy.vs:
        for attr in list(v.attributes()):
            if not isinstance(v[attr], allowed_types):
                del v[attr]
    for e in graph_copy.es:
        for attr in list(e.attributes()):
            if not isinstance(e[attr], allowed_types):
                del e[attr]
    return graph_copy


def _file_has_content(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0


def _resolve_graph_paths(config: dict):
    state_cfg = _state_config(config)
    return _graph_artifact_path(state_cfg, config), _graph_manifest_path(state_cfg, config)


def _merge_config_from_manifest(config: dict, manifest_path: str):
    merged_config = config
    if not _file_has_content(manifest_path):
        return merged_config

    try:
        with open(manifest_path, "r", encoding="utf-8") as file_handle:
            manifest = json.load(file_handle)
        graph_cfg = manifest.get("graph_config")
        manifest_meta_path = manifest.get("meta_path", config.get("meta_path", []))
        if isinstance(graph_cfg, dict):
            merged_config = dict(config)
            merged_config["graph_construction"] = graph_cfg
            merged_config["meta_path"] = manifest_meta_path
    except Exception as e:
        print(f"[INFO] failed to load manifest: {e}", file=sys.stderr)
        merged_config = config

    return merged_config


def _load_representation_graph(config: dict, graph_path: str, manifest_path: str):
    merged_config = _merge_config_from_manifest(config, manifest_path)

    if _file_has_content(graph_path):
        print(f"[INFO] loading graph from {graph_path}", file=sys.stderr)
        graph = dyn_graph_generation(merged_config)
        graph.load_graph(graph_path)
        return graph

    print(f"[INFO] creating fresh graph (no existing file)", file=sys.stderr)
    return dyn_graph_generation(config)


def get(key):
    return STATE_CACHE.get(key)


def _bootstrap_dyn_roots_if_empty(graph, config):
    if not hasattr(graph, "get_graph") or not hasattr(graph, "dyn_roots"):
        print("[bootstrap] graph or dyn_roots missing", file=sys.stderr)
        return

    g = graph.get_graph()
    dyn_roots = graph.dyn_roots

    # In Argo, graph construction and random-walk often run in separate pods.
    # dyn_roots may be empty after reload, so rebuild a sane default root set.
    if isinstance(dyn_roots, set):
        if dyn_roots:
            return
        rebuilt = set()
        for v in g.vs:
            attrs = v.attributes()
            node_class = attrs.get("node_class")
            if isinstance(node_class, dict) and node_class.get("isroot", False):
                rebuilt.add(int(v.index))
                continue
            name = attrs.get("name")
            if isinstance(name, str) and name.startswith("idx__"):
                rebuilt.add(int(v.index))
        print(f"[bootstrap] rebuilt set-based roots: {len(rebuilt)} nodes", file=sys.stderr)
        graph.dyn_roots = rebuilt
        return

    if isinstance(dyn_roots, dict):
        has_any = any(bool(v) for v in dyn_roots.values())
        if has_any:
            return

        meta_path = config.get("meta_path", []) if isinstance(config, dict) else []
        start_types = []
        if isinstance(meta_path, list) and meta_path:
            if isinstance(meta_path[0], list):
                start_types = [str(path[0]) for path in meta_path if path]
            else:
                start_types = [str(meta_path[0])]

        rebuilt = {k: set() for k in dyn_roots.keys()}
        for v in g.vs:
            vtype = v.attributes().get("type")
            if vtype in rebuilt:
                rebuilt[vtype].add(int(v.index))
        if not any(bool(v) for v in rebuilt.values()) and start_types:
            for st in start_types:
                rebuilt.setdefault(st, set())
                for v in g.vs:
                    if v.attributes().get("type") == st:
                        rebuilt[st].add(int(v.index))
        print(f"[bootstrap] rebuilt dict-based roots: {[(k, len(v)) for k, v in rebuilt.items()]}", file=sys.stderr)
        graph.dyn_roots = rebuilt
    else:
        print(f"[bootstrap] dyn_roots is unsupported type: {type(dyn_roots)}", file=sys.stderr)

def ensure_representation_graph(config: dict, force_reload: bool=False):
    current = get("representation_graph")
    if current is not None and hasattr(current, "build_relation") and not force_reload:
        print("[INFO] graph already cached", file=sys.stderr)
        return current

    graph_path, manifest_path = _resolve_graph_paths(config)
    if graph_path is None:
        graph = dyn_graph_generation(config)
        STATE_CACHE["representation_graph"] = graph
        _persist_state_cache()
        return graph

    graph = _load_representation_graph(config, graph_path, manifest_path)

    STATE_CACHE["representation_graph"] = graph
    _persist_state_cache()
    return graph


def load_graph_from_path(config: dict, graph_path: str, persist_cache: bool = False):
    manifest_path = get_manifest_file(graph_path)
    graph = _load_representation_graph(config, graph_path, manifest_path)

    STATE_CACHE["representation_graph"] = graph
    if persist_cache:
        _persist_state_cache()
    return graph


def update(key, value):
    STATE_CACHE[key] = value
    _persist_state_cache()

    if key != "representation_graph":
        return

    config = load_config()
    state_cfg = _state_config(config)
    graph_path = _graph_artifact_path(state_cfg, config)

    graph = STATE_CACHE.get("representation_graph")
    if isinstance(graph, RepresentationGraph):
        temp_graph_path = f"{graph_path}.tmp"
        temp_manifest_path = f"{_graph_manifest_path(state_cfg, config)}.tmp"
        manifest_path = _graph_manifest_path(state_cfg, config)
        graph_save = _clean_graph_copy(graph.graph)
        try:
            graph_save.write_graphml(temp_graph_path)
            manifest = {
                "graph_class": type(graph).__name__,
                "graph_config": _graph_config(config),
                "meta_path": config.get("meta_path", []),
            }
            with open(temp_manifest_path, "w", encoding="utf-8") as file_handle:
                json.dump(manifest, file_handle, ensure_ascii=False, indent=2)

            os.replace(temp_graph_path, graph_path)
            os.replace(temp_manifest_path, manifest_path)
        except Exception:
            for temp_path in (temp_graph_path, temp_manifest_path):
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            raise
    else:
        _write_text(graph_path, graph)

def incremental_graph_construction(config, processed_data):
    graph = ensure_representation_graph(config, force_reload=True)
    return graph_construction(graph, processed_data)

def batch_graph_construction(config, processed_data):
    graph = ensure_representation_graph(config)
    return graph_construction(graph, processed_data)

def graph_construction(graph, processed_data):
    if not isinstance(processed_data, pd.DataFrame):
        raise ValueError("processed_data must be a pandas DataFrame for graph construction.")
    if not hasattr(graph, "build_relation"):
        raise ValueError("representation_graph must support build_relation for incremental updates.")

    graph.build_relation(processed_data)
    update("representation_graph", graph)
    g = graph.get_graph()

    sample_vertices = [v["name"] for v in g.vs[:5] if "name" in v.attributes()]
    print(f"[graph] sample_vertices={sample_vertices}", file=sys.stderr)

    return {"status": "graph_built"}

def incremental_random_walk(config, graph_path):
    graph = load_graph_from_path(config, graph_path)
    return random_walk(graph, config)

def batch_random_walk(config):
    graph = ensure_representation_graph(config)
    return random_walk(graph, config)

def random_walk(graph, config):
    _bootstrap_dyn_roots_if_empty(graph, config)
    
    has_get_graph = hasattr(graph, "get_graph")
    has_dyn_roots = hasattr(graph, "dyn_roots")
    
    if has_get_graph and has_dyn_roots:
        walks = dynrandom_walks_generation(config, graph)
        return walks
    return []

def _exit(output_path=None, output=None, output_buffer_path=None):
    if output_buffer_path:
        write_eos(output_buffer_path, reason=f"timeout_no_initial_buffer")
    payload = json.dumps(serialize_for_json(output))
    if output_path and output_path != "-":
        with open(output_path, "w", encoding="utf-8") as file_handle:
            file_handle.write(payload)
    else:
        print(payload)

def run_argo_batch(mode: str, function: str, processed_data: str, output_path: str = "-"):
    config = load_config()
    config["mode"] = mode
    config["function"] = function
    processed_data_value = load_processed_data(processed_data)

    function_map = {
        "graph_construction": lambda: batch_graph_construction(config, processed_data_value),
        "random_walk": lambda: batch_random_walk(config),
    }

    if function not in function_map:
        raise ValueError(f"Unsupported function: {function}")

    output = function_map[function]()
    _exit(output_path=output_path, output=output)

def run_argo_incremental(function: str, output_path: str = "-"):
    config = load_config()
    config["function"] = function
    
    if function == "graph_construction":
        load_buffer_path = BUFFER_PATH + "processed_data"
        output_buffer_path = BUFFER_PATH + "graph"
        extension = "graphml"
    else:
        load_buffer_path = BUFFER_PATH + "graph"
        output_buffer_path = BUFFER_PATH + "sequences"
        extension = "json"

    first_ready = wait_for_buffer(load_buffer_path, timeout_seconds=120)
    if first_ready is None:
        _exit(output_path, output_buffer_path)
        return

    data = load_earliest_buffer(load_buffer_path)
    while data is not None:
        if data.empty if isinstance(data, pd.DataFrame) else False:
            next_ready = wait_for_buffer(load_buffer_path, timeout_seconds=30)
            data = load_earliest_buffer(load_buffer_path) if next_ready is not None else None
            continue

        function_map = {
            "graph_construction": lambda: incremental_graph_construction(config, data),
            "random_walk": lambda: incremental_random_walk(config, data),
        }
        
        output = function_map[function]()

        if function == "graph_construction":
            output = get("representation_graph").graph

        window_index = get_earliest_window_index(load_buffer_path)

        write_buffer(output, output_buffer_path, window_index, extension=extension)
        delete_earliest_buffer_file(load_buffer_path)
        next_ready = wait_for_buffer(load_buffer_path, timeout_seconds=30)
        data = load_earliest_buffer(load_buffer_path)

    write_eos(output_buffer_path, reason=f"{function}_completed")
    _exit(output_path, output_buffer_path)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Graph construction/random walk distribution for Argo")
    parser.add_argument("--mode", default="default")
    parser.add_argument("--processed_data", default="")
    parser.add_argument("--function", default="")
    parser.add_argument("--output", default="-")
    args = parser.parse_args()
    if "embedding" in args.mode and "inference" in args.mode and "training" not in args.mode:
        run_argo_incremental(function=args.function, output_path=args.output)
    else:        
        run_argo_batch(mode=args.mode, function=args.function, processed_data=args.processed_data, output_path=args.output)
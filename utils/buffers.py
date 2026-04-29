import json
from time import time, time_ns
import os
import sys
import pandas as pd
from ruamel.yaml import YAML
from pandas.errors import EmptyDataError

CONFIG_PATH = os.environ.get("EAER_CONFIG_PATH", "/app/config/examples/config-embedding.yaml")

def load_config(config_path: str = CONFIG_PATH):
    if not os.path.exists(config_path):
        return {}
    yaml = YAML(typ="safe")
    with open(config_path, "r", encoding="utf-8") as file_handle:
        loaded = yaml.load(file_handle) or {}
    return loaded if isinstance(loaded, dict) else {}

def _wait_for_buffer(buffer_dir: str, timeout_seconds: int = 60) -> str | None:
    start_time = time()
    while time() - start_time < timeout_seconds:
        last_buffer_file = _get_earliest_buffer_file(buffer_dir)
        if last_buffer_file:
            return last_buffer_file
    return None

def _get_earliest_buffer_file(buffer_dir: str) -> str | None:
    accepted_extensions = {".csv", ".json", ".graphml"}

    if not os.path.isdir(buffer_dir):
        return None
    buffer_files = [
        f
        for f in os.listdir(buffer_dir)
        if (
            os.path.isfile(os.path.join(buffer_dir, f))
            and os.path.splitext(f)[1] in accepted_extensions
            and not f.endswith(".manifest.json")
        )
    ]
    if not buffer_files:
        return None
    buffer_files.sort(key=lambda x: os.path.getmtime(os.path.join(buffer_dir, x)), reverse=False)
    return os.path.join(buffer_dir, buffer_files[0])

def get_latest_graphml_file(buffer_dir: str) -> str | None:
    if not os.path.isdir(buffer_dir):
        return None
    graphml_files = [f for f in os.listdir(buffer_dir) if f.endswith(".graphml")]
    if not graphml_files:
        return None
    graphml_files.sort(key=lambda x: os.path.getmtime(os.path.join(buffer_dir, x)), reverse=True)
    return os.path.join(buffer_dir, graphml_files[0])

def get_earliest_graphml_file(buffer_dir: str) -> str | None:
    if not os.path.isdir(buffer_dir):
        return None
    graphml_files = [f for f in os.listdir(buffer_dir) if f.endswith(".graphml")]
    if not graphml_files:
        return None
    graphml_files.sort(key=lambda x: os.path.getmtime(os.path.join(buffer_dir, x)), reverse=False)
    return os.path.join(buffer_dir, graphml_files[0])

def get_manifest_file(graph: str) -> str | None:
    if not graph or not os.path.isfile(graph):
        print(f"[WARNING] Graph file '{graph}' does not exist; cannot find manifest.", file=sys.stderr, flush=True)
        return None
    graph_dir = os.path.dirname(graph)
    graph_name = os.path.basename(graph)
    # Get the file {graph_name}.manifest.json in the same directory
    manifest_name = f"{os.path.splitext(graph_name)[0]}.manifest.json"
    manifest_path = os.path.join(graph_dir, manifest_name)
    if os.path.isfile(manifest_path):
        return manifest_path
    print(f"[WARNING] Manifest file '{manifest_path}' not found for graph '{graph}'.", file=sys.stderr, flush=True)
    return None

def _buffer_directory_has_files(buffer_dir: str) -> bool:
    if not os.path.isdir(buffer_dir):
        return False
    return any(
        f.endswith(".csv") or f.endswith(".json") or f.endswith(".graphml") or f.endswith(".manifest.json")
        for f in os.listdir(buffer_dir)
    )


def _delete_file_if_exists(file_path: str):
    if not file_path:
        return False
    if not os.path.isfile(file_path):
        print(f"[INFO] File already absent: {file_path}", file=sys.stderr, flush=True)
        return True
    if file_path.endswith(".graphml"):
        manifest_path = f"{os.path.splitext(file_path)[0]}.manifest.json"
        if os.path.isfile(manifest_path):
            try:
                os.remove(manifest_path)
                print(f"[INFO] Deleted associated manifest file: {manifest_path}", file=sys.stderr, flush=True)
            except Exception as e:
                print(f"[WARNING] Failed to delete manifest file '{manifest_path}': {e}", file=sys.stderr, flush=True)
    try:
        os.remove(file_path)
    except Exception as e:
        print(f"[ERROR] Failed to delete file '{file_path}': {e}", file=sys.stderr, flush=True)
        return False
    deleted = not os.path.exists(file_path)
    if deleted:
        print(f"[INFO] Deleted file: {file_path}", file=sys.stderr, flush=True)
    else:
        print(f"[ERROR] Delete attempted but file still exists: {file_path}", file=sys.stderr, flush=True)
    return deleted

def _delete_earliest_buffer_file(buffer_dir: str):
    earliest_buffer_file = _get_earliest_buffer_file(buffer_dir)
    if earliest_buffer_file:
        return _delete_file_if_exists(earliest_buffer_file)
    print(f"[INFO] No buffer file found to delete in directory: {buffer_dir}", file=sys.stderr, flush=True)
    return True

def load_earliest_buffer(buffer_dir: str):
    earliest_buffer_file = _get_earliest_buffer_file(buffer_dir)
    if earliest_buffer_file and os.path.basename(earliest_buffer_file).startswith("eos_"):
        print(f"[normalization] Earliest buffer file '{earliest_buffer_file}' is an EOS marker; skipping load.", file=sys.stderr, flush=True)
        return None
    if earliest_buffer_file and os.path.isfile(earliest_buffer_file):
        file_size = os.path.getsize(earliest_buffer_file)
        if file_size == 0:
            print(f"[WARNING] Empty buffer file detected; deleting: {earliest_buffer_file}", file=sys.stderr, flush=True)
            return pd.DataFrame()
        try:
            if earliest_buffer_file.endswith(".json"):
                with open(earliest_buffer_file, "r", encoding="utf-8") as json_file:
                    data = json.load(json_file)
                value = data.get("value", {}) if isinstance(data, dict) else {}
            elif earliest_buffer_file.endswith(".csv"):
                value = pd.read_csv(earliest_buffer_file)
            elif earliest_buffer_file.endswith(".graphml"):
                value = earliest_buffer_file
        except EmptyDataError:
            print(f"[WARNING] Unreadable buffer file detected; deleting: {earliest_buffer_file}", file=sys.stderr, flush=True)
            return pd.DataFrame()
        except Exception as exc:
            print(f"[WARNING] Failed to read buffer file '{earliest_buffer_file}': {exc}; deleting file.", file=sys.stderr, flush=True)
            return pd.DataFrame()
        print(f"[INFO] Read buffer file: {earliest_buffer_file}", file=sys.stderr, flush=True)
        print(f"[INFO] Buffer content preview: {str(value)[:500]}...", file=sys.stderr, flush=True)
        return value
    return pd.DataFrame()

def _write_buffer(data_buffer, output_dir: str, extension: str = "json"):
    if not output_dir or data_buffer is None:
        return
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    write_functions = {
        "json": _write_json_buffer,
        "csv": _write_csv_buffer,
        "graphml": _write_graph_buffer,
    }
    write_function = write_functions.get(extension)
    if not write_function:
        print(f"[ERROR] Unsupported buffer extension '{extension}'; supported extensions are: {list(write_functions.keys())}", file=sys.stderr, flush=True)
        return
    
    try:
        write_function(data_buffer, output_dir)
    except Exception as e:
        print(f"[ERROR] Failed to write buffer to {output_dir} with extension '{extension}': {e}", file=sys.stderr, flush=True)

def _write_graph_buffer(graph, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    config = load_config()
    graph_name = time_ns()
    graph_path = os.path.join(output_dir, f"{graph_name}.graphml")
    manifest_path = os.path.join(output_dir, f"{graph_name}.manifest.json")
    temp_graph_path = f"{graph_path}.tmp"
    temp_manifest_path = f"{manifest_path}.tmp"

    try:
        graph_save = _clean_graph_copy(graph)
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
        print(f"[INFO] Wrote graph buffer to {graph_path} with manifest {manifest_path}.", flush=True)
    except Exception as exc:
        for temp_path in (temp_graph_path, temp_manifest_path):
            if os.path.exists(temp_path):
                os.remove(temp_path)
        raise exc

def _write_csv_buffer(data_buffer, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(data_buffer)
    output_path = os.path.join(output_dir, f"{time_ns()}.csv")
    temp_path = f"{output_path}.tmp"
    df.to_csv(temp_path, index=False)
    os.replace(temp_path, output_path)
    print(f"[INFO] Wrote buffer with {len(data_buffer)} records to {output_path} as CSV.", flush=True)

def _write_json_buffer(data_buffer, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{time_ns()}.json")
    temp_path = f"{output_path}.tmp"
    try:
        with open(temp_path, "w", encoding="utf-8") as json_file:
            json.dump(data_buffer, json_file, default=str)
        os.replace(temp_path, output_path)
    except Exception as e:
        print(f"[ERROR] Failed to write JSON buffer to {output_path}: {e}", file=sys.stderr, flush=True)
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return
    print(f"[INFO] Wrote buffer with {len(data_buffer)} records to {output_path} as JSON.", file=sys.stderr , flush=True)

def _write_eos(output_dir: str, reason: str = "normalization_completed"):
    if not output_dir:
        return
    os.makedirs(output_dir, exist_ok=True)
    eos_payload = {"reason": reason}
    output_path = os.path.join(output_dir, f"eos_{time_ns()}.json")
    with open(output_path, "w", encoding="utf-8") as eos_file:
        json.dump(eos_payload, eos_file)
    print(f"[INFO] Wrote EOS marker: {output_path} reason={reason}", file=sys.stderr, flush=True)

def _clear_buffer_directory(buffer_dir: str):
    if not buffer_dir or not os.path.exists(buffer_dir):
        print(f"[INFO] Buffer directory '{buffer_dir}' does not exist; skipping clear.", flush=True)
        return
    for filename in os.listdir(buffer_dir):
        file_path = os.path.join(buffer_dir, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"[INFO] Deleted old buffer file: {file_path}", flush=True)
        except Exception as e:
            print(f"[WARNING] Failed to delete buffer file {file_path}: {e}", flush=True)

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
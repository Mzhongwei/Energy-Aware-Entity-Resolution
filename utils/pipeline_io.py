import json
import os
import sys
from time import sleep, time, time_ns

import pandas as pd
from pandas.errors import EmptyDataError
from ruamel.yaml import YAML


DATA_ROOT = "/app/data"

# =========================
# Config I/O
# =========================


def load_config(config_path: str):
    if not os.path.exists(config_path):
        return {}
    yaml = YAML(typ="safe")
    with open(config_path, "r", encoding="utf-8") as file_handle:
        loaded = yaml.load(file_handle) or {}
    return loaded if isinstance(loaded, dict) else {}


# =========================
# Directory helpers
# =========================

MODEL_PATH_KEYS = {
    "index": ("feature_index-dir", "feature_index-name", "/app/data/models/index"),
    "graph": ("graph-dir", "graph-name", "/app/data/models/graph"),
    "embedding": ("embedding-dir", "embedding_model-name", "/app/data/models/embedding"),
    "bert": ("bert-dir", "bert-name", "/app/data/models/bert"),
    "predicted_match": ("predicted_match-dir", "predicted_match-name", "/app/data/models/predicted"),
}


def get_transfer_data_directory(workload: str, data_type: str) -> str:
    return os.path.join(DATA_ROOT, workload, "communication", data_type)


def get_model_directory(config: dict, model_type: str) -> str:
    if model_type not in MODEL_PATH_KEYS:
        raise ValueError(f"Unsupported model_type: {model_type}")

    dir_key, name_key, default_root = MODEL_PATH_KEYS[model_type]
    state = config.get("state_management", {}) if isinstance(config, dict) else {}
    root = state.get(dir_key) or default_root
    name = state.get(name_key) or config.get("version_name", "default")
    return os.path.join(root, name)


def get_buffer_directory(workload: str, data_type: str) -> str:
    return os.path.join(DATA_ROOT, workload, "buffers", data_type)


# =========================
# Transfer data I/O
# =========================

def ensure_parent_dir(path: str):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def serialize_for_json(obj):
    if isinstance(obj, pd.DataFrame):
        return {"__dataframe__": True, "data": obj.to_dict(orient="records")}
    if isinstance(obj, set):
        return sorted(obj)
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


def parse_json_payload(content: str, deserializer=deserialize_from_json):
    stripped = content.strip()
    if not stripped:
        return None

    for line in reversed([ln.strip() for ln in stripped.splitlines() if ln.strip()]):
        try:
            return deserializer(json.loads(line))
        except json.JSONDecodeError:
            continue

    try:
        return deserializer(json.loads(stripped))
    except json.JSONDecodeError:
        return None


def write_text(path: str, value, serializer=serialize_for_json, ensure_ascii: bool = False):
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as file_handle:
        if isinstance(value, str):
            file_handle.write(value)
        else:
            file_handle.write(json.dumps(serializer(value), ensure_ascii=ensure_ascii))


def write_step_output(directory: str, file_name: str, extension: str, output, serializer=serialize_for_json):
    output_filename = f"{file_name}.{extension}" if extension else file_name
    output_path = os.path.join(directory, output_filename)
    ensure_parent_dir(output_path)
    if extension.lower() == "csv":
        if not isinstance(output, pd.DataFrame):
            raise ValueError("CSV step output requires a pandas DataFrame.")
        output.to_csv(output_path, index=False)
        return

    payload = json.dumps(serializer(output))
    with open(output_path, "w", encoding="utf-8") as file_handle:
        file_handle.write(payload)

def load_processed_data(processed_data_path: str):
    if not processed_data_path:
        raise ValueError("processed_data_path is required.")

    if not os.path.isfile(processed_data_path):
        raise FileNotFoundError(f"processed_data_path does not exist: {processed_data_path}")

    if processed_data_path.lower().endswith(".csv"):
        return pd.read_csv(processed_data_path)

    with open(processed_data_path, "r", encoding="utf-8") as file_handle:
        content = file_handle.read().strip()

    parsed = parse_json_payload(content, deserializer=deserialize_from_json)
    if parsed is None:
        raise ValueError(f"processed_data_path does not contain valid JSON payload: {processed_data_path}")
    return parsed


# =========================
# Buffer I/O
# =========================


def wait_for_buffer(buffer_dir: str, timeout_seconds: int = 60) -> str | None:
    start_time = time()
    while time() - start_time < timeout_seconds:
        last_buffer_file = _get_earliest_buffer_file(buffer_dir)
        if last_buffer_file:
            return last_buffer_file
        sleep(1)
    return None


def _get_earliest_buffer_file(buffer_dir: str) -> str | None:
    accepted_extensions = {".csv", ".json"}

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


def get_manifest_file(graph: str) -> str | None:
    if not graph or not os.path.isfile(graph):
        print(f"[WARNING] Graph file '{graph}' does not exist; cannot find manifest.", file=sys.stderr, flush=True)
        return None
    graph_dir = os.path.dirname(graph)
    graph_name = os.path.basename(graph)
    manifest_name = f"{os.path.splitext(graph_name)[0]}.manifest.json"
    manifest_path = os.path.join(graph_dir, manifest_name)
    if os.path.isfile(manifest_path):
        return manifest_path
    print(f"[WARNING] Manifest file '{manifest_path}' not found for graph '{graph}'.", file=sys.stderr, flush=True)
    return None


def _delete_directory_if_exists(dir_path: str):
    if not dir_path:
        return False
    if not os.path.isdir(dir_path):
        return True
    try:
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        os.rmdir(dir_path)
        print(f"[INFO] Deleted directory: {dir_path}", file=sys.stderr, flush=True)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to delete directory '{dir_path}': {e}", file=sys.stderr, flush=True)
        return False


def delete_earliest_buffer_directory(buffer_dir: str):
    earliest_dir = _get_earliest_buffer_directory(buffer_dir)
    if earliest_dir:
        return _delete_directory_if_exists(earliest_dir)
    return True


def _delete_file_if_exists(file_path: str):
    if not file_path:
        return False
    if not os.path.isfile(file_path):
        return True
    if file_path.endswith(".graphml") or file_path.endswith(".emb"):
        manifest_path = f"{os.path.splitext(file_path)[0]}.manifest.json"
        if os.path.isfile(manifest_path):
            try:
                os.remove(manifest_path)
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


def delete_earliest_buffer_file(buffer_dir: str):
    earliest_buffer_file = _get_earliest_buffer_file(buffer_dir)
    if earliest_buffer_file:
        return _delete_file_if_exists(earliest_buffer_file)
    return True


def load_earliest_buffer(buffer_dir: str):
    earliest_buffer_file = _get_earliest_buffer_file(buffer_dir)
    if earliest_buffer_file and os.path.basename(earliest_buffer_file).startswith("eos_"):
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
        except EmptyDataError:
            print(f"[WARNING] Unreadable buffer file detected; deleting: {earliest_buffer_file}", file=sys.stderr, flush=True)
            return pd.DataFrame()
        except Exception as exc:
            print(f"[WARNING] Failed to read buffer file '{earliest_buffer_file}': {exc}; deleting file.", file=sys.stderr, flush=True)
            return pd.DataFrame()
        print(f"[INFO] Read buffer file: {earliest_buffer_file}", file=sys.stderr, flush=True)
        return value
    return pd.DataFrame()


def get_cg_index_buffer_file(buffer_dir: str, window_index: int) -> str | None:
    if not os.path.isdir(buffer_dir):
        print(f"[WARNING] Buffer directory '{buffer_dir}' does not exist; cannot retrieve CG index buffer file.", file=sys.stderr, flush=True)
        return None
    index_dirs = [
        d for d in os.listdir(buffer_dir)
        if os.path.isdir(os.path.join(buffer_dir, d)) and d.startswith(f"{window_index}_")
    ]
    if not index_dirs:
        print(f"[WARNING] No CG index buffer directories found in '{buffer_dir}' for window index {window_index}.", file=sys.stderr, flush=True)
        return None
    index_dirs.sort(key=lambda d: os.path.getmtime(os.path.join(buffer_dir, d)), reverse=False)
    return os.path.join(buffer_dir, index_dirs[0])


def wait_for_embedding_buffer(buffer_dir: str, window_index: int, timeout_seconds: int = 60) -> str | None:
    start_time = time()
    while time() - start_time < timeout_seconds:
        emb_file = get_embedding_buffer_file(buffer_dir, window_index)
        if emb_file:
            return emb_file
        sleep(1)
    return None


def get_embedding_buffer_file(buffer_dir: str, window_index: int) -> str | None:
    if not os.path.isdir(buffer_dir):
        print(f"[WARNING] Buffer directory '{buffer_dir}' does not exist; cannot retrieve embedding buffer file.", file=sys.stderr, flush=True)
        return None
    emb_files = [
        f for f in os.listdir(buffer_dir)
        if os.path.isfile(os.path.join(buffer_dir, f)) and f.startswith(f"{window_index}_") and f.endswith(".emb")
    ]
    if not emb_files:
        return None
    emb_files.sort(key=lambda f: os.path.getmtime(os.path.join(buffer_dir, f)), reverse=False)
    return os.path.join(buffer_dir, emb_files[0])


def write_buffer(data_buffer, output_dir: str, prefix: str, extension: str = "json"):
    if not output_dir or data_buffer is None:
        return
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    write_functions = {
        "json": _write_json_buffer,
        "csv": _write_csv_buffer,
        "emb": _write_embedding_model_buffer,
        "index": _write_cg_index_buffer,
    }
    write_function = write_functions.get(extension)
    if not write_function:
        print(f"[ERROR] Unsupported buffer extension '{extension}'; supported extensions are: {list(write_functions.keys())}", file=sys.stderr, flush=True)
        return

    try:
        write_function(data_buffer, output_dir, prefix)
    except Exception as e:
        print(f"[ERROR] Failed to write buffer to {output_dir} with extension '{extension}': {e}", file=sys.stderr, flush=True)


def _write_cg_index_buffer(cg_index, output_dir: str, prefix: str):
    os.makedirs(output_dir, exist_ok=True)
    index_dir = os.path.join(output_dir, f"{prefix}_{time_ns()}")
    cg_index.index_dir = index_dir
    cg_index.persist()
    print(f"[INFO] Wrote CGIndex buffer to {index_dir}", flush=True)


def _write_csv_buffer(data_buffer, output_dir: str, prefix: str):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(data_buffer)
    output_path = os.path.join(output_dir, f"{prefix}_{time_ns()}.csv")
    temp_path = f"{output_path}.tmp"
    df.to_csv(temp_path, index=False)
    os.replace(temp_path, output_path)
    print(f"[INFO] Wrote buffer with {len(data_buffer)} records to {output_path} as CSV.", flush=True)


def _write_json_buffer(data_buffer, output_dir: str, prefix: str):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{prefix}_{time_ns()}.json")
    temp_path = f"{output_path}.tmp"
    try:
        with open(temp_path, "w", encoding="utf-8") as json_file:
            json.dump({"value": data_buffer}, json_file, default=str)
        os.replace(temp_path, output_path)
    except Exception as e:
        print(f"[ERROR] Failed to write JSON buffer to {output_path}: {e}", file=sys.stderr, flush=True)
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return
    print(f"[INFO] Wrote buffer with {len(data_buffer)} records to {output_path} as JSON.", file=sys.stderr, flush=True)


def _write_embedding_model_buffer(model, output_dir: str, prefix: str):
    emb_path = os.path.join(output_dir, f"{prefix}_{time_ns()}.emb")
    temp_emb_path = f"{emb_path}.tmp"
    temp_meta_path = f"{temp_emb_path}.meta.json"
    final_meta_path = f"{emb_path}.meta.json"

    try:
        model.save(temp_emb_path)
        os.replace(temp_emb_path, emb_path)
        if os.path.exists(temp_meta_path):
            os.replace(temp_meta_path, final_meta_path)
    except Exception:
        for path in (temp_emb_path, temp_meta_path):
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass
        raise


def _get_earliest_buffer_directory(buffer_dir: str) -> str | None:
    if not os.path.isdir(buffer_dir):
        return None
    subdirs = [d for d in os.listdir(buffer_dir) if os.path.isdir(os.path.join(buffer_dir, d))]
    if not subdirs:
        return None
    subdirs.sort(key=lambda d: os.path.getmtime(os.path.join(buffer_dir, d)), reverse=False)
    return os.path.join(buffer_dir, subdirs[0])


def get_earliest_window_index(buffer_dir: str) -> int:
    earliest_file = _get_earliest_buffer_file(buffer_dir)
    if not earliest_file:
        earliest_dir = _get_earliest_buffer_directory(buffer_dir)
        if not earliest_dir:
            return 0
        dir_name = os.path.basename(earliest_dir)
        prefix = dir_name.split("_")[0]
        source_name = dir_name
    else:
        filename = os.path.basename(earliest_file)
        prefix = filename.split("_")[0]
        source_name = filename
    try:
        return int(prefix)
    except ValueError:
        print(f"[WARNING] Failed to parse window index from '{source_name}'; defaulting to 0.", file=sys.stderr, flush=True)
        return 0


def write_eos(output_dir: str, reason: str = "stream_completed"):
    if not output_dir:
        return
    os.makedirs(output_dir, exist_ok=True)
    eos_payload = {"reason": reason}
    output_path = os.path.join(output_dir, f"eos_{time_ns()}.json")
    with open(output_path, "w", encoding="utf-8") as eos_file:
        json.dump(eos_payload, eos_file)
    print(f"[INFO] Wrote EOS marker: {output_path} reason={reason}", file=sys.stderr, flush=True)


def clear_buffer_directory(buffer_dir: str):
    if not buffer_dir or not os.path.exists(buffer_dir):
        return
    for filename in os.listdir(buffer_dir):
        file_path = os.path.join(buffer_dir, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"[WARNING] Failed to delete buffer file {file_path}: {e}", flush=True)

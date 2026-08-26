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


def get_incremental_wait_config(config: dict) -> tuple[int | None, float]:
    """Return the single startup timeout and the non-terminal polling interval."""
    incremental = config.get("incremental", {}) if isinstance(config, dict) else {}
    startup_timeout = int(incremental.get("startup_timeout_seconds", 1800))
    poll_interval = float(incremental.get("poll_interval_seconds", 1))
    if startup_timeout <= 0:
        raise ValueError("incremental.startup_timeout_seconds must be greater than zero")
    if poll_interval <= 0:
        raise ValueError("incremental.poll_interval_seconds must be greater than zero")
    if os.environ.get("EAER_HOT_RESTART", "").strip().lower() == "true":
        return None, poll_interval
    return startup_timeout, poll_interval


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


def _window_index_from_name(name: str) -> int | None:
    token = name.split("_", 1)[0].split(".", 1)[0]
    try:
        return int(token)
    except ValueError:
        return None


def _find_earliest(buffer_dir: str, predicate, missing_dir_msg: str | None = None, not_found_msg: str | None = None) -> str | None:
    if not os.path.isdir(buffer_dir):
        if missing_dir_msg:
            print(missing_dir_msg, file=sys.stderr, flush=True)
        return None
    entries = [name for name in os.listdir(buffer_dir) if predicate(os.path.join(buffer_dir, name), name)]
    if not entries:
        if not_found_msg:
            print(not_found_msg, file=sys.stderr, flush=True)
        return None
    entries.sort(
        key=lambda name: (
            _window_index_from_name(name) is None,
            _window_index_from_name(name) or 0,
            os.path.getmtime(os.path.join(buffer_dir, name)),
        )
    )
    return os.path.join(buffer_dir, entries[0])


def _wait_for(finder, timeout_seconds: int | None, should_stop=None, poll_interval_seconds: float = 1) -> str | None:
    start_time = time()
    while timeout_seconds is None or time() - start_time < timeout_seconds:
        if should_stop is not None and should_stop():
            return None
        found = finder()
        if found:
            return found
        sleep(poll_interval_seconds)
    return None


def wait_for_buffer(
    buffer_dir: str,
    timeout_seconds: int | None = None,
    should_stop=None,
    poll_interval_seconds: float = 1,
) -> str | None:
    """Wait for a data buffer or EOS marker.

    ``None`` means wait indefinitely. A finite timeout is intended only for the
    first input of a worker; normal stream completion is signalled exclusively
    by an EOS marker.
    """
    return _wait_for(
        lambda: _get_earliest_buffer_file(buffer_dir),
        timeout_seconds,
        should_stop=should_stop,
        poll_interval_seconds=poll_interval_seconds,
    )


def _get_earliest_buffer_file(buffer_dir: str) -> str | None:
    accepted_extensions = {".csv", ".json"}
    data_file = _find_earliest(
        buffer_dir,
        lambda path, name: (
            os.path.isfile(path)
            and os.path.splitext(name)[1] in accepted_extensions
            and not name.endswith(".manifest.json")
            and not name.startswith("eos_")
        ),
    )
    if data_file:
        return data_file
    # EOS is considered only after every data buffer in this directory has been
    # consumed. This prevents equal/coarse mtimes from making the final buffer appear
    # newer than EOS and being skipped.
    return _find_earliest(
        buffer_dir,
        lambda path, name: os.path.isfile(path) and name.startswith("eos_") and name.endswith(".json"),
    )


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
    return _find_earliest(
        buffer_dir,
        lambda path, name: (
            os.path.isdir(path)
            and name == str(window_index)
            and os.path.isfile(os.path.join(path, ".complete"))
        ),
        missing_dir_msg=f"[WARNING] Buffer directory '{buffer_dir}' does not exist; cannot retrieve CG index buffer file.",
        not_found_msg=f"[WARNING] No CG index buffer directories found in '{buffer_dir}' for window index {window_index}.",
    )


def wait_for_checkpoint_reference(
    buffer_dir: str,
    window_index: int,
    timeout_seconds: int | None = None,
    should_stop=None,
    poll_interval_seconds: float = 1,
) -> str | None:
    def find_reference_or_eos():
        reference_path = os.path.join(buffer_dir, f"{window_index}.json")
        if os.path.isfile(reference_path):
            return reference_path
        return _find_earliest(
            buffer_dir,
            lambda path, name: os.path.isfile(path) and name.startswith("eos_") and name.endswith(".json"),
        )

    return _wait_for(
        find_reference_or_eos,
        timeout_seconds,
        should_stop=should_stop,
        poll_interval_seconds=poll_interval_seconds,
    )


def load_checkpoint_reference(reference_path: str) -> dict:
    with open(reference_path, "r", encoding="utf-8") as reference_file:
        payload = json.load(reference_file)
    reference = payload.get("value") if isinstance(payload, dict) else None
    return resolve_checkpoint_reference(reference, reference_path)


def resolve_checkpoint_reference(reference, source: str = "checkpoint reference") -> dict:
    if not isinstance(reference, dict):
        raise ValueError(f"Invalid checkpoint reference: {source}")
    checkpoint_path = reference.get("checkpoint_path")
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint target does not exist: {checkpoint_path}")
    if checkpoint_path.endswith(".emb") and not os.path.isfile(
        os.path.join(os.path.dirname(checkpoint_path), ".complete")
    ):
        raise RuntimeError(f"Embedding checkpoint is incomplete: {checkpoint_path}")
    if checkpoint_path.endswith(".graphml"):
        metadata_path = os.path.splitext(checkpoint_path)[0] + ".json"
        if not os.path.isfile(metadata_path):
            raise RuntimeError(f"Graph checkpoint is incomplete: {checkpoint_path}")
    return reference


def write_checkpoint_reference(
    output_dir: str,
    reference_name: int | str,
    checkpoint_path: str,
    checkpoint_window: int,
    **metadata,
) -> str:
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_path.endswith(".emb"):
        checkpoint_dir = os.path.dirname(checkpoint_dir)
    reference = {
        "checkpoint_path": checkpoint_path,
        "checkpoint_dir": checkpoint_dir,
        "checkpoint_window": checkpoint_window,
        **metadata,
    }
    return _write_json_buffer(reference, output_dir, reference_name)


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
        raise ValueError(
            f"Unsupported buffer extension '{extension}'; supported extensions are: {list(write_functions.keys())}"
        )

    return write_function(data_buffer, output_dir, prefix)


def _write_cg_index_buffer(cg_index, output_dir: str, prefix: str):
    os.makedirs(output_dir, exist_ok=True)
    final_dir = os.path.join(output_dir, str(prefix))
    if os.path.isfile(os.path.join(final_dir, ".complete")):
        return final_dir
    if os.path.isdir(final_dir):
        _delete_directory_if_exists(final_dir)
    temp_dir = os.path.join(output_dir, f".{prefix}.{time_ns()}.tmp")
    try:
        os.makedirs(temp_dir)
        cg_index.index_dir = temp_dir
        cg_index.persist()
        with open(os.path.join(temp_dir, ".complete"), "w", encoding="utf-8"):
            pass
        os.replace(temp_dir, final_dir)
    except Exception:
        _delete_directory_if_exists(temp_dir)
        raise
    print(f"[INFO] Wrote CGIndex buffer to {final_dir}", flush=True)
    return final_dir


def _write_csv_buffer(data_buffer, output_dir: str, prefix: str):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(data_buffer)
    output_path = os.path.join(output_dir, f"{prefix}.csv")
    temp_path = os.path.join(output_dir, f".{prefix}.{time_ns()}.csv.tmp")
    df.to_csv(temp_path, index=False)
    os.replace(temp_path, output_path)
    print(f"[INFO] Wrote buffer with {len(data_buffer)} records to {output_path} as CSV.", flush=True)
    return output_path


def _write_json_buffer(data_buffer, output_dir: str, prefix: str):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{prefix}.json")
    temp_path = os.path.join(output_dir, f".{prefix}.{time_ns()}.json.tmp")
    try:
        with open(temp_path, "w", encoding="utf-8") as json_file:
            json.dump({"value": data_buffer}, json_file, default=str)
        os.replace(temp_path, output_path)
    except Exception as e:
        print(f"[ERROR] Failed to write JSON buffer to {output_path}: {e}", file=sys.stderr, flush=True)
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise
    print(f"[INFO] Wrote buffer with {len(data_buffer)} records to {output_path} as JSON.", file=sys.stderr, flush=True)
    return output_path


def _write_embedding_model_buffer(model, output_dir: str, prefix: str):
    buffer_name = str(prefix)
    final_dir = os.path.join(output_dir, buffer_name)
    final_emb_path = os.path.join(final_dir, "embedding.emb")
    if os.path.isfile(final_emb_path) and os.path.isfile(os.path.join(final_dir, ".complete")):
        return final_emb_path
    if os.path.isdir(final_dir):
        _delete_directory_if_exists(final_dir)
    temp_dir = os.path.join(output_dir, f".{buffer_name}.{time_ns()}.tmp")
    temp_emb_path = os.path.join(temp_dir, "embedding.emb")

    try:
        os.makedirs(temp_dir)
        model.save(temp_emb_path)
        with open(os.path.join(temp_dir, ".complete"), "w", encoding="utf-8"):
            pass
        os.replace(temp_dir, final_dir)
    except Exception:
        _delete_directory_if_exists(temp_dir)
        raise
    return os.path.join(final_dir, "embedding.emb")


def _get_earliest_buffer_directory(buffer_dir: str) -> str | None:
    return _find_earliest(
        buffer_dir,
        lambda path, name: os.path.isdir(path) and _window_index_from_name(name) is not None,
    )


def get_earliest_window_index(buffer_dir: str) -> int:
    earliest_file = _get_earliest_buffer_file(buffer_dir)
    if not earliest_file:
        earliest_dir = _get_earliest_buffer_directory(buffer_dir)
        if not earliest_dir:
            return 0
        dir_name = os.path.basename(earliest_dir)
        prefix = _window_index_from_name(dir_name)
        source_name = dir_name
    else:
        filename = os.path.basename(earliest_file)
        prefix = _window_index_from_name(filename)
        source_name = filename
    if prefix is None:
        print(f"[WARNING] Failed to parse window index from '{source_name}'; defaulting to 0.", file=sys.stderr, flush=True)
        return 0
    return prefix


def write_window_checkpoint(checkpoint_dir: str, window_index: int, value) -> str:
    return _write_json_buffer(value, checkpoint_dir, window_index)


def load_window_checkpoint(checkpoint_dir: str, window_index: int):
    checkpoint_path = os.path.join(checkpoint_dir, f"{window_index}.json")
    if not os.path.isfile(checkpoint_path):
        return None
    with open(checkpoint_path, "r", encoding="utf-8") as checkpoint_file:
        payload = json.load(checkpoint_file)
    return payload.get("value") if isinstance(payload, dict) else None


def mark_window_published(checkpoint_dir: str, window_index: int) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    marker_path = os.path.join(checkpoint_dir, f"{window_index}.published")
    temp_path = f"{marker_path}.{time_ns()}.tmp"
    with open(temp_path, "w", encoding="utf-8"):
        pass
    os.replace(temp_path, marker_path)


def is_window_published(checkpoint_dir: str, window_index: int) -> bool:
    return os.path.isfile(os.path.join(checkpoint_dir, f"{window_index}.published"))


def acknowledge_window_checkpoint(checkpoint_dir: str, window_index: int) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    marker_path = os.path.join(checkpoint_dir, f"{window_index}.ack")
    temp_path = f"{marker_path}.{time_ns()}.tmp"
    with open(temp_path, "w", encoding="utf-8"):
        pass
    os.replace(temp_path, marker_path)
    print(f"[INFO] Acknowledged checkpoint window {window_index}: {checkpoint_dir}", flush=True)


def is_window_checkpoint_acknowledged(checkpoint_dir: str, window_index: int) -> bool:
    return os.path.isfile(os.path.join(checkpoint_dir, f"{window_index}.ack"))


def wait_for_window_checkpoint_ack(
    checkpoint_dir: str,
    window_index: int,
    should_stop=None,
    poll_interval_seconds: float = 1,
) -> bool:
    marker_path = os.path.join(checkpoint_dir, f"{window_index}.ack")
    if not os.path.isfile(marker_path):
        print(f"[INFO] Waiting for checkpoint window {window_index} to be consumed", flush=True)
    return bool(
        _wait_for(
            lambda: marker_path if os.path.isfile(marker_path) else None,
            None,
            should_stop=should_stop,
            poll_interval_seconds=poll_interval_seconds,
        )
    )


def latest_graph_checkpoint(checkpoint_dir: str):
    if not os.path.isdir(checkpoint_dir):
        return None
    indexes = [
        index
        for name in os.listdir(checkpoint_dir)
        if name.endswith(".json")
        for index in [_window_index_from_name(name)]
        if index is not None
        and os.path.isfile(os.path.join(checkpoint_dir, f"{index}.graphml"))
    ]
    if not indexes:
        return None
    window_index = max(indexes)
    return window_index, os.path.join(checkpoint_dir, f"{window_index}.graphml")


def latest_embedding_checkpoint(checkpoint_dir: str):
    if not os.path.isdir(checkpoint_dir):
        return None
    indexes = [
        index
        for name in os.listdir(checkpoint_dir)
        if os.path.isdir(os.path.join(checkpoint_dir, name))
        for index in [_window_index_from_name(name)]
        if index is not None
        and os.path.isfile(os.path.join(checkpoint_dir, name, "embedding.emb"))
        and os.path.isfile(os.path.join(checkpoint_dir, name, ".complete"))
    ]
    if not indexes:
        return None
    window_index = max(indexes)
    return window_index, os.path.join(checkpoint_dir, str(window_index), "embedding.emb")


def prune_window_checkpoints(checkpoint_dir: str, keep_window_index: int) -> None:
    if not os.path.isdir(checkpoint_dir):
        return
    names = os.listdir(checkpoint_dir)
    indexes = sorted({
        window_index
        for name in names
        for window_index in [_window_index_from_name(name)]
        if window_index is not None and window_index < keep_window_index
    })
    for window_index in indexes:
        ack_path = os.path.join(checkpoint_dir, f"{window_index}.ack")
        if not os.path.isfile(ack_path):
            continue
        deleted = True
        for name in names:
            if _window_index_from_name(name) != window_index or name == f"{window_index}.ack":
                continue
            path = os.path.join(checkpoint_dir, name)
            if os.path.isdir(path):
                deleted = _delete_directory_if_exists(path) and deleted
            elif os.path.isfile(path):
                os.remove(path)
        if deleted:
            os.remove(ack_path)


def write_eos(output_dir: str, reason: str = "stream_completed"):
    if not output_dir:
        return
    os.makedirs(output_dir, exist_ok=True)
    eos_payload = {"reason": reason}
    output_path = os.path.join(output_dir, "eos_stream.json")
    temp_path = f"{output_path}.{time_ns()}.tmp"
    with open(temp_path, "w", encoding="utf-8") as eos_file:
        json.dump(eos_payload, eos_file)
    os.replace(temp_path, output_path)
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

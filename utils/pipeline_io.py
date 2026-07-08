import json
import os

import pandas as pd


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


def write_json(path: str, value, serializer=serialize_for_json, **dump_kwargs):
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as file_handle:
        json.dump(serializer(value), file_handle, **dump_kwargs)


def write_step_output(output_path: str | None, output, serializer=serialize_for_json):
    payload = json.dumps(serializer(output))
    if output_path and output_path != "-":
        ensure_parent_dir(output_path)
        with open(output_path, "w", encoding="utf-8") as file_handle:
            file_handle.write(payload)
    else:
        print(payload)

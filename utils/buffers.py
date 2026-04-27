import json
from time import time, time_ns
import os
import sys
import pandas as pd
from pandas.errors import EmptyDataError

def _wait_for_buffer(buffer_dir: str, timeout_seconds: int = 60) -> str | None:
    start_time = time()
    while time() - start_time < timeout_seconds:
        last_buffer_file = _get_last_buffer_file(buffer_dir)
        if last_buffer_file:
            return last_buffer_file
    print(f"[WARNING] No buffer file found in '{buffer_dir}' after waiting for {timeout_seconds} seconds.", flush=True)
    return None

def _get_last_buffer_file(buffer_dir: str) -> str | None:
    if not os.path.isdir(buffer_dir):
        return None
    buffer_files = [
        f
        for f in os.listdir(buffer_dir)
        if (f.endswith(".csv") or f.endswith(".json")) and not f.endswith(".tmp")
    ]
    if not buffer_files:
        return None
    buffer_files.sort(key=lambda x: os.path.getmtime(os.path.join(buffer_dir, x)), reverse=True)
    return os.path.join(buffer_dir, buffer_files[0])

def _delete_file_if_exists(file_path: str):
    if not file_path:
        return False
    if not os.path.isfile(file_path):
        print(f"[INFO] File already absent: {file_path}", file=sys.stderr, flush=True)
        return True
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


def load_latest_buffer(buffer_dir: str) -> pd.DataFrame:
    last_buffer_file = _get_last_buffer_file(buffer_dir)
    if last_buffer_file and os.path.basename(last_buffer_file).startswith("eos_"):
        print(f"[normalization] Latest buffer file '{last_buffer_file}' is an EOS marker; skipping load.", file=sys.stderr, flush=True)
        _delete_file_if_exists(last_buffer_file)
        return None
    if last_buffer_file and os.path.isfile(last_buffer_file):
        file_size = os.path.getsize(last_buffer_file)
        if file_size == 0:
            print(f"[WARNING] Empty buffer file detected; deleting: {last_buffer_file}", file=sys.stderr, flush=True)
            _delete_file_if_exists(last_buffer_file)
            return pd.DataFrame()
        try:
            df = pd.read_csv(last_buffer_file)
        except EmptyDataError:
            print(f"[WARNING] Unreadable buffer file detected; deleting: {last_buffer_file}", file=sys.stderr, flush=True)
            _delete_file_if_exists(last_buffer_file)
            return pd.DataFrame()
        except Exception as exc:
            print(f"[WARNING] Failed to read buffer file '{last_buffer_file}': {exc}; deleting file.", file=sys.stderr, flush=True)
            _delete_file_if_exists(last_buffer_file)
            return pd.DataFrame()
        print(f"[INFO] Read buffer file: {last_buffer_file}", file=sys.stderr, flush=True)
        _delete_file_if_exists(last_buffer_file)
        return df
    return pd.DataFrame()

def _write_buffer(data_buffer: list[dict], output_dir: str):
    if not output_dir or not data_buffer:
        return
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    # If data buffer is a list of objects, write a single JSON file instead of CSV
    if all(not isinstance(record, dict) for record in data_buffer):
        output_path = os.path.join(output_dir, f"{time_ns()}.json")
        temp_path = f"{output_path}.tmp"
        with open(temp_path, "w", encoding="utf-8") as json_file:
            json.dump(data_buffer, json_file, default=str)
        os.replace(temp_path, output_path)
        print(f"[INFO] Wrote buffer with {len(data_buffer)} records to {output_path} as JSON.", flush=True)
        return
    df = pd.DataFrame(data_buffer)
    output_path = os.path.join(output_dir, f"{time_ns()}.csv")
    temp_path = f"{output_path}.tmp"
    df.to_csv(temp_path, index=False)
    os.replace(temp_path, output_path)
    print(f"[INFO] Wrote buffer with {len(data_buffer)} records to {output_path} as CSV.", flush=True)

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
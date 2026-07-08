import ast
import os
import sys
import time

import pandas as pd

from utils.pipeline_io import deserialize_from_json, parse_json_payload


def log(message: str):
    """Write diagnostic messages to stderr for Argo/Kubernetes logs.

    类别：诊断和小工具类
    """
    print(message, file=sys.stderr)


def safe_len(value):
    """Return len(value) when possible without raising on unsupported objects.

    类别：诊断和小工具类
    """
    try:
        return len(value)
    except Exception:
        return "n/a"


def sample_pairs(candidate_pairs, limit: int = 3):
    """Build a small readable preview of candidate or matching pairs for diagnostics.

    类别：诊断和小工具类
    """
    if not isinstance(candidate_pairs, list) or not candidate_pairs:
        return []
    preview = []
    for item in candidate_pairs[:limit]:
        if isinstance(item, (list, tuple)):
            preview.append([str(part) for part in item[:2]])
        else:
            preview.append([str(item)])
    return preview


def sample_ids_from_pairs(candidate_pairs, limit: int = 5):
    """Extract representative record ids from pairs for embedding diagnostics.

    类别：诊断和小工具类
    """
    if not isinstance(candidate_pairs, list) or not candidate_pairs:
        return []
    ids = []
    for item in candidate_pairs[:limit]:
        if not isinstance(item, (list, tuple)) or not item:
            continue
        ids.append(str(item[0]))
        right_side = item[1]
        if isinstance(right_side, (list, tuple)):
            ids.extend(str(value) for value in right_side[:3])
        elif right_side is not None:
            ids.append(str(right_side))
    return ids[: limit * 4]


def parse_embedding_json_payload(content: str):
    """Parse embedding-entry JSON payloads using the shared pipeline deserializer.

    类别：IO / payload 解析类
    """
    return parse_json_payload(content, deserializer=deserialize_from_json)


def unwrap_string_payload(value):
    """Repeatedly unwrap JSON or Python-literal strings into structured objects.

    类别：IO / payload 解析类
    """
    current = value
    for _ in range(5):
        if not isinstance(current, str):
            break
        stripped = current.strip()
        if not stripped:
            break

        parsed = parse_embedding_json_payload(stripped)
        if parsed is None:
            try:
                parsed = ast.literal_eval(stripped)
            except Exception:
                break

        if parsed == current:
            break
        current = parsed
    return current


def load_input_payload(input_value: str):
    """Load an entrypoint input from a path, argfile reference, or inline payload.

    类别：IO / payload 解析类
    """
    original_input = input_value
    if input_value.startswith("@"):
        argfile_path = input_value[1:]
        if os.path.isfile(argfile_path):
            input_value = argfile_path

    looks_like_path = isinstance(input_value, str) and (
        os.path.sep in input_value or input_value.endswith((".json", ".csv", ".emb", ".graphml"))
    )
    if looks_like_path and not os.path.isfile(input_value):
        deadline = time.time() + 30
        while time.time() < deadline and not os.path.isfile(input_value):
            time.sleep(0.5)

    if os.path.isfile(input_value) and input_value.lower().endswith(".csv"):
        loaded = pd.read_csv(input_value)
        log(f"[load_input_payload] source={input_value} type=DataFrame rows={len(loaded)} cols={len(loaded.columns)}")
        return loaded

    if os.path.isfile(input_value):
        with open(input_value, "r", encoding="utf-8") as file_handle:
            content = file_handle.read()
        parsed = unwrap_string_payload(content)
        if parsed is not None:
            if isinstance(parsed, dict):
                if "value" in parsed:
                    parsed = parsed["value"]
                elif "data" in parsed:
                    parsed = parsed["data"]
            log(f"[load_input_payload] source={input_value} type={type(parsed).__name__} size={safe_len(parsed)}")
            return parsed
        log(f"[load_input_payload] source={input_value} type=str size={len(content)}")
        return content

    parsed = unwrap_string_payload(input_value)
    if parsed is not None and not (isinstance(parsed, str) and parsed == input_value):
        if isinstance(parsed, dict):
            if "value" in parsed:
                parsed = parsed["value"]
            elif "data" in parsed:
                parsed = parsed["data"]
        log(f"[load_input_payload] source=inline type={type(parsed).__name__} size={safe_len(parsed)}")
        return parsed

    if looks_like_path:
        raise FileNotFoundError(f"Expected input artifact at {original_input!r} was not available")
    log(f"[load_input_payload] source=inline type=str size={len(original_input)}")
    return input_value


def normalize_candidate_pairs(candidate_pairs):
    """Validate and normalize candidate-pair payloads into two-element pairs.

    类别：IO / payload 解析类
    """
    candidate_pairs = unwrap_string_payload(candidate_pairs)

    if isinstance(candidate_pairs, dict):
        if "candidate_pairs" in candidate_pairs:
            candidate_pairs = candidate_pairs["candidate_pairs"]
        elif "data" in candidate_pairs:
            candidate_pairs = candidate_pairs["data"]

    if isinstance(candidate_pairs, str):
        candidate_pairs = unwrap_string_payload(candidate_pairs)

    if not isinstance(candidate_pairs, list):
        raise ValueError(f"candidate_pairs must be a list. got={type(candidate_pairs).__name__}")

    normalized = []
    for item in candidate_pairs:
        if isinstance(item, tuple) and len(item) == 2:
            normalized.append((item[0], item[1]))
            continue
        if isinstance(item, list) and len(item) == 2:
            normalized.append((item[0], item[1]))
            continue
        raise ValueError("Each candidate_pairs item must contain exactly 2 elements.")
    return normalized

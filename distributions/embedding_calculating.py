import argparse
import ast
import hashlib
import json
import os
import sys
import time

import pandas as pd
from ruamel.yaml import YAML

from models.embedding_model import EmbeddingModel
from pipeline.embedding_training import train_embeddings
from pipeline.calculating_similarity import score_candidate_pairs

CONFIG_PATH = os.environ.get("EAER_CONFIG_PATH", "/app/config/examples/config-embedding.yaml")
WORKFLOW_NAME = os.environ.get("WORKFLOW_NAME", "").strip()
STATE_CACHE_PATH = f"/app/data/{WORKFLOW_NAME}/state_cache.json" if WORKFLOW_NAME else "/app/data/state_cache.json"
STATE_MANIFEST_PATH = "/app/data/state_manifest.json"


def _log(message: str):
    print(message, file=sys.stderr)


def _safe_len(value):
    try:
        return len(value)
    except Exception:
        return "n/a"


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


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256_file(path: str) -> str:
    if not os.path.exists(path):
        return ""
    digest = hashlib.sha256()
    with open(path, "rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_state_manifest(path: str = STATE_MANIFEST_PATH) -> dict:
    try:
        if not os.path.exists(path):
            return {"versions": {}}
        with open(path, "r", encoding="utf-8") as file_handle:
            loaded = json.load(file_handle)
        if isinstance(loaded, dict):
            loaded.setdefault("versions", {})
            return loaded
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        pass
    return {"versions": {}}


def _persist_state_manifest(manifest: dict, path: str = STATE_MANIFEST_PATH):
    manifest_dir = os.path.dirname(path)
    if manifest_dir:
        os.makedirs(manifest_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_handle:
        json.dump(manifest, file_handle)


def _register_manifest_artifact(config: dict, logical_name: str, artifact_path: str):
    version_name = str(config.get("version_name", "test"))
    manifest = _load_state_manifest()
    versions = manifest.setdefault("versions", {})
    version_entry = versions.setdefault(version_name, {})
    version_entry["updated_at"] = _now_iso()
    version_entry["last_run_id"] = WORKFLOW_NAME
    version_entry["window"] = {
        "id": os.environ.get("WINDOW_ID", ""),
        "reason": os.environ.get("WINDOW_REASON", ""),
        "record_count": os.environ.get("WINDOW_RECORD_COUNT", ""),
        "start_offset": os.environ.get("WINDOW_START_OFFSET", ""),
        "end_offset": os.environ.get("WINDOW_END_OFFSET", ""),
    }
    artifacts = version_entry.setdefault("artifacts", {})
    artifacts[logical_name] = {
        "path": artifact_path,
        "exists": os.path.exists(artifact_path),
        "sha256": _sha256_file(artifact_path),
        "updated_at": _now_iso(),
    }
    _persist_state_manifest(manifest)
    _log(
        f"[manifest] version={version_name} run_id={WORKFLOW_NAME} "
        f"window_id={version_entry['window'].get('id', '')} artifact={logical_name} path={artifact_path}"
    )


def _manifest_artifact_path(config: dict, logical_name: str) -> str:
    version_name = str(config.get("version_name", "test"))
    manifest = _load_state_manifest()
    path = (
        manifest.get("versions", {})
        .get(version_name, {})
        .get("artifacts", {})
        .get(logical_name, {})
        .get("path")
    )
    return path if isinstance(path, str) else ""


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


def _unwrap_string_payload(value):
    current = value
    for _ in range(5):
        if not isinstance(current, str):
            break
        stripped = current.strip()
        if not stripped:
            break

        parsed = _parse_json_payload(stripped)
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
    original_input = input_value
    if input_value.startswith("@"):
        argfile_path = input_value[1:]
        if os.path.isfile(argfile_path):
            input_value = argfile_path

    if os.path.isfile(input_value) and input_value.lower().endswith(".csv"):
        loaded = pd.read_csv(input_value)
        _log(f"[load_input_payload] source={input_value} type=DataFrame rows={len(loaded)} cols={len(loaded.columns)}")
        return loaded

    if os.path.isfile(input_value):
        with open(input_value, "r", encoding="utf-8") as file_handle:
            content = file_handle.read()
        parsed = _unwrap_string_payload(content)
        if parsed is not None:
            _log(f"[load_input_payload] source={input_value} type={type(parsed).__name__} size={_safe_len(parsed)}")
            return parsed
        _log(f"[load_input_payload] source={input_value} type=str size={len(content)}")
        return content

    parsed = _unwrap_string_payload(input_value)
    if parsed is not None:
        _log(f"[load_input_payload] source=inline type={type(parsed).__name__} size={_safe_len(parsed)}")
        return parsed
    _log(f"[load_input_payload] source=inline type=str size={len(original_input)}")
    return input_value


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


def _embedding_artifact_path(state_config: dict, config: dict) -> str:
    emb_dir = state_config.get("embedding-dir", "data/embedding")
    version_name = config.get("version_name", "test")
    emb_name = state_config.get("embedding_model-name") or version_name
    os.makedirs(emb_dir, exist_ok=True)
    return os.path.join(emb_dir, f"{emb_name}.emb")


def _looks_like_embedding_model(obj) -> bool:
    if obj is None:
        return False
    if isinstance(obj, EmbeddingModel):
        return True
    base = getattr(obj, "model", obj)
    return hasattr(base, "wv")

def _file_has_content(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0

def get(key):
    return STATE_CACHE.get(key)

def update(key, value):
    STATE_CACHE[key] = value
    _persist_state_cache()

    if key != "embedding_model":
        return

    config = load_config()
    state_cfg = _state_config(config)
    emb_path = _embedding_artifact_path(state_cfg, config)

    model = STATE_CACHE.get("embedding_model")
    if _looks_like_embedding_model(model):
        _log(f"[state] saving embedding_model to {emb_path}")
        model.save(emb_path)
    else:
        _log(f"[state] saving non-model embedding payload to {emb_path}")
        _write_text(emb_path, model)
    _register_manifest_artifact(config, "embedding_model", emb_path)


def ensure_embedding_model(config: dict):
    current = get("embedding_model")
    if _looks_like_embedding_model(current):
        _log("[state] embedding_model already in memory cache")
        return current

    state_cfg = _state_config(config)
    emb_path = _manifest_artifact_path(config, "embedding_model") or _embedding_artifact_path(state_cfg, config)
    _log(f"[state] embedding path={emb_path} exists={_file_has_content(emb_path)}")
    if _file_has_content(emb_path):
        try:
            model = EmbeddingModel.load(emb_path)
            STATE_CACHE["embedding_model"] = model
            _persist_state_cache()
            _log("[state] embedding_model loaded from disk")
            _register_manifest_artifact(config, "embedding_model", emb_path)
            return model
        except Exception:
            _log("[state] failed to load embedding_model from disk")
            pass
    _log("[state] embedding_model not found in cache or disk")
    return current


def _normalize_candidate_pairs(candidate_pairs):
    candidate_pairs = _unwrap_string_payload(candidate_pairs)

    if isinstance(candidate_pairs, dict):
        if "candidate_pairs" in candidate_pairs:
            candidate_pairs = candidate_pairs["candidate_pairs"]
        elif "data" in candidate_pairs:
            candidate_pairs = candidate_pairs["data"]

    if isinstance(candidate_pairs, str):
        candidate_pairs = _unwrap_string_payload(candidate_pairs)

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


def embedding_training(config, sequences):
    _log("[embedding_training] start")
    _log(f"[embedding_training] sequences type={type(sequences).__name__} size={_safe_len(sequences)}")
    model = ensure_embedding_model(config)
    model = train_embeddings(config, model, sequences)
    update("embedding_model", model)
    vocab_size = _safe_len(getattr(model, "wv", {}).key_to_index) if hasattr(model, "wv") else "n/a"
    _log(f"[embedding_training] done vocab_size={vocab_size} model_type={type(model).__name__}")
    return {"status": "embedding_trained"}


def calculating_similarity(config, candidate_pairs):
    _log("[calculating_similarity] start")
    embedding_model = ensure_embedding_model(config)
    if embedding_model is None:
        raise ValueError("embedding_model must be initialized or loaded before calculating_similarity.")
    candidate_pairs = _normalize_candidate_pairs(candidate_pairs)
    _log(f"[calculating_similarity] candidate_pairs={len(candidate_pairs)}")
    sim_cfg = config.get("similarity", {})
    batch_threshold = int(sim_cfg.get("batch_threshold", 2048))
    _log(f"[calculating_similarity] batch_threshold={batch_threshold}")
    candidate_pairs_score = score_candidate_pairs(embedding_model, candidate_pairs, batch_threshold=batch_threshold)
    preview = candidate_pairs_score[:3] if isinstance(candidate_pairs_score, list) else candidate_pairs_score
    _log(f"[calculating_similarity] score_count={_safe_len(candidate_pairs_score)} preview={preview}")
    return {
        "matching_pairs": candidate_pairs_score,
        "count": len(candidate_pairs_score),
    }


def run_argo_once(
    mode: str,
    function: str,
    sequences: str = "",
    candidate_pairs: str = "",
    output_path: str = "-",
):
    config = load_config()
    config["mode"] = mode
    config["function"] = function
    _log(f"[run] function={function} mode={mode} output_path={output_path}")

    if function == "embedding_training":
        if not isinstance(sequences, str) or not sequences.strip():
            raise ValueError("embedding_training requires --sequences input.")
        sequences_value = load_input_payload(sequences)
        output = embedding_training(config, sequences_value)
    elif function == "calculating_similarity":
        if not isinstance(candidate_pairs, str) or not candidate_pairs.strip():
            raise ValueError("calculating_similarity requires --candidate_pairs input.")
        candidate_pairs_value = load_input_payload(candidate_pairs)
        output = calculating_similarity(config, candidate_pairs_value)
    else:
        raise ValueError(f"Unsupported function: {function}")

    payload = json.dumps(serialize_for_json(output))
    if output_path and output_path != "-":
        with open(output_path, "w", encoding="utf-8") as file_handle:
            file_handle.write(payload)
    else:
        print(payload)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embedding training/similarity distribution for Argo")
    parser.add_argument("--mode", default="default")
    parser.add_argument("--sequences", default="")
    parser.add_argument("--candidate_pairs", default="")
    parser.add_argument("--function", default="")
    parser.add_argument("--output", default="-")
    args = parser.parse_args()
    run_argo_once(
        mode=args.mode,
        function=args.function,
        sequences=args.sequences,
        candidate_pairs=args.candidate_pairs,
        output_path=args.output,
    )

import hashlib
import json
import os
import sys
import time

from models.embedding_model import EmbeddingModel
from utils.pipeline_io import deserialize_from_json, serialize_for_json


WORKFLOW_NAME = os.environ.get("WORKFLOW_NAME", "").strip()
STATE_CACHE_PATH = f"/app/data/{WORKFLOW_NAME}/state_cache.json" if WORKFLOW_NAME else "/app/data/state_cache.json"
STATE_MANIFEST_PATH = "/app/data/state_manifest.json"


def _log(message: str):
    print(message, file=sys.stderr)


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


# Manifest lookup fallback disabled: use the configured embedding artifact path.
# def _manifest_artifact_path(config: dict, logical_name: str) -> str:
#     version_name = str(config.get("version_name", "test"))
#     manifest = _load_state_manifest()
#     path = (
#         manifest.get("versions", {})
#         .get(version_name, {})
#         .get("artifacts", {})
#         .get(logical_name, {})
#         .get("path")
#     )
#     return path if isinstance(path, str) else ""


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


def _state_config(config: dict) -> dict:
    if not isinstance(config, dict):
        return {}
    # Legacy fallback disabled: use state_management only.
    # cfg = config.get("state_management", {}) or config.get("state_config", {}) or {}
    cfg = config.get("state_management", {})
    return cfg if isinstance(cfg, dict) else {}


def embedding_artifact_path(config: dict) -> str:
    state_config = _state_config(config)
    emb_dir = state_config.get("embedding-dir", "data/embedding")
    version_name = config.get("version_name", "test")
    emb_name = state_config.get("embedding_model-name") or version_name
    os.makedirs(emb_dir, exist_ok=True)
    return os.path.join(emb_dir, f"{emb_name}.emb")


def looks_like_embedding_model(obj) -> bool:
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


def update(key, value, config: dict | None = None):
    STATE_CACHE[key] = value
    _persist_state_cache()

    if key != "embedding_model":
        return

    if config is None:
        # Fallback disabled: distribution entrypoints must pass config explicitly.
        # config = load_config()
        raise ValueError("config is required when saving embedding_model.")
    emb_path = embedding_artifact_path(config)

    model = STATE_CACHE.get("embedding_model")
    if not looks_like_embedding_model(model):
        # Fallback disabled: do not serialize arbitrary non-model payloads as text.
        # write_text(emb_path, model, serializer=serialize_for_json)
        raise TypeError(f"embedding_model must be an EmbeddingModel-like object, got {type(model).__name__}.")

    _log(f"[state] saving embedding_model to {emb_path}")
    model.save(emb_path)
    _register_manifest_artifact(config, "embedding_model", emb_path)


def ensure_embedding_model(config: dict, force_reload: bool = False):
    current = get("embedding_model")
    if looks_like_embedding_model(current) and not force_reload:
        _log("[state] embedding_model already in memory cache")
        return current

    # Manifest path fallback disabled: use the configured embedding artifact path.
    # emb_path = _manifest_artifact_path(config, "embedding_model") or embedding_artifact_path(config)
    emb_path = embedding_artifact_path(config)
    _log(f"[state] embedding path={emb_path} exists={_file_has_content(emb_path)}")
    return load_embedding_model(config, emb_path)


def load_embedding_model(config: dict, emb_path: str):
    _log(f"[state] attempting to load embedding_model path={emb_path} exists={_file_has_content(emb_path)}")
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
            # Fallback disabled: do not return a stale cached object after a failed disk load.
            # pass
            raise
    _log("[state] embedding_model at specified path is missing or empty")
    # Fallback disabled: do not return cached object when the configured artifact is missing.
    # return get("embedding_model")
    return None

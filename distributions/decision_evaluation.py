import argparse
import ast
import json
import os
import sys

import pandas as pd
from ruamel.yaml import YAML

from models.embedding_model import EmbeddingModel
from models.similarity_graph import SimilarityGraph
from pipeline.decision_making import decide_matches
from pipeline.evaluation import compare_ground_truth

CONFIG_PATH = os.environ.get("EAER_CONFIG_PATH", "/app/config/examples/config-embedding.yaml")
STATE_CACHE_PATH = "/app/cache/state_cache.json"


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
        parsed = _parse_json_payload(content)
        if parsed is not None:
            _log(f"[load_input_payload] source={input_value} type={type(parsed).__name__} size={_safe_len(parsed)}")
            return parsed
        _log(f"[load_input_payload] source={input_value} type=str size={len(content)}")
        return content

    parsed = _parse_json_payload(input_value)
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


def _predicted_artifact_path(state_config: dict, config: dict, output_format: str) -> str:
    predicted_dir = state_config.get("predicted_match-dir", "data/predicted")
    version_name = config.get("version_name", "test")
    predicted_name = state_config.get("predicted_match-name") or version_name
    extension = ".graphml" if str(output_format).lower() == "graphml" else ".txt"
    os.makedirs(predicted_dir, exist_ok=True)
    return os.path.join(predicted_dir, f"{predicted_name}{extension}")


def _result_artifact_path(state_config: dict, config: dict) -> str:
    predicted_dir = state_config.get("predicted_match-dir", "data/predicted")
    version_name = config.get("version_name", "test")
    predicted_name = state_config.get("predicted_match-name") or version_name
    os.makedirs(predicted_dir, exist_ok=True)
    return os.path.join(predicted_dir, f"{predicted_name}.result.txt")


def _clean_graph_copy(graph):
    graph_copy = graph.copy()
    allowed_types = (str, int, float, bool)
    for vertex in graph_copy.vs:
        for attr in list(vertex.attributes()):
            if not isinstance(vertex[attr], allowed_types):
                del vertex[attr]
    for edge in graph_copy.es:
        for attr in list(edge.attributes()):
            if not isinstance(edge[attr], allowed_types):
                del edge[attr]
    return graph_copy


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

    config = load_config()
    state_cfg = _state_config(config)
    output_format = config.get("similarity", {}).get("output_format", config.get("output_format", "graphml"))
    config["output_format"] = output_format

    if key == "embedding_model":
        emb_path = _embedding_artifact_path(state_cfg, config)
        model = STATE_CACHE.get("embedding_model")
        if _looks_like_embedding_model(model):
            _log(f"[state] saving embedding_model to {emb_path}")
            model.save(emb_path)
        else:
            _log(f"[state] saving non-model embedding payload to {emb_path}")
            _write_text(emb_path, model)
        return

    if key == "predicted_matching":
        predicted_path = _predicted_artifact_path(state_cfg, config, output_format)
        predicted = STATE_CACHE.get("predicted_matching")
        if isinstance(predicted, SimilarityGraph):
            predicted_save = _clean_graph_copy(predicted.graph)
            predicted_save.write_graphml(predicted_path)
            _log(f"[state] saved predicted_matching graph to {predicted_path}")
        else:
            _write_text(predicted_path, predicted)
            _log(f"[state] saved predicted_matching payload to {predicted_path}")
        return

    if key in {"result", "evaluation_result"}:
        result_path = _result_artifact_path(state_cfg, config)
        _write_text(result_path, STATE_CACHE.get(key))
        _log(f"[state] saved {key} to {result_path}")
        return


def ensure_embedding_model(config: dict):
    current = get("embedding_model")
    if _looks_like_embedding_model(current):
        _log("[state] embedding_model already in memory cache")
        return current

    state_cfg = _state_config(config)
    emb_path = _embedding_artifact_path(state_cfg, config)
    _log(f"[state] embedding path={emb_path} exists={_file_has_content(emb_path)}")
    if _file_has_content(emb_path):
        try:
            model = EmbeddingModel.load(emb_path)
            STATE_CACHE["embedding_model"] = model
            _persist_state_cache()
            _log("[state] embedding_model loaded from disk")
            return model
        except Exception:
            _log("[state] failed to load embedding_model from disk")
            pass
    _log("[state] embedding_model not found in cache or disk")
    return current


def _normalize_matching_pairs(matching_pairs):
    if isinstance(matching_pairs, dict):
        if "matching_pairs" in matching_pairs:
            matching_pairs = matching_pairs["matching_pairs"]
        elif "data" in matching_pairs:
            matching_pairs = matching_pairs["data"]

    if isinstance(matching_pairs, str):
        parsed = _parse_json_payload(matching_pairs)
        if parsed is not None:
            matching_pairs = parsed
        else:
            try:
                matching_pairs = ast.literal_eval(matching_pairs)
            except Exception:
                pass

    if not isinstance(matching_pairs, list):
        raise ValueError(f"matching_pairs must be a list. got={type(matching_pairs).__name__}")

    normalized = []
    for item in matching_pairs:
        if isinstance(item, tuple) and len(item) == 3:
            normalized.append((str(item[0]), str(item[1]), float(item[2])))
            continue
        if isinstance(item, list) and len(item) == 3:
            normalized.append((str(item[0]), str(item[1]), float(item[2])))
            continue
        raise ValueError("Each matching_pairs item must contain exactly 3 elements.")
    return normalized


def decision_making(config, matching_pairs):
    _log("[decision_making] start")
    model = ensure_embedding_model(config)
    if model is None:
        raise ValueError("embedding_model must be initialized or loaded before decision_making.")

    matching_pairs = _normalize_matching_pairs(matching_pairs)
    previous_pairs = get("predicted_matching_pairs")
    if previous_pairs is not None:
        previous_pairs = _normalize_matching_pairs(previous_pairs)

    output_format = config.get("similarity", {}).get("output_format", "graphml")
    config["output_format"] = output_format
    final_pairs, predicted_graph = decide_matches(
        mutualtop_pairs=matching_pairs,
        previous_pairs=previous_pairs,
        model=model,
        output_format=output_format,
    )
    update("predicted_matching_pairs", final_pairs)
    update("predicted_matching", predicted_graph)
    _log(f"[decision_making] done pair_count={len(final_pairs)}")
    _log(f"[decision_making] final_pairs={final_pairs[:5]}{'...' if len(final_pairs) > 5 else ''}")
    return {"status": "decision_completed", "pair_count": len(final_pairs)}


def evaluation(config):
    _log("[evaluation] start")
    output_format = config.get("similarity", {}).get("output_format", "graphml")
    config["output_format"] = output_format
    result = compare_ground_truth(config)
    update("result", result)
    _log("[evaluation] done")
    return result


def run_argo_once(
    mode: str,
    function: str,
    matching_pairs: str = "",
    output_path: str = "-",
):
    config = load_config()
    config["mode"] = mode
    selected_function = function.strip() if isinstance(function, str) else ""
    if not selected_function:
        selected_function = "decision_making" if isinstance(matching_pairs, str) and matching_pairs.strip() else "evaluation"
    config["function"] = selected_function
    _log(f"[run] function={selected_function} mode={mode} output_path={output_path}")

    if selected_function == "decision_making":
        if not isinstance(matching_pairs, str) or not matching_pairs.strip():
            raise ValueError("decision_making requires --matching_pairs input.")
        matching_pairs_value = load_input_payload(matching_pairs)
        output = decision_making(config, matching_pairs_value)
    elif selected_function == "evaluation":
        output = evaluation(config)
    else:
        raise ValueError(f"Unsupported function: {selected_function}")

    payload = json.dumps(serialize_for_json(output))
    if output_path and output_path != "-":
        with open(output_path, "w", encoding="utf-8") as file_handle:
            file_handle.write(payload)
    else:
        print(payload)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decision making/evaluation distribution for Argo")
    parser.add_argument("--mode", default="default")
    parser.add_argument("--matching_pairs", default="")
    parser.add_argument("--function", default="")
    parser.add_argument("--output", default="-")
    args = parser.parse_args()
    run_argo_once(
        mode=args.mode,
        function=args.function,
        matching_pairs=args.matching_pairs,
        output_path=args.output,
    )

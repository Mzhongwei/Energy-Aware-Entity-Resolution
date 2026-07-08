import argparse
import json
import os
import sys
import signal
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from ruamel.yaml import YAML

from models.representation_graph import RepresentationGraph
from pipeline.graph_construction import dyn_graph_generation
from pipeline.random_walk import dynrandom_walks_generation
from utils.buffers import (
    get_earliest_window_index,
    load_earliest_buffer,
    write_buffer,
    wait_for_buffer,
    write_eos,
    delete_earliest_buffer_file,
    get_manifest_file,
)
from utils.pipeline_io import deserialize_from_json, parse_json_payload, serialize_for_json, write_step_output, write_text

CONFIG_PATH = os.environ.get("EAER_CONFIG_PATH", "/app/config/examples/config-embedding.yaml")
STATE_CACHE_PATH = "/app/data/state_cache.json"
BUFFER_PATH = "/app/data/buffers/"
GRAPH_SNAPSHOT_DIR = os.environ.get(
    "EAER_GRAPH_SNAPSHOT_DIR",
    os.path.join(BUFFER_PATH, "graph_snapshots"),
)

# Safer default for incremental mode: do not rebuild dyn_roots from the full graph.
# Rebuilding all roots can make random_walk run on the whole graph for every window.
ALLOW_FULL_ROOT_BOOTSTRAP = os.environ.get(
    "EAER_ALLOW_FULL_ROOT_BOOTSTRAP",
    "false",
).lower() in {"1", "true", "yes", "y"}

stop_requested = False

def _log(message: str):
    """Write distribution diagnostics to stderr.

    类别：诊断和小工具类
    """
    print(message, file=sys.stderr)


def handle_sigterm(signum, frame):
    """Record termination requests so long-running buffer loops can stop cleanly.

    类别：Pod / Argo 入口类
    """
    global stop_requested
    stop_requested = True


signal.signal(signal.SIGTERM, handle_sigterm)
signal.signal(signal.SIGINT, handle_sigterm)


def load_config(config_path: str = CONFIG_PATH):
    """Load the mounted pipeline YAML config into a dictionary.

    类别：IO / payload 解析类
    """
    if not os.path.exists(config_path):
        return {}
    yaml = YAML(typ="safe")
    with open(config_path, "r", encoding="utf-8") as file_handle:
        loaded = yaml.load(file_handle) or {}
    return loaded if isinstance(loaded, dict) else {}


def _resolve_cache_file(path: str) -> str:
    """Resolve a state-cache path when the configured value is a directory.

    类别：数据管理类
    """
    if os.path.isdir(path):
        return os.path.join(path, "state_cache.json")
    return path


def _load_state_cache(path: str = STATE_CACHE_PATH):
    """Load this distribution state cache from disk if it exists.

    类别：数据管理类
    """
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
    """Persist the in-memory distribution state cache to disk.

    类别：数据管理类
    """
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
    """Parse JSON payloads written by upstream steps, including serialized DataFrames.

    类别：IO / payload 解析类
    """
    return parse_json_payload(content, deserializer=deserialize_from_json)


def load_processed_data(processed_data_value: str):
    """Load processed data from a file path or serialized inline payload.

    类别：IO / payload 解析类
    """
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
    """Write plain text or serialized structured data to a file path.

    类别：IO / payload 解析类
    """
    write_text(path, value, serializer=serialize_for_json)


def _state_config(config: dict) -> dict:
    """Extract state-management configuration from the loaded pipeline config.

    类别：数据管理类
    """
    if not isinstance(config, dict):
        return {}
    cfg = config.get("state_management", {}) or config.get("state_config", {}) or {}
    return cfg if isinstance(cfg, dict) else {}


def _graph_artifact_path(state_config: dict, config: dict) -> str:
    """Resolve where the representation graph artifact should be stored.

    类别：数据管理类
    """
    graph_dir = state_config.get("graph-dir", "/app/data/graph")
    version_name = config.get("version_name", "test")
    graph_name = state_config.get("graph-name") or version_name
    os.makedirs(graph_dir, exist_ok=True)
    return os.path.join(graph_dir, f"{graph_name}.graphml")


def _graph_manifest_path(state_config: dict, config: dict) -> str:
    """Resolve where graph metadata should be stored.

    类别：数据管理类
    """
    graph_dir = state_config.get("graph-dir", "/app/data/graph")
    version_name = config.get("version_name", "test")
    graph_name = state_config.get("graph-name") or version_name
    os.makedirs(graph_dir, exist_ok=True)
    return os.path.join(graph_dir, f"{graph_name}.manifest.json")


def _graph_config(config: dict) -> dict:
    """Extract graph-related configuration with safe defaults.

    类别：数据管理类
    """
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
    """Return a graph object prepared for durable serialization.

    类别：数据管理类
    """
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
    """Check whether a path exists and contains data.

    类别：数据管理类
    """
    return os.path.exists(path) and os.path.getsize(path) > 0


def _resolve_graph_paths(config: dict):
    """Resolve graph artifact and manifest paths from the config.

    类别：数据管理类
    """
    state_cfg = _state_config(config)
    return _graph_artifact_path(state_cfg, config), _graph_manifest_path(state_cfg, config)


def _merge_config_from_manifest(config: dict, manifest_path: str):
    """Merge graph metadata from a manifest into the current config.

    类别：数据管理类
    """
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
    """Load a persisted representation graph and merge its metadata.

    类别：数据管理类
    """
    merged_config = _merge_config_from_manifest(config, manifest_path)

    if _file_has_content(graph_path):
        print(f"[INFO] loading graph from {graph_path}", file=sys.stderr)
        graph = dyn_graph_generation(merged_config)
        graph.load_graph(graph_path)
        return graph

    print(f"[INFO] creating fresh graph (no existing file)", file=sys.stderr)
    return dyn_graph_generation(config)


def get(key):
    """Read an object from this distribution local state cache.

    类别：数据管理类
    """
    return STATE_CACHE.get(key)


def _as_igraph(graph):
    """Return the igraph object behind a RepresentationGraph-like wrapper.

    类别：数据管理类
    """
    if hasattr(graph, "get_graph"):
        return graph.get_graph()
    if hasattr(graph, "graph"):
        return graph.graph
    return graph


def _root_count(dyn_roots) -> int:
    """Count dynamic random-walk roots across root groups.

    类别：数据管理类
    """
    if isinstance(dyn_roots, set):
        return len(dyn_roots)
    if isinstance(dyn_roots, dict):
        return sum(len(v) for v in dyn_roots.values() if hasattr(v, "__len__"))
    return 0


def _extract_root_names(graph, roots):
    """Store both indices and names so roots can be restored after GraphML reload.

    类别：数据管理类
    """
    i_graph = _as_igraph(graph)
    indices = []
    names = []

    for root in roots or []:
        try:
            idx = int(root)
        except (TypeError, ValueError):
            continue
        indices.append(idx)
        try:
            vertex = i_graph.vs[idx]
            if "name" in vertex.attributes():
                names.append(str(vertex["name"]))
        except Exception:
            pass

    return {
        "indices": sorted(set(indices)),
        "names": sorted(set(names)),
    }


def _serialize_dyn_roots(graph) -> dict:
    """Serialize dynamic random-walk roots for cross-pod handoff.

    类别：数据管理类
    """
    dyn_roots = getattr(graph, "dyn_roots", None)

    if isinstance(dyn_roots, set):
        return {
            "kind": "set",
            "total": len(dyn_roots),
            "roots": _extract_root_names(graph, dyn_roots),
        }

    if isinstance(dyn_roots, dict):
        items = {}
        total = 0
        for key, roots in dyn_roots.items():
            root_set = set(roots or [])
            total += len(root_set)
            items[str(key)] = _extract_root_names(graph, root_set)
        return {
            "kind": "dict",
            "total": total,
            "items": items,
        }

    return {
        "kind": "unknown",
        "total": 0,
        "type": type(dyn_roots).__name__,
    }


def _restore_root_set(graph, root_payload: dict) -> set:
    """Restore a set of graph root vertices from serialized root names.

    类别：数据管理类
    """
    i_graph = _as_igraph(graph)
    name_to_index = {}
    try:
        for vertex in i_graph.vs:
            if "name" in vertex.attributes():
                name_to_index[str(vertex["name"])] = int(vertex.index)
    except Exception:
        pass

    restored = set()

    for name in root_payload.get("names", []) or []:
        if str(name) in name_to_index:
            restored.add(name_to_index[str(name)])

    # Fallback to stored indices only when a name could not be resolved.
    vertex_count = len(i_graph.vs) if hasattr(i_graph, "vs") else 0
    for idx in root_payload.get("indices", []) or []:
        try:
            idx = int(idx)
        except (TypeError, ValueError):
            continue
        if 0 <= idx < vertex_count:
            restored.add(idx)

    return restored


def _restore_dyn_roots(graph, dyn_roots_payload: Optional[dict]) -> bool:
    """Restore dynamic random-walk root groups onto a loaded graph.

    类别：数据管理类
    """
    if not isinstance(dyn_roots_payload, dict):
        return False

    kind = dyn_roots_payload.get("kind")
    if kind == "set":
        graph.dyn_roots = _restore_root_set(graph, dyn_roots_payload.get("roots", {}))
        _log(f"[roots] restored set dyn_roots={len(graph.dyn_roots)}")
        return True

    if kind == "dict":
        restored = {}
        for key, root_payload in (dyn_roots_payload.get("items", {}) or {}).items():
            restored[str(key)] = _restore_root_set(graph, root_payload)
        graph.dyn_roots = restored
        _log(f"[roots] restored dict dyn_roots={[(k, len(v)) for k, v in restored.items()]}")
        return True

    _log(f"[roots] unsupported dyn_roots payload kind={kind!r}")
    return False


def _write_graph_manifest(path: str, config: dict):
    """Write metadata needed to reload graph artifacts consistently.

    类别：数据管理类
    """
    manifest = {
        "graph_class": "RepresentationGraph",
        "graph_config": _graph_config(config),
        "meta_path": config.get("meta_path", []),
    }
    manifest_path = get_manifest_file(path)
    manifest_dir = os.path.dirname(manifest_path)
    if manifest_dir:
        os.makedirs(manifest_dir, exist_ok=True)
    temp_manifest_path = f"{manifest_path}.tmp"
    with open(temp_manifest_path, "w", encoding="utf-8") as file_handle:
        json.dump(manifest, file_handle, ensure_ascii=False, indent=2)
    os.replace(temp_manifest_path, manifest_path)


def _write_graphml_atomic(graph, graph_path: str, config: dict):
    """Write graphml through a temporary file and atomically replace the target.

    类别：数据管理类
    """
    graph_dir = os.path.dirname(graph_path)
    if graph_dir:
        os.makedirs(graph_dir, exist_ok=True)

    i_graph = _as_igraph(graph)
    graph_save = _clean_graph_copy(i_graph)
    temp_graph_path = f"{graph_path}.tmp"
    graph_save.write_graphml(temp_graph_path)
    os.replace(temp_graph_path, graph_path)
    _write_graph_manifest(graph_path, config)


def _make_graph_handoff(config: dict, graph, window_index) -> dict:
    """
    Build a small JSON buffer for the random-walk pod.

    The important part is dyn_roots: GraphML does not reliably preserve this
    Python-side incremental state, so the random-walk pod must receive it
    explicitly. Otherwise it may rebuild roots from the whole graph and walk
    far more nodes than intended.
    类别：数据管理类
    """
    graph_path = os.path.join(GRAPH_SNAPSHOT_DIR, f"graph_window_{window_index}.graphml")
    _write_graphml_atomic(graph, graph_path, config)

    dyn_roots_payload = _serialize_dyn_roots(graph)
    _log(
        f"[handoff] window={window_index} graph_path={graph_path} "
        f"dyn_roots_total={dyn_roots_payload.get('total', 0)}"
    )

    return {
        "type": "representation_graph_handoff",
        "graph_path": graph_path,
        "manifest_path": get_manifest_file(graph_path),
        "dyn_roots": dyn_roots_payload,
        "window_index": window_index,
        "delete_graph_snapshot": True,
    }


def _normalize_graph_handoff(payload) -> Tuple[str, Optional[dict], bool]:
    """Accept the new JSON handoff and legacy string graph-path buffers.

    类别：数据管理类
    """
    if isinstance(payload, dict):
        graph_path = payload.get("graph_path") or payload.get("path")
        if not isinstance(graph_path, str) or not graph_path.strip():
            raise ValueError(f"Invalid graph handoff payload: missing graph_path. payload={payload!r}")
        return graph_path, payload.get("dyn_roots"), bool(payload.get("delete_graph_snapshot", False))

    if isinstance(payload, str):
        return payload, None, False

    raise ValueError(f"Unsupported graph handoff payload type: {type(payload).__name__}")


def _cleanup_graph_handoff(payload):
    """Remove temporary graph handoff artifacts after a downstream step consumes them.

    类别：数据管理类
    """
    if not isinstance(payload, dict) or not payload.get("delete_graph_snapshot"):
        return

    graph_path = payload.get("graph_path")
    if not isinstance(graph_path, str) or not graph_path:
        return

    for path in (graph_path, get_manifest_file(graph_path)):
        try:
            if path and os.path.exists(path):
                os.remove(path)
                _log(f"[cleanup] removed {path}")
        except OSError as exc:
            _log(f"[cleanup] failed to remove {path}: {exc}")


def _bootstrap_dyn_roots_if_empty(graph, config):
    """Initialize random-walk roots from the full graph when allowed and necessary.

    类别：数据管理类
    """
    if not hasattr(graph, "get_graph") or not hasattr(graph, "dyn_roots"):
        print("[bootstrap] graph or dyn_roots missing", file=sys.stderr)
        return

    g = graph.get_graph()
    dyn_roots = graph.dyn_roots

    # Legacy fallback only. Do not use in incremental mode unless explicitly enabled,
    # because it can rebuild a root set as large as the whole graph.
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


def ensure_representation_graph(config: dict, force_reload: bool = False):
    """Load or initialize the representation graph for graph construction.

    类别：数据管理类
    """
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
    """Load a graph from an explicit graph artifact path.

    类别：数据管理类
    """
    manifest_path = get_manifest_file(graph_path)
    graph = _load_representation_graph(config, graph_path, manifest_path)

    STATE_CACHE["representation_graph"] = graph
    if persist_cache:
        _persist_state_cache()
    return graph


def update(key, value):
    """Update local state and persist durable artifacts when required.

    类别：数据管理类
    """
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
    """Update the representation graph from one incremental processed-data window.

    类别：业务包装类
    """
    graph = ensure_representation_graph(config, force_reload=True)
    return graph_construction(graph, processed_data)


def batch_graph_construction(config, processed_data):
    """Build or update the representation graph for batch mode.

    类别：业务包装类
    """
    graph = ensure_representation_graph(config)
    return graph_construction(graph, processed_data)


def graph_construction(graph, processed_data):
    """Build or update a representation graph from processed records.

    类别：业务包装类
    """
    if not isinstance(processed_data, pd.DataFrame):
        raise ValueError("processed_data must be a pandas DataFrame for graph construction.")
    if not hasattr(graph, "build_relation"):
        raise ValueError("representation_graph must support build_relation for incremental updates.")

    graph.build_relation(processed_data)
    update("representation_graph", graph)
    g = graph.get_graph()

    sample_vertices = [v["name"] for v in g.vs[:5] if "name" in v.attributes()]
    print(f"[graph] sample_vertices={sample_vertices}", file=sys.stderr)
    print(f"[graph] written ids example: {g.vs[0]['id'] if len(g.vs) > 0 and 'id' in g.vs[0].attributes() else 'N/A'}", file=sys.stderr)
    print(f"[graph] dyn_roots_count={_root_count(getattr(graph, 'dyn_roots', None))}", file=sys.stderr)

    return {"status": "graph_built"}


def incremental_random_walk(config, graph_handoff):
    """Run random walk after loading graph state from an incremental handoff.

    类别：业务包装类
    """
    graph_path, dyn_roots_payload, _delete_snapshot = _normalize_graph_handoff(graph_handoff)
    graph = load_graph_from_path(config, graph_path)

    restored = _restore_dyn_roots(graph, dyn_roots_payload)
    if not restored:
        _log("[roots] no dyn_roots payload found in graph handoff")

    return random_walk(
        graph,
        config,
        allow_full_bootstrap=ALLOW_FULL_ROOT_BOOTSTRAP,
    )


def batch_random_walk(config):
    """Run random walk from the batch representation graph state.

    类别：业务包装类
    """
    graph = ensure_representation_graph(config)
    return random_walk(graph, config, allow_full_bootstrap=True)


def random_walk(graph, config, allow_full_bootstrap: bool = False):
    """Generate random-walk sequences from the current representation graph.

    类别：业务包装类
    """
    has_get_graph = hasattr(graph, "get_graph")
    has_dyn_roots = hasattr(graph, "dyn_roots")

    if not has_get_graph or not has_dyn_roots:
        return []

    if _root_count(graph.dyn_roots) == 0:
        if allow_full_bootstrap:
            _bootstrap_dyn_roots_if_empty(graph, config)
        else:
            _log("[random_walk] dyn_roots is empty; skipping instead of walking the whole graph")
            return []

    _log(f"[random_walk] dyn_roots_count={_root_count(graph.dyn_roots)}")
    return dynrandom_walks_generation(config, graph)


def _exit(output_path=None, output=None, output_buffer_path=None):
    """Write final output or EOS markers before the distribution process exits.

    类别：Pod / Argo 入口类
    """
    if output_buffer_path:
        write_eos(output_buffer_path, reason=f"timeout_no_initial_buffer")
    write_step_output(output_path, output, serializer=serialize_for_json)


def run_argo_batch(mode: str, function: str, processed_data: str, output_path: str = "-"):
    """Dispatch a batch Argo invocation to the requested business function.

    类别：Pod / Argo 入口类
    """
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
    """Dispatch an incremental worker loop to the requested business function.

    类别：Pod / Argo 入口类
    """
    config = load_config()
    config["function"] = function

    if function == "graph_construction":
        load_buffer_path = BUFFER_PATH + "processed_data"
        output_buffer_path = BUFFER_PATH + "graph"
        # The graph buffer is now a JSON handoff:
        # {graph_path, dyn_roots, window_index, ...}
        extension = "json"
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
            if stop_requested:
                sys.exit(0)
            next_ready = wait_for_buffer(load_buffer_path, timeout_seconds=30)
            data = load_earliest_buffer(load_buffer_path) if next_ready is not None else None
            continue

        window_index = get_earliest_window_index(load_buffer_path)

        if function == "graph_construction":
            incremental_graph_construction(config, data)
            output = _make_graph_handoff(config, get("representation_graph"), window_index)
        elif function == "random_walk":
            output = incremental_random_walk(config, data)
        else:
            raise ValueError(f"Unsupported function: {function}")

        write_buffer(output, output_buffer_path, window_index, extension=extension)

        if function == "random_walk":
            _cleanup_graph_handoff(data)

        delete_earliest_buffer_file(load_buffer_path)
        if stop_requested:
            sys.exit(0)
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

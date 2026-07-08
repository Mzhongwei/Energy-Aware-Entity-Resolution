import argparse
import os
import signal
import sys

from pipeline.calculating_similarity import score_mutual_top1_candidate_pairs
from state_io.embedding_state import ensure_embedding_model, load_embedding_model
from utils.buffers import (
    _delete_file_if_exists,
    delete_earliest_buffer_file,
    get_earliest_window_index,
    load_earliest_buffer,
    wait_for_buffer,
    wait_for_embedding_buffer,
    write_buffer,
    write_eos,
)
from utils.config_io import load_config
from utils.pipeline_io import serialize_for_json, write_step_output
from embedding_common import (
    load_input_payload,
    log,
    normalize_candidate_pairs,
    safe_len,
    sample_ids_from_pairs,
    sample_pairs,
)


CONFIG_PATH = os.environ.get("EAER_CONFIG_PATH", "/app/config/examples/config-embedding.yaml")
BUFFER_PATH = "/app/data/buffers/"
FUNCTION = "calculating_similarity"

stop_requested = False


def handle_sigterm(signum, frame):
    """Record termination requests so long-running buffer loops can stop cleanly.

    类别：Pod / Argo 入口类
    """
    global stop_requested
    stop_requested = True


signal.signal(signal.SIGTERM, handle_sigterm)
signal.signal(signal.SIGINT, handle_sigterm)


def _exit(output_path=None, output=None, output_buffer_path=None):
    """Write final output or EOS markers before the distribution process exits.

    类别：Pod / Argo 入口类
    """
    if output_buffer_path:
        write_eos(output_buffer_path, reason="timeout_no_initial_buffer")
    write_step_output(output_path, output, serializer=serialize_for_json)


def run_calculating_similarity(config, candidate_pairs, embedding_model):
    """Score candidate pairs with the embedding model and return mutual top-1 matches.

    类别：业务包装类
    """
    if embedding_model is None:
        raise ValueError("embedding_model must be initialized or loaded before calculating_similarity.")

    candidate_pairs = normalize_candidate_pairs(candidate_pairs)
    log(f"[calculating_similarity] candidate_pairs={len(candidate_pairs)}")
    log(f"[calculating_similarity] candidate_pairs_sample={sample_pairs(candidate_pairs)}")

    if not candidate_pairs:
        log("[calculating_similarity] candidate_pairs is empty; skipping scoring and returning empty result")
        return {
            "matching_pairs": [],
            "count": 0,
        }

    sim_cfg = config.get("similarity", {})
    batch_threshold = int(sim_cfg.get("batch_threshold", 2048))
    log(f"[calculating_similarity] batch_threshold={batch_threshold}")

    try:
        base_kv = getattr(embedding_model, "wv", getattr(embedding_model, "model", None))
        vocab_size = len(base_kv.key_to_index) if base_kv and hasattr(base_kv, "key_to_index") else 0
        log(f"[diagnostic] embedding vocab_size={vocab_size}")
        sample_keys = list(base_kv.key_to_index)[:10] if vocab_size else []
        log(f"[diagnostic] embedding sample_keys={sample_keys}")
        sample_ids = sample_ids_from_pairs(candidate_pairs)
        log(f"[diagnostic] candidate sample_ids={sample_ids}")
        missing = [
            item
            for item in set(sample_ids)
            if not (base_kv and hasattr(base_kv, "key_to_index") and item in base_kv.key_to_index)
        ]
        if missing:
            log(f"[diagnostic] missing_ids_sample={missing[:20]}")
    except Exception as exc:
        log(f"[diagnostic] failed to introspect embedding model: {exc}")

    candidate_pairs_score = score_mutual_top1_candidate_pairs(
        embedding_model,
        candidate_pairs,
        batch_threshold=batch_threshold,
    )
    preview = candidate_pairs_score[:3] if isinstance(candidate_pairs_score, list) else candidate_pairs_score
    log(f"[calculating_similarity] score_count={safe_len(candidate_pairs_score)} preview={preview}")
    return {
        "matching_pairs": candidate_pairs_score,
        "count": len(candidate_pairs_score),
    }


def run_batch(mode: str, candidate_pairs: str = "", output_path: str = "-"):
    """Execute this entrypoint once with file/path inputs supplied by the batch workflow.

    类别：Pod / Argo 入口类
    """
    if not isinstance(candidate_pairs, str) or not candidate_pairs.strip():
        raise ValueError("calculating_similarity requires --candidate_pairs input.")

    config = load_config(CONFIG_PATH)
    config["mode"] = mode
    config["function"] = FUNCTION
    embedding_model = ensure_embedding_model(config)
    output = run_calculating_similarity(config, load_input_payload(candidate_pairs), embedding_model)
    _exit(output_path=output_path, output=output)


def run_incremental(output_path: str = "-"):
    """Run this entrypoint as an incremental buffer-driven worker loop.

    类别：Pod / Argo 入口类
    """
    config = load_config(CONFIG_PATH)
    config["function"] = FUNCTION

    load_buffer_path = BUFFER_PATH + "candidate_pairs"
    load_emb_buffer_path = BUFFER_PATH + "embedding_calculating"
    output_buffer_path = BUFFER_PATH + "matching_pairs"

    first_ready = wait_for_buffer(load_buffer_path, timeout_seconds=600)
    if first_ready is None:
        print("[INFO] No incoming buffer within startup timeout; writing EOS and exiting.", file=sys.stderr)
        _exit(output_path=output_path, output_buffer_path=output_buffer_path)
        return

    data = load_earliest_buffer(load_buffer_path)
    while data is not None:
        embedding_ready = wait_for_buffer(load_emb_buffer_path, timeout_seconds=30)
        if embedding_ready is None:
            print("[INFO] No embedding model buffer received within timeout; writing EOS and exiting.", file=sys.stderr)
            _exit(output_path=output_path, output_buffer_path=output_buffer_path)
            return

        window_index = get_earliest_window_index(load_buffer_path)
        embedding_path = wait_for_embedding_buffer(load_emb_buffer_path, window_index, timeout_seconds=30)
        if embedding_path is None:
            print("[INFO] No matching embedding model buffer received within timeout; writing EOS and exiting.", file=sys.stderr)
            _exit(output_path=output_path, output_buffer_path=output_buffer_path)
            return
        log(
            f"[incremental_calculating_similarity] window={window_index} "
            f"candidate_buffer={load_buffer_path} embedding_path={embedding_path}"
        )

        embedding_model = load_embedding_model(config, embedding_path)
        try:
            normalized_preview = normalize_candidate_pairs(data)
            log(
                f"[incremental_calculating_similarity] candidate_pairs_count={len(normalized_preview)} "
                f"sample_pairs={sample_pairs(normalized_preview)} sample_ids={sample_ids_from_pairs(normalized_preview)}"
            )
        except Exception as exc:
            log(f"[incremental_calculating_similarity] failed to preview candidate_pairs: {exc}")

        output = run_calculating_similarity(config, data, embedding_model)
        log(
            f"[incremental_calculating_similarity] window={window_index} "
            f"output_count={safe_len(output.get('matching_pairs') if isinstance(output, dict) else output)}"
        )

        write_buffer(output, output_buffer_path, window_index, extension="json")
        _delete_file_if_exists(embedding_path)
        delete_earliest_buffer_file(load_buffer_path)

        if stop_requested:
            sys.exit(0)

        wait_for_buffer(load_buffer_path, timeout_seconds=30)
        data = load_earliest_buffer(load_buffer_path)

    _exit(output_path=output_path, output_buffer_path=output_buffer_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculating similarity entrypoint")
    parser.add_argument("--mode", default="default")
    parser.add_argument("--candidate_pairs", default="")
    parser.add_argument("--output", default="-")
    args = parser.parse_args()

    if "embedding" in args.mode and "inference" in args.mode and "training" not in args.mode:
        run_incremental(output_path=args.output)
    else:
        run_batch(mode=args.mode, candidate_pairs=args.candidate_pairs, output_path=args.output)

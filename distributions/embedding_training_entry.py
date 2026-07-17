## disabled
import argparse
import os
import signal
import sys

from pipeline.embedding_training import train_embeddings
from embedding_state import ensure_embedding_model, get, update
from utils.pipeline_io import (
    delete_earliest_buffer_file,
    get_earliest_window_index,
    load_earliest_buffer,
    wait_for_buffer,
    write_buffer,
    write_eos,
)
from utils.pipeline_io import load_config
from utils.pipeline_io import serialize_for_json, write_step_output
from embedding_common import load_input_payload, log, safe_len


CONFIG_PATH = os.environ.get("EAER_CONFIG_PATH", "/app/config/examples/config-embedding.yaml")
BUFFER_PATH = "/app/data/buffers/"
FUNCTION = "embedding_training"

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
    write_step_output(
        os.path.dirname(output_path) or ".",
        os.path.splitext(os.path.basename(output_path))[0],
        os.path.splitext(os.path.basename(output_path))[1].lstrip("."),
        output,
        serializer=serialize_for_json,
    )


def run_embedding_training(config, sequences, force_reload: bool = False):
    """Run the embedding training business step and persist the updated model state.

    类别：业务包装类
    """
    model = ensure_embedding_model(config, force_reload=force_reload)
    model = train_embeddings(config, model, sequences)
    update("embedding_model", model, config=config)
    vocab_size = safe_len(getattr(model, "wv", {}).key_to_index) if hasattr(model, "wv") else "n/a"
    log(f"[embedding_training] done vocab_size={vocab_size} model_type={type(model).__name__}")
    return {"status": "embedding_trained"}


def run_batch(mode: str, sequences: str = "", output_path: str = "-"):
    """Execute this entrypoint once with file/path inputs supplied by the batch workflow.

    类别：Pod / Argo 入口类
    """
    if not isinstance(sequences, str) or not sequences.strip():
        raise ValueError("embedding_training requires --sequences input.")

    config = load_config(CONFIG_PATH)
    config["mode"] = mode
    config["function"] = FUNCTION
    output = run_embedding_training(config, load_input_payload(sequences), force_reload=False)
    _exit(output_path=output_path, output=output)


def run_incremental(output_path: str = "-"):
    """Run this entrypoint as an incremental buffer-driven worker loop.

    类别：Pod / Argo 入口类
    """
    config = load_config(CONFIG_PATH)
    config["function"] = FUNCTION

    load_buffer_path = BUFFER_PATH + "sequences"
    output_buffer_path = BUFFER_PATH + "embedding"

    first_ready = wait_for_buffer(load_buffer_path, timeout_seconds=600)
    if first_ready is None:
        print("[INFO] No incoming buffer within startup timeout; writing EOS and exiting.", file=sys.stderr)
        _exit(output_path=output_path, output_buffer_path=output_buffer_path)
        return

    data = load_earliest_buffer(load_buffer_path)
    while data is not None:
        run_embedding_training(config, data, force_reload=True)

        window_index = get_earliest_window_index(load_buffer_path)
        model = get("embedding_model")
        print(f"[DEBUG] model type : {type(model)}")

        write_buffer(model, output_buffer_path + "_calculating", window_index, extension="emb")
        write_buffer(model, output_buffer_path + "_decision", window_index, extension="emb")
        delete_earliest_buffer_file(load_buffer_path)

        if stop_requested:
            sys.exit(0)

        wait_for_buffer(load_buffer_path, timeout_seconds=30)
        data = load_earliest_buffer(load_buffer_path)

    _exit(output_path=output_path, output_buffer_path=output_buffer_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embedding training entrypoint")
    parser.add_argument("--mode", default="default")
    parser.add_argument("--sequences", default="")
    parser.add_argument("--output", default="-")
    args = parser.parse_args()

    if "embedding" in args.mode and "inference" in args.mode and "training" not in args.mode:
        run_incremental(output_path=args.output)
    else:
        run_batch(mode=args.mode, sequences=args.sequences, output_path=args.output)

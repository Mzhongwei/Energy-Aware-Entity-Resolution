import argparse
import gc

from utils.pipeline_io import (
    BufferIO,
    StreamStage,
    acknowledge_window_checkpoint,
    is_empty_payload,
    is_window_checkpoint_acknowledged,
    load_config,
    resolve_checkpoint_reference,
)
from pipeline.graph_construction import load_graph_reader, restore_dyn_roots
from pipeline.random_walk import dynrandom_walks_generation

"""
task: random walk sequence generation
mode: incremental + embedding
input: graph handoff (snapshot path + dyn_roots) [buffer]
output: walk sequences [buffer]
description: dyn_roots restored from the handoff drive each window's walk directly --
no whole-graph traversal is needed since graph construction already carries the roots.
"""

INPUT_DATA_TYPE = "graph"
OUTPUT_DATA_TYPE = "sequences"


def main():
    parser = argparse.ArgumentParser(description="Worker entry for incremental random walk.")
    parser.add_argument("--config", default="/app/config/examples/config-embedding.yaml")
    parser.add_argument("--workload", default="default")
    args = parser.parse_args()

    config = load_config(args.config)
    stage = StreamStage(
        "random_walk",
        BufferIO(args.workload, INPUT_DATA_TYPE, OUTPUT_DATA_TYPE, config),
        is_empty=is_empty_payload,
    )

    def process(window):
        handoff = window.take()
        checkpoint_dir = handoff.get("checkpoint_dir", "")
        checkpoint_window = int(handoff.get("checkpoint_window", window.index))
        if not checkpoint_dir:
            raise ValueError(f"[random_walk] missing checkpoint_dir for window {window.index}")
        if is_window_checkpoint_acknowledged(checkpoint_dir, checkpoint_window):
            return
        handoff = resolve_checkpoint_reference(handoff)
        graph = load_graph_reader(config, handoff["checkpoint_path"])
        restore_dyn_roots(graph, handoff.get("dyn_roots"))
        del handoff

        sequences = dynrandom_walks_generation(config, graph) if graph.dyn_roots else []
        # The next window loads a complete graph snapshot. Release this window's graph
        # before serializing walks so the old and new snapshots never overlap in memory.
        del graph
        gc.collect()

        stage.send(window.index, sequences)
        del sequences
        gc.collect()

        acknowledge_window_checkpoint(checkpoint_dir, checkpoint_window)

    stage.run(process)


if __name__ == "__main__":
    main()

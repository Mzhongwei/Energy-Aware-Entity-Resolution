# Energy-Aware Entity Resolution

This folder contains the Python implementation of the Energy-Aware Entity Resolution pipeline used by the Kubernetes and Argo setup in the repository root.

## What Runs Here

The project supports two execution families:

- `embedding-*` - graph-based entity resolution with normalization, graph construction, random walks, embedding training, candidate generation, similarity, and decision stages.
- `bert-*` - sequence-pair classification with normalization, BERT training, inference, and evaluation stages.

Batch embedding training runs six persistent stage Pods, and Kafka inference runs its own distributed workers. Both paths call the same functions under [pipeline](pipeline). Training stages are selected with `EmbTrai-training.py --stage <task> --workload <workflow-name>`.

## Layout

- [main_distribution.py](main_distribution.py) - pipeline entrypoint and mode dispatcher. It will not work with the current Kubernetes setup.
- [pipeline](pipeline) - reusable pipeline stages and task helpers.
- [models](models) - runtime model wrappers and graph/index data structures.
- [services](services) - producer, consumer, and simulator utilities.
- [scripts](scripts) - helper scripts used by ConfigMap generation and runtime wiring.
- [config](config) - example runtime configuration files.
- [requirements/](requirements/) - dependency sets for entry images and legacy deployment profiles.
- [entries/batch/EmbTrai-training.py](entries/batch/EmbTrai-training.py) - bounded-window CSV/JSONL embedding training entry.
- [entries/worker](entries/worker) - long-running Kafka window workers.

## Modes

The active mode is read from the runtime config and determines which pipeline branch executes.

### Embedding

- `embedding-training` - normalization -> graph construction -> random walk -> embedding training -> CG feature extraction -> feature index construction.
- `embedding-inference` - normalization -> graph construction -> random walk -> embedding training -> CG feature extraction -> feature index construction -> candidate enumeration -> similarity -> decision making.
- `embedding-evaluation` - evaluation on top of an already loaded state.

### BERT

- `bert-training` - normalization -> BERT training.
- `bert-inference` - normalization -> BERT inference.
- `bert-b_evaluation` - normalization -> BERT evaluation.

## Windowed Embedding Training

The Argo training DAG starts six Pods together, one per task. Each Pod handles all windows for its task; it does not create a Pod per window:

```text
normalization ──→ graph-construction → random-walk → embedding-training
             └─→ cg-feature-extraction → feature-index-construction
```

Normalization reads `data_source_A` in bounded windows. Each window is published to both branches. The next window waits for embedding and index acknowledgments, bounding intermediate files to one window. Graph construction keeps its graph in memory and sends an atomic GraphML snapshot plus root names to random walk. Random walk loads the snapshot, prepares samplers, writes sequences, and removes the consumed snapshot. The model and index remain in their owning Pods across windows.

Handoffs use the shared communication PVC under `<workflow-name>/communication/embedding-training`, including graph snapshots so readers can run on different nodes. The graph, model and index are saved to the existing model PVCs on EOS. Incremental Jobs start only after **all six** training Pods succeed.

Stage exceptions publish a failure marker to stop waiting peers. Automatic stage retries are disabled: the handoff protocol is not a resumable checkpoint protocol. Restart a failed training run with a fresh run/version, rather than retrying an individual stage. A configurable handoff timeout catches workers lost without publishing an error (for example OOM or node loss). Configure it with:

```yaml
batch_processing:
  rows_per_batch: 10000
  max_bytes_per_batch: 134217728  # JSONL input-byte limit
  poll_interval_seconds: 0.5
  handoff_timeout_seconds: 86400  # maximum wait for a peer; increase for very long windows
  # input_format: jsonl           # optional when the extension is unambiguous
```

Supported file types are `.csv`, `.jsonl`, and `.ndjson`. CSV is bounded by row count; JSONL is bounded by both row count and input bytes. The graph, embedding vocabulary, and candidate index intentionally accumulate across windows, so windowing bounds transient DataFrames/walks/features rather than total model-state memory.

For JSONL with stable source IDs, set `record_ids.source_field`; otherwise training IDs are generated from the configured left prefix. Kafka simulation accepts the same three file types through the existing `csv.file.path` property and continues feeding the existing worker pipeline.

## Runtime Data Flow

Distributed entries exchange window artifacts through shared storage; Python objects stay inside each process. The legacy `main_distribution.py` uses a shared runtime dictionary. Typical values include:

- `raw_data`
- `processed_data`
- `sequences`
- `cg_feature`
- `candidate_pairs`
- `matching_pairs`
- `predicted_matching`
- `evaluation_result`

Stateful stages use the in-memory objects in [models](models) and [pipeline](pipeline), then persist or reuse them as needed across runs.

## Local Hints

For the Kubernetes-backed pipeline, regenerate the ConfigMaps from the repository root after changing an entry script:

```bash
bash k8s/pipeline/configmaps.sh code/Energy-Aware-Entity-Resolution/config/examples/config-embedding.yaml
```

## Notes

- This folder is the source of truth for the containerized pipeline logic.
- The root README that documents the Kubernetes and Argo workflow is in a the following repository: [k8s-python-llm](https://github.com/kevin-oulai/k8s-python-llm).

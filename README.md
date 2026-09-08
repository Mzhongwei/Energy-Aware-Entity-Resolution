# Energy-Aware Entity Resolution

This folder contains the Python implementation of the Energy-Aware Entity Resolution pipeline used by the Kubernetes and Argo setup in the repository root.

## What Runs Here

The project supports two execution families:

- `embedding-*` - graph-based entity resolution with normalization, graph construction, random walks, embedding training, candidate generation, similarity, and decision stages.
- `bert-*` - sequence-pair classification with normalization, BERT training, inference, and evaluation stages.

Batch embedding training uses one windowed entry, while Kafka inference keeps its distributed workers. Both paths call the same functions under [pipeline](pipeline); only input delivery and lifecycle differ.

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

`EmbTrai-training.py` reads `data_source_A` incrementally and runs normalization, graph update, random walk, embedding update, feature extraction, and index update for each window. Configure it with:

```yaml
batch_processing:
  rows_per_batch: 10000
  max_bytes_per_batch: 134217728  # JSONL input-byte limit
  # input_format: jsonl           # optional when the extension is unambiguous
```

Supported file types are `.csv`, `.jsonl`, and `.ndjson`. CSV is bounded by row count; JSONL is bounded by both row count and input bytes. The graph, embedding vocabulary, and candidate index intentionally accumulate across windows, so windowing bounds transient DataFrames/walks/features rather than total model-state memory.

For JSONL with stable source IDs, set `record_ids.source_field`; otherwise training IDs are generated from the configured left prefix. Kafka simulation accepts the same three file types through the existing `csv.file.path` property and continues feeding the existing worker pipeline.

## Runtime Data Flow

The pipeline passes a shared runtime dictionary through the stages. Typical values include:

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
bash k8s/scripts/erctl.sh configmaps embedding
```

## Notes

- This folder is the source of truth for the containerized pipeline logic.
- The root README that documents the Kubernetes and Argo workflow is in a the following repository: [k8s-python-llm](https://github.com/kevin-oulai/k8s-python-llm).

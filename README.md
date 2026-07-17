# Energy-Aware Entity Resolution

This folder contains the Python implementation of the Energy-Aware Entity Resolution pipeline used by the Kubernetes and Argo setup in the repository root.

## What Runs Here

The project supports two execution families:

- `embedding-*` - graph-based entity resolution with normalization, graph construction, random walks, embedding training, candidate generation, similarity, and decision stages.
- `bert-*` - sequence-pair classification with normalization, BERT training, inference, and evaluation stages.

The main orchestrator is [main_distribution.py](main_distribution.py), while the files under [distributions](distributions) are the Argo/Kubernetes entrypoints that wrap the same pipeline logic for each stage.

## Layout

- [main_distribution.py](main_distribution.py) - pipeline entrypoint and mode dispatcher. It will not work with the current Kubernetes setup.
- [pipeline](pipeline) - reusable pipeline stages and task helpers.
- [models](models) - runtime model wrappers and graph/index data structures.
- [services](services) - producer, consumer, and simulator utilities.
- [scripts](scripts) - helper scripts used by ConfigMap generation and runtime wiring.
- [config](config) - example runtime configuration files.
- [requirements/](requirements/) - dependency sets for entry images and legacy deployment profiles.
- [distributions](distributions) - per-stage wrappers used by the container images.

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

## Useful Entry Points

- [distributions/normalization_distribution.py](distributions/normalization_distribution.py)
- [distributions/graph_randomwalk.py](distributions/graph_randomwalk.py)
- [distributions/embedding_training_entry.py](distributions/embedding_training_entry.py)
- [distributions/calculating_similarity_entry.py](distributions/calculating_similarity_entry.py)
- [distributions/featureindex_candidate.py](distributions/featureindex_candidate.py)
- [distributions/decision_evaluation.py](distributions/decision_evaluation.py)
- [distributions/bert_distribution.py](distributions/bert_distribution.py)
- [distributions/cg_feature_distribution.py](distributions/cg_feature_distribution.py)

## Local Hints

For the Kubernetes-backed pipeline, regenerate the ConfigMaps from the repository root after changing any of the distribution scripts:

```bash
bash k8s/scripts/erctl.sh configmaps embedding
```

## Notes

- This folder is the source of truth for the containerized pipeline logic.
- The root README that documents the Kubernetes and Argo workflow is in a the following repository: [k8s-python-llm](https://github.com/kevin-oulai/k8s-python-llm).

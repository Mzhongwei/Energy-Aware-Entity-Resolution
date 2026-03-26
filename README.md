# Energy-Aware ER

## 1. Overview

This project implements an entity resolution pipeline that supports two execution families:

- `embedding-*`: graph-based incremental or batch-style entity resolution with random walk embeddings, candidate generation, similarity calculation, and decision making.
- `bert-*`: sequence-pair classification with BERT-style training, inference, and evaluation.

The runtime entry point is [main_distribution.py](/home/zhongwei/Data_integration/energy_aware_er/main_distribution.py).  
Pipeline state is managed in memory through [stateManager.py](/home/zhongwei/Data_integration/energy_aware_er/governance/stateManager.py), and can be persisted at the end of a run.

The current implementation mixes two levels:

- pipeline modules under [pipeline](/home/zhongwei/Data_integration/energy_aware_er/pipeline) provide reusable building blocks
- endpoint functions in [main_distribution.py](/home/zhongwei/Data_integration/energy_aware_er/main_distribution.py) orchestrate task execution and interact with `StateManager`


## 2. Tasks

The DAG tasks are defined in [main_distribution.py](/home/zhongwei/Data_integration/energy_aware_er/main_distribution.py).

- `normalization`
  Normalizes input records. In embedding mode it calls `index_normalization(...)`. In bert mode it converts paired data into sentence-pair format with `sequence_generating_m1(...)`.

- `graph_construction`
  Builds or updates the representation graph. When the input is a `DataFrame`, it can use [graph_construction.py](/home/zhongwei/Data_integration/energy_aware_er/pipeline/graph_construction.py); otherwise it falls back to placeholder in-memory behavior.

- `random_walk`
  Generates walk sequences from the graph. When a real graph object is present, it uses [random_walk.py](/home/zhongwei/Data_integration/energy_aware_er/pipeline/random_walk.py).

- `embedding_training`
  Trains or retrains the embedding model from walk sequences using [embedding_training.py](/home/zhongwei/Data_integration/energy_aware_er/pipeline/embedding_training.py).

- `bert_training`
  Trains a BERT-style classifier using [bert_training.py](/home/zhongwei/Data_integration/energy_aware_er/pipeline/bert_training.py).

- `cg_feature_extraction`
  Computes candidate-generation features from a `DataFrame` using [cg_feature_extraction.py](/home/zhongwei/Data_integration/energy_aware_er/pipeline/cg_feature_extraction.py).

- `feature_index_construction`
  Builds or updates the candidate-generation index in memory using [feature_index_construction.py](/home/zhongwei/Data_integration/energy_aware_er/pipeline/feature_index_construction.py).

- `candidate_enumeration`
  Queries the committed CG index and generates candidate pairs using [candidate_enumeration.py](/home/zhongwei/Data_integration/energy_aware_er/pipeline/candidate_enumeration.py).

- `calculating_similarity`
  Placeholder stage in the current `main_distribution.py`. Dedicated similarity utilities live in [calculating_similarity.py](/home/zhongwei/Data_integration/energy_aware_er/pipeline/calculating_similarity.py).

- `decision_making`
  Placeholder stage in the current `main_distribution.py`. Decision utilities live in [decision_making.py](/home/zhongwei/Data_integration/energy_aware_er/pipeline/decision_making.py).

- `bert_inference`
  Runs inference with a saved BERT model through [bert_inference.py](/home/zhongwei/Data_integration/energy_aware_er/pipeline/bert_inference.py).

- `evaluation`
  Current endpoint-level evaluation stage that writes result data back into `StateManager`.

- `bert_evaluation`
  Evaluates a saved BERT model through [bert_evaluation.py](/home/zhongwei/Data_integration/energy_aware_er/pipeline/bert_evaluation.py).


## 3. Modes

Execution mode is read from `config["mode"]` in [main_distribution.py](/home/zhongwei/Data_integration/energy_aware_er/main_distribution.py).

### Embedding Modes

- `embedding-training`
  Runs:
  `normalization -> graph_construction -> random_walk -> embedding_training -> cg_feature_extraction -> feature_index_construction`

- `embedding-inference`
  Runs:
  `normalization -> graph_construction -> random_walk -> embedding_training -> cg_feature_extraction -> feature_index_construction -> candidate_enumeration -> calculating_similarity -> decision_making`

  In the current code this mode is Kafka-oriented: it loads state once, processes buffered Kafka messages in batches, updates the same in-memory state, and saves at the end.

- `embedding-evaluation`
  Runs only the `evaluation` stage on top of loaded state.

### BERT Modes

- `bert-training`
  Runs:
  `normalization -> bert_training`

- `bert-inference`
  Runs:
  `normalization -> bert_inference`

- `bert-b_evaluation`
  Runs:
  `normalization -> bert_evaluation`


## 4. Data Flow

### Common Flow

The execution engine is `run_pipeline(...)` in [main_distribution.py](/home/zhongwei/Data_integration/energy_aware_er/main_distribution.py).

For each task:

1. It checks task dependencies.
2. It collects the required inputs from `data_store`.
3. It executes the endpoint function.
4. It stores returned outputs back into `data_store`.

Stateful artifacts are not primarily passed through return values. They are stored in:

- `data_store["state_manager"]`
- `state_manager.cache`

### Embedding Flow

Typical embedding flow is:

1. Raw records enter `normalization`
2. Normalized records update the graph in `graph_construction`
3. The graph produces walk sequences in `random_walk`
4. Sequences retrain the same `EmbeddingModel` in `embedding_training`
5. Records are converted to CG features in `cg_feature_extraction`
6. Features are inserted into `CGIndex` in `feature_index_construction`
7. Candidate pairs are generated in `candidate_enumeration`
8. Similarity and decision stages produce predicted matches

In Kafka mode, the same `StateManager` instance is reused across batches, so the same in-memory graph, embedding model, and candidate index can be updated incrementally.

### BERT Flow

Typical BERT flow is:

1. CSV data is loaded into `raw_data`
2. `normalization` converts records into pairwise text input
3. `bert_training` trains a classifier or `bert_inference` / `bert_evaluation` reuses a saved one
4. Results are written into `state_manager.cache`


## 5. Core Data Structures

The current system revolves around a few central runtime objects.

- `StateManager`
  Defined in [stateManager.py](/home/zhongwei/Data_integration/energy_aware_er/governance/stateManager.py).  
  Holds all mutable runtime state in `cache`, including graph, embedding model, candidate index, predicted matches, and evaluation results.

- `RepresentationGraph` / `DynGraphIgraph`
  Graph structures used by graph construction and random walk generation.  
  `DynGraphIgraph` is implemented in [graph_construction.py](/home/zhongwei/Data_integration/energy_aware_er/pipeline/graph_construction.py).

- `EmbeddingModel`
  Wrapper around gensim models, implemented in [embedding_model.py](/home/zhongwei/Data_integration/energy_aware_er/models/embedding_model.py).  
  Supports initialization, retraining, save, and load.

- `CGIndex`
  Candidate-generation index, implemented in [cg_index.py](/home/zhongwei/Data_integration/energy_aware_er/models/cg_index.py).  
  Supports methods such as `fullindexing`, `key-blocking`, `token-blocking`, and `minhash-lsh`.

- `bert_model`
  BERT-related state is stored in cache as a dictionary containing model/trainer/tokenizer artifacts, depending on stage.

- `data_store`
  The runtime data exchange dictionary in [main_distribution.py](/home/zhongwei/Data_integration/energy_aware_er/main_distribution.py).  
  It carries transient data such as `raw_data`, `processed_data`, `sequences`, `cg_feature`, `candidate_pairs`, and the shared `state_manager`.


## Current Notes

- Some endpoint functions in [main_distribution.py](/home/zhongwei/Data_integration/energy_aware_er/main_distribution.py) still contain placeholder fallback logic.
- The project is already partially refactored into pipeline modules, but endpoint orchestration and pipeline implementations are still being aligned.
- The intended runtime pattern is: load state once, update in memory during the pipeline, and save once at the end.

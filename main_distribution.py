import argparse
import json
import os
import argparse
from typing import Any
from pandas import DataFrame
from ruamel.yaml import YAML
import pandas as pd
from confluent_kafka import Consumer, KafkaException, KafkaError

from governance import StateManager
from pipeline import (
    sequence_generating_m1,
    index_normalization,
    train_model,
    evaluate_from_saved_model,
    process_inference,
    train_embeddings,
    compute_features,
)
from pipeline.candidate_enumeration import generate_candidates_from_index
from pipeline.feature_index_construction import build_index as build_cg_index, commit_cg_index, create_cg_index
from pipeline.graph_construction import dyn_graph_generation
from pipeline.random_walk import dynrandom_walks_generation
from utils.write_log import write_log

# =========================
# endpoints
# =========================

def normalization(config: dict, raw_data: dict | DataFrame):
    print("[normalization]")
    if 'embedding' in config['mode']:
        # in incremental mode, we index records and normalize
 
        # # data example
        # raw_data = pd.DataFrame(
        #     data = {
        #         "name": ["kkk", "ttt", ["hhh", "JJJ"]],
        #         "adress": ["d ? rue", "yes addre", "ad . r"]
        #     }
        # )

        index_normalization(config, raw_data)
    else:
        # for bert mode, we do not need to index the records. We normalize records values and generate directly the appropriate df structure 
        for k, df in raw_data.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                raw_data[k] = sequence_generating_m1(df)

    processed_data = raw_data
    return processed_data

def graph_construction(config, processed_data, state_manager: StateManager):
    graph = state_manager.get("representation_graph")

    if isinstance(processed_data, pd.DataFrame):
        if graph is None:
            print("[INIT GRAPH]")
            graph = dyn_graph_generation(config)
        else:
            print("[UPDATE GRAPH]")

        if hasattr(graph, "build_relation"):
            graph.build_relation(processed_data)
            state_manager.update("representation_graph", graph)
            return None

    if graph is None:
        print("[INIT GRAPH]")
        graph = {"nodes": []}
    else:
        print("[UPDATE GRAPH]")

    graph["nodes"].append(len(graph["nodes"]))
    state_manager.update("representation_graph", graph)

    return None


def random_walk(config, state_manager: StateManager):
    graph = state_manager.get("representation_graph")
    print("[random_walk]")
    if hasattr(graph, "get_graph") and hasattr(graph, "dyn_roots"):
        walks_number = int(config.get("walks", {}).get("walks_number", 0))
        return dynrandom_walks_generation(config, graph, walk_nums=walks_number)
    return "sequences"


def embedding_training(config, sequences, state_manager: StateManager):
    print("[embedding_training]")
    model = state_manager.get("embedding_model")
    model = train_embeddings(config, model, sequences)
    state_manager.update("embedding_model", model)
    return None


def bert_training(config, processed_data, state_manager: StateManager):
    print("[bert_training]")
    trainer, tokenizer = train_model(config, processed_data)
    state_manager.update("bert_model", {"trainer": trainer, "tokenizer": tokenizer})
    return None


def cg_feature_extraction(config, processed_data):
    print("[cg_feature_extraction]")
    if isinstance(processed_data, pd.DataFrame):
        method = config.get("candidate_generation", {}).get("method", "fullindexing")
        return compute_features(processed_data, method, config)
    return {"feature": 1}


def feature_index_construction(config, cg_feature, state_manager: StateManager):
    print("[feature_index_construction]")
    if isinstance(cg_feature, list):
        index = state_manager.get("cg_feature_index")
        if index is None:
            method = config.get("candidate_generation", {}).get("method", "fullindexing")
            index = create_cg_index(method, config)
        if hasattr(index, "upsert"):
            build_cg_index(cg_feature, index)
            commit_cg_index(index)
            state_manager.update("cg_feature_index", index)
            return None

    index = state_manager.get("cg_feature_index") or {}
    index["size"] = index.get("size", 0) + 1
    state_manager.update("cg_feature_index", index)
    return None


def candidate_enumeration(config, cg_feature, state_manager: StateManager):
    print("[candidate_enumeration]")
    index = state_manager.get("cg_feature_index")
    if isinstance(cg_feature, list) and index is not None and hasattr(index, "query"):
        top_k = config.get("candidate_generation", {}).get("top_k")
        return generate_candidates_from_index(cg_feature, index, top_k=top_k)
    return ["pair1", "pair2"]


def calculating_similarity(config, candidate_pairs, state_manager: StateManager):
    print("[calculating_similarity]")
    return ["match1"]


def decision_making(config, matching_pairs, state_manager: StateManager):
    print("[decision_making]")
    predicted_matching = ["predicted_matching"]
    state_manager.update("predicted_matching", predicted_matching)
    return None


def bert_inference(config, processed_data, state_manager: StateManager):
    print("[bert_inference]") 
    predicted_pairs = process_inference(processed_data, state_manager)
    state_manager.update("predicted_matching", predicted_pairs)
    return None


def evaluation(config, state_manager: StateManager):
    print("[evaluation]")
    result = state_manager.get("predicted_matching")
    state_manager.update("result", result)
    return None

def bert_evaluation(config, processed_data, state_manager: StateManager):
    print("[evaluation]")
    result = evaluate_from_saved_model(processed_data, config)
    state_manager.update("evaluation_result", result)
    return None


# =========================
# TASKS
# =========================

TASKS = {
    "normalization": {
        "deps": [],
        "input": ["raw_data"],
        "output": ["processed_data"],
        "func": normalization
    },
    "graph_construction": {
        "deps": ["normalization"],
        "input": ["processed_data", "state_manager"],
        "output": [],
        "func": graph_construction
    },
    "random_walk": {
        "deps": ["graph_construction"],
        "input": ["state_manager"],
        "output": ["sequences"],
        "func": random_walk
    },
    "embedding_training": {
        "deps": ["random_walk"],
        "input": ["sequences", "state_manager"],
        "output": [],
        "func": embedding_training
    },
    "bert_training": {
        "deps": ["normalization"],
        "input": ["processed_data", "state_manager"],
        "output": [],
        "func": bert_training
    },
    "cg_feature_extraction": {
        "deps": ["normalization"],
        "input": ["processed_data"],
        "output": ["cg_feature"],
        "func": cg_feature_extraction
    },
    "feature_index_construction": {
        "deps": ["cg_feature_extraction"],
        "input": ["cg_feature", "state_manager"],
        "output": [],
        "func": feature_index_construction
    },
    "candidate_enumeration": {
        "deps": ["feature_index_construction"],
        "input": ["cg_feature", "state_manager"],
        "output": ["candidate_pairs"],
        "func": candidate_enumeration
    },
    "calculating_similarity": {
        "deps": ["candidate_enumeration"],
        "input": ["candidate_pairs", "state_manager"],
        "output": ["matching_pairs"],
        "func": calculating_similarity
    },
    "decision_making": {
        "deps": ["calculating_similarity"],
        "input": ["matching_pairs", "state_manager"],
        "output": [],
        "func": decision_making
    },
    "bert_inference": {
        "deps": ["normalization"],
        "input": ["processed_data", "state_manager"],
        "output": [],
        "func": bert_inference
    },
    "evaluation": {
        "deps": [],
        "input": ["state_manager"],
        "output": [],
        "func": evaluation
    },
    "bert_evaluation": {
        "deps": ["normalization"],
        "input": ["processed_data", "state_manager"],
        "output": [],
        "func": bert_evaluation
    },
}

# =========================
# Pipeline
# =========================


def can_run(task_name, finished, tasks):
    deps = tasks[task_name].get("deps", [])
    rel = tasks[task_name].get("rel", "and")

    if not deps:
        return True

    if rel == "and":
        return all(dep in finished for dep in deps)
    elif rel == "or":
        return any(dep in finished for dep in deps)

    return False

def run_pipeline(config, tasks, data_store):

    finished = set()

    while len(finished) < len(tasks):

        progress = False

        for name, task in tasks.items():

            if name in finished:
                continue

            # 1. Determine whether it can be executed
            if not can_run(name, finished, tasks):
                continue

            # 2. Check if all required information has been provided
            inputs = {}
            missing = False

            for key in task.get("input", []):
                if key in data_store:
                    inputs[key] = data_store[key]
                else:
                    missing = True
                    break

            if missing:
                continue

            print(f"[RUN] {name}")

            func = task["func"]

            # 3. execute the missions
            try:
                output = func(config=config, **inputs)
            except Exception as e:
                raise RuntimeError(f"Task {name} failed: {e}")

            # 4. process output
            output_keys = task.get("output", [])

            if len(output_keys) == 0:
                pass

            elif len(output_keys) == 1:
                data_store[output_keys[0]] = output

            else:
                # Multiple outputs --> return a dictionary
                if not isinstance(output, dict):
                    raise ValueError(f"{name} must return dict for multiple outputs")

                for k in output_keys:
                    if k not in output:
                        raise ValueError(f"{name} missing output: {k}")
                    data_store[k] = output[k]

            finished.add(name)
            progress = True

        if not progress:
            raise RuntimeError("DAG stuck! Check dependencies or inputs.")

    return data_store

# =========================
# Driver
# =========================

def safe_read_csv(path):
    if not path:
        return pd.DataFrame()

    if not os.path.exists(path):
        return pd.DataFrame()

    return pd.read_csv(path)

def kafka_driver(config):
    try:
        # prepare kafka consumer
        consumer = Consumer({
            'bootstrap.servers': f'{config["kafka"]["bootstrap_servers"]}:{config["kafka"]["port"]}',
            'group.id': config['kafka']["groupid"],
            'auto.offset.reset': 'latest',   # latest / earliest
            'enable.auto.commit': False
        })

        # subscribe a topic
        consumer.subscribe([config['kafka']['topicid']])

    except Exception as e:
        app_logger = write_log(config["log"]["path"], "app", "app")
        app_logger.error(f"Fatal error in consumer service: {str(e)}")
        print(f"Fatal error in consumer service: {str(e)}")
        return
    return consumer

def driver(config):
    mode = config["mode"].split("-")
    state = StateManager()
    data_store = {
        "state_manager": state,
        "raw_data": None,
    }
    
    if "embedding" in mode:
        if "training" in mode:
            stages = [
                "normalization",
                "graph_construction",
                "random_walk",
                "embedding_training",
                "cg_feature_extraction",
                "feature_index_construction",
                ]
            sub_tasks = {k: TASKS[k] for k in stages}
            data_store['state_manager'].load(config, stages)
            data_store = run_pipeline(config, sub_tasks, data_store)
            data_store['state_manager'].save(config, stages)
        if "inference" in mode:
            
            stages = [
                "normalization",
                "graph_construction",
                "random_walk",
                "embedding_training",
                "cg_feature_extraction",
                "feature_index_construction",
                "candidate_enumeration",
                "calculating_similarity",
                "decision_making",
                ]
            sub_tasks = {k: TASKS[k] for k in stages}
            # check
            if not data_store['state_manager'].cache:
                data_store['state_manager'].load(config, stages)
            # start kafka
            poll_timeout = 500
            max_empty_polls = 500
            empty_poll_count = 0
            consumer = kafka_driver(config)

            data_buffer = []
            while True:
                
                msg = consumer.poll(poll_timeout)  # Non-blocking batch pull
                

                if msg is None:  # no new message
                    empty_poll_count += 1
                    if empty_poll_count >= max_empty_polls:
                        print("[INFO] No new messages for a while. Exiting consumer loop.")
                        if data_buffer:
                            data_store['raw_data'] = data_buffer
                            data_store = run_pipeline(config, sub_tasks, data_store)
                        break
                    continue
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        # end of partition
                        continue
                    else:
                        raise KafkaException(msg.error())
                empty_poll_count = 0  # reset counter
                
                # get message
                metadata = json.loads(msg.value().decode('utf-8'))
                data_buffer.append(metadata)
                if len(data_buffer) >= config["kafka"]["window_count"]:
                    data_store['raw_data'] = data_buffer
                    data_store = run_pipeline(config, sub_tasks, data_store)
                    data_buffer = []
            state_manager = data_store['state_manager']
            state_manager.save(config, stages)
        if "evaluation" in mode:
            
            stages = ["evaluation"]
            sub_tasks = {k: TASKS[k] for k in stages}
            # check
            if not data_store['state_manager'].cache:
                data_store['state_manager'].load(config, stages)
            data_store = run_pipeline(config, sub_tasks, data_store)
            state_manager = data_store['state_manager']
            state_manager.save(config, stages)

    elif "bert" in mode:
        # bert mode
        # data_store['raw_data'] is a dict of df
        if "training" in mode:
            stages = ["normalization", "bert_training"]
    
            data_store['raw_data'] = {
            'train': safe_read_csv(config.get("trainset_path")),
            'eval':  safe_read_csv(config.get("evalset_path")),
        }
            sub_tasks = {k: TASKS[k] for k in stages}
            data_store = run_pipeline(config, sub_tasks, data_store)
            state_manager = data_store['state_manager']
            state_manager.save(config, stages)
        if "inference" in mode:
            stages = ["normalization", "bert_inference"]
            data_store['raw_data'] = {
                'test': safe_read_csv(config.get("testset_path"))
            }

            sub_tasks = {k: TASKS[k] for k in stages}
            # check: initiate data_store
            if not data_store['state_manager'].cache:
                data_store['state_manager'].load(config, stages)

            data_store = run_pipeline(config, sub_tasks, data_store)
            state_manager = data_store['state_manager']
            state_manager.save(config, stages)
        if "b_evaluation" in mode:
            stages = ["normalization", "bert_evaluation"]
            data_store['raw_data'] = {
                'test':  safe_read_csv(config.get("testset_path"))
            }
            sub_tasks = {k: TASKS[k] for k in stages}
            # check initiate data_store
            if not data_store['state_manager'].cache:
                data_store['state_manager'].load(config, stages)
            data_store = run_pipeline(config, sub_tasks, data_store)
            state_manager = data_store['state_manager']
            state_manager.save(config, stages)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--config_file', type=str, required=True)
    return parser.parse_args()


# =========================
# Mock data source (example)
# =========================
def get_data_stream(config):
    # simulate incremental data
    yield [{"id": 1, "name": "Alice"}]
    yield [{"id": 2, "name": "Bob"}]
    yield [{"id": 3, "name": "Alice"}]


# =========================
# Main
# =========================
if __name__ == '__main__':
    args = parse_args()
    config_path=args.config_file

    # load config file
    config_file = os.path.abspath(config_path)
    yaml = YAML()
    with open(config_file, 'r') as f:
        config = yaml.load(f)
    
    # create folders
    os.makedirs('tmp/', exist_ok=True)
    os.makedirs('logs/', exist_ok=True)
    os.makedirs('data/', exist_ok=True)
    os.makedirs('storage/', exist_ok=True)

    print('#' * 46)
    print(f'###  Energy-Aware Energy Resolution System ###')
    print('#' * 46)
    print(f'# Preperation is done')
    print(f'# Configuration file path: {config_file}')
    print(f'# Executuin mode chosen: {config["mode"]}')
    print(f'# The program will start soon')

    driver(config)

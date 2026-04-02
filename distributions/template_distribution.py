from pandas import DataFrame
import pandas as pd
import socket
import json

from governance import StateManager
from pipeline import (
    sequence_generating_m1,
    index_normalization,
    train_model,
    evaluate_from_saved_model,
    compare_ground_truth,
    process_inference,
    train_embeddings,
    compute_features,
)
from pipeline.candidate_enumeration import enumerate_candidates
from pipeline.calculating_similarity import score_candidate_pairs
from pipeline.decision_making import decide_matches
from pipeline.feature_index_construction import build_index as build_cg_index
from pipeline.graph_construction import dyn_graph_generation
from pipeline.random_walk import dynrandom_walks_generation
from utils.write_log import write_log

listener_host = "localhost"
listener_port = 8080
manager_service = "manager-service"

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

    if not isinstance(processed_data, pd.DataFrame):
        raise ValueError("processed_data must be a pandas DataFrame for graph construction.")
    if not hasattr(graph, "build_relation"):
        raise ValueError("representation_graph must support build_relation for incremental updates.")

    graph.build_relation(processed_data)
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
    else:
        print('error')
    return None


def feature_index_construction(config, cg_feature, state_manager: StateManager):
    print("[feature_index_construction]")
    if not isinstance(cg_feature, list):
        raise ValueError("cg_feature must be a feature list.")

    index = state_manager.get("cg_feature_index")
    if index is None:
        raise ValueError("cg_feature_index must be initialized before feature_index_construction.")
    if not hasattr(index, "build"):
        raise ValueError("cg_feature_index must be a CGIndex instance.")

    build_cg_index(cg_feature, index)
    state_manager.update("cg_feature_index", index)
    return None


def candidate_enumeration(config, cg_feature, state_manager: StateManager):
    print("[candidate_enumeration]")
    index = state_manager.get("cg_feature_index")
    if isinstance(cg_feature, list) and cg_feature and index is not None and hasattr(index, "query"):
        return enumerate_candidates(cg_feature, index)
    else:
        print("error")
    return None

def calculating_similarity(config, candidate_pairs, state_manager: StateManager):
    print("[calculating_similarity]")
    embedding_model = state_manager.get("embedding_model")
    if embedding_model is None:
        raise ValueError("embedding_model must be initialized before calculating_similarity.")
    sim_cfg = config.get("similarity", {})
    batch_threshold = int(sim_cfg.get("batch_threshold", 2048))
    return score_candidate_pairs(embedding_model, candidate_pairs, batch_threshold=batch_threshold)


def decision_making(config, matching_pairs, state_manager: StateManager):
    print("[decision_making]")
    output_format = config.get("similarity", {}).get("output_format", "graphml")
    predicted_matching = decide_matches(matching_pairs, output_format=output_format)
    state_manager.update("predicted_matching", predicted_matching)
    return None


def bert_inference(config, processed_data, state_manager: StateManager):
    print("[bert_inference]") 
    predicted_pairs = process_inference(processed_data, state_manager)
    state_manager.update("predicted_matching", predicted_pairs)
    return None


def evaluation(config, state_manager: StateManager):
    print("[evaluation]")
    result = compare_ground_truth(config)
    state_manager.update("result", result)
    return None

def bert_evaluation(config, processed_data, state_manager: StateManager):
    print("[evaluation]")
    result = evaluate_from_saved_model(processed_data, config)
    state_manager.update("evaluation_result", result)
    return None


# =========================
# Listener
# =========================

def listener():

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((listener_host, listener_port))
    server_socket.listen()
    print(f"Listener started on {listener_host}:{listener_port}")

    while True:
        client_socket, addr = server_socket.accept()
        print(f"Accepted connection from {addr}")

        with client_socket:
            try:
                raw = b""
                while not raw.endswith(b"\n"):
                    chunk = client_socket.recv(4096)
                    if not chunk:
                        break
                    raw += chunk

                if not raw:
                    continue

                data = raw.decode("utf-8").strip()
                print(f"Received data: {data}")

                request = json.loads(data)
                if request.get("protocol_version") != "2.0":
                    raise ValueError("Unsupported protocol_version")

                task = request.get("task", "")
                print(f"Function to execute: {task}")

                output = function_dispatcher(task, **request.get("inputs", {}))

                callback = request.get("callback") or {}
                callback_host = callback.get("host")
                callback_port = callback.get("port")

                if callback_host and callback_port:
                    response = {
                        "protocol_version": "2.0",
                        "task": task,
                        "status": "ok" if output else "error",
                        "result": output if output else f"Function '{task}' not found",
                    }
                    send(json.dumps(response), callback_host, int(callback_port))
                else:
                    print("No callback provided; skipping response send.")

            except json.JSONDecodeError as e:
                print(f"Invalid JSON request: {e}")
            except Exception as e:
                print(f"Listener error: {e}")

def send(payload, service, port):
    with socket.create_connection((service, port), timeout=5) as sock:
        data = (payload + "\n").encode("utf-8")
        sock.sendall(data)

def function_dispatcher(function_name: str, **kwargs):
    function_map = {
        "normalization": normalization,
        "graph_construction": graph_construction,
        "random_walk": random_walk,
        "embedding_training": embedding_training,
        "bert_training": bert_training,
        "cg_feature_extraction": cg_feature_extraction,
        "feature_index_construction": feature_index_construction,
        "candidate_enumeration": candidate_enumeration,
        "calculating_similarity": calculating_similarity,
        "decision_making": decision_making,
        "bert_inference": bert_inference,
        "evaluation": evaluation,
        "bert_evaluation": bert_evaluation
    }
    func = function_map.get(function_name)
    if func is not None:
        return func(**kwargs)
    else:
        print(f"Function {function_name} not found in dispatcher.")
        return ""

if __name__ == '__main__':
    listener()
import argparse
import json
import os
import argparse
import shlex
import signal
import subprocess
import time
from typing import Any
from pandas import DataFrame
from ruamel.yaml import YAML
import pandas as pd
from confluent_kafka import Consumer, KafkaException, KafkaError
from confluent_kafka.admin import AdminClient, NewTopic

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
from pipeline.candidate_enumeration import enumerate_candidates, fetch_candidates
from pipeline.calculating_similarity import score_mutual_top1_candidate_pairs
from pipeline.decision_making import decide_matches
from pipeline.feature_index_construction import build_index as build_cg_index
from pipeline.graph_construction import dyn_graph_generation
from pipeline.random_walk import dynrandom_walks_generation
from utils.write_log import write_log

ACTIVE_JAVA_PROC = None
ACTIVE_CONSUMER = None

# =========================
# endpoints
# =========================

def _config_section(config: dict, primary_key: str, legacy_key: str | None = None) -> dict:
    if not isinstance(config, dict):
        return {}
    section = config.get(primary_key)
    if isinstance(section, dict):
        return section
    if legacy_key:
        legacy_section = config.get(legacy_key)
        if isinstance(legacy_section, dict):
            return legacy_section
    return {}


def _ensure_kafka_topic_ready(config: dict, timeout: float = 10.0, ready_wait_seconds: float = 10.0) -> None:
    kafka_config = config["kafka"]
    bootstrap_servers = f'{kafka_config["bootstrap_servers"]}:{kafka_config["port"]}'
    topic_name = kafka_config["topicid"]
    num_partitions = int(kafka_config.get("partitions", 3))
    replication_factor = int(kafka_config.get("replication_factor", 1))

    admin_client = AdminClient({"bootstrap.servers": bootstrap_servers})
    metadata = admin_client.list_topics(topic=topic_name, timeout=timeout)
    topic_metadata = metadata.topics.get(topic_name)

    topic_missing = (
        topic_metadata is None
        or (
            topic_metadata.error is not None
            and topic_metadata.error.code() == KafkaError.UNKNOWN_TOPIC_OR_PART
        )
    )

    if topic_missing:
        futures = admin_client.create_topics(
            [NewTopic(topic_name, num_partitions=num_partitions, replication_factor=replication_factor)]
        )
        try:
            futures[topic_name].result(timeout=timeout)
            print(
                f"[kafka] Created topic '{topic_name}' "
                f"with {num_partitions} partitions and replication factor {replication_factor}."
            )
        except Exception as exc:
            raise KafkaException(
                KafkaError(
                    KafkaError.UNKNOWN_TOPIC_OR_PART,
                    f"Failed to create Kafka topic '{topic_name}': {exc}"
                )
            ) from exc

    deadline = time.time() + ready_wait_seconds
    last_error = None
    while time.time() < deadline:
        metadata = admin_client.list_topics(topic=topic_name, timeout=timeout)
        topic_metadata = metadata.topics.get(topic_name)

        if topic_metadata is not None and topic_metadata.error is None and topic_metadata.partitions:
            return

        if topic_metadata is not None and topic_metadata.error is not None:
            last_error = topic_metadata.error
        else:
            last_error = KafkaError(
                KafkaError.UNKNOWN_TOPIC_OR_PART,
                f"Kafka topic '{topic_name}' is still unavailable on broker {bootstrap_servers}."
            )
        time.sleep(1)

    if last_error is not None:
        raise KafkaException(last_error)
    raise KafkaException(
        KafkaError(
            KafkaError.UNKNOWN_TOPIC_OR_PART,
            f"Kafka topic '{topic_name}' did not become ready within {ready_wait_seconds} seconds."
        )
    )

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
        raw_data_path = config.get("data_source_A")
        processed_data = index_normalization(config, raw_data, raw_data_path)
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
        return dynrandom_walks_generation(config, graph)
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
    if config.get("candidate_generation", {}).get("status", True):
        print("[cg_feature_extraction]")
        if isinstance(processed_data, pd.DataFrame):
            method = config.get("candidate_generation", {}).get("method", "fullindexing")
            return compute_features(processed_data, method, config)
        else:
            print('error')
    else:
        print("[cg_feature_extraction] skip")
    return None


def feature_index_construction(config, cg_feature, state_manager: StateManager):
    if config.get("candidate_generation", {}).get("status", True):
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
    else:
        print("[feature_index_construction] skip")
    return None


def candidate_enumeration(config, cg_feature, state_manager: StateManager):
    if config.get("candidate_generation", {}).get("status", True):
        print("[candidate_enumeration]")
        index = state_manager.get("cg_feature_index")
        if isinstance(cg_feature, list) and cg_feature and index is not None and hasattr(index, "query"):
            return enumerate_candidates(cg_feature, index)
        else:
            print("error")
    else:
        data_pairs_file = config.get("candidate_generation", {}).get("data_pairs_fixed", "")
        if data_pairs_file:
            print(f"[candidate_enumeration] fetch from file {data_pairs_file}")
            return fetch_candidates(data_pairs_file)
        else:
            print(f"error: [candidate_enumeration] fetch from file {data_pairs_file}, file not found")
    return None

def calculating_similarity(config, candidate_pairs, state_manager: StateManager):
    print("[calculating_similarity]")
    embedding_model = state_manager.get("embedding_model")
    if embedding_model is None:
        raise ValueError("embedding_model must be initialized before calculating_similarity.")
    sim_cfg = _config_section(config, "calculating_similarity", "similarity")
    batch_threshold = int(sim_cfg.get("batch_threshold", 2048))
    return score_mutual_top1_candidate_pairs(embedding_model, candidate_pairs, batch_threshold=batch_threshold)


def decision_making(config, mutualtop_pairs, state_manager: StateManager):
    print("[decision_making]")
    decision_cfg = _config_section(config, "decision_making", "similarity")
    output_format = decision_cfg.get("output_format", config.get("output_format", "graphml"))
    previous_pairs = state_manager.get("mutualtop_pairs")
    embedding_model = state_manager.get("embedding_model")
    final_pairs, predicted_matching = decide_matches(
        mutualtop_pairs,
        previous_pairs=previous_pairs,
        model=embedding_model,
        output_format=output_format,
    )
    state_manager.update("mutualtop_pairs", final_pairs)
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
    print("[bert evaluation]")
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
        "deps": ["cg_feature_extraction"],
        "input": ["cg_feature", "state_manager"],
        "output": ["candidate_pairs"],
        "func": candidate_enumeration
    },
    "calculating_similarity": {
        "deps": ["candidate_enumeration"],
        "input": ["candidate_pairs", "state_manager"],
        "output": ["mutualtop_pairs"],
        "func": calculating_similarity
    },
    "decision_making": {
        "deps": ["calculating_similarity"],
        "input": ["mutualtop_pairs", "state_manager"],
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
            task_start = time.perf_counter()
            try:
                output = func(config=config, **inputs)
            except Exception as e:
                task_duration = time.perf_counter() - task_start
                print(f"[TIME] {name}: {task_duration:.3f}s (failed)")
                raise RuntimeError(f"Task {name} failed after {task_duration:.3f}s: {e}")
            task_duration = time.perf_counter() - task_start
            print(f"[TIME] {name}: {task_duration:.3f}s")

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
        _ensure_kafka_topic_ready(config)

        # prepare kafka consumer
        consumer = Consumer({
            'bootstrap.servers': f'{config["kafka"]["bootstrap_servers"]}:{config["kafka"]["port"]}',
            'group.id': config['kafka']["groupid"],
            'auto.offset.reset': 'latest',   # latest / earliest
            'enable.auto.commit': False,
            'max.poll.interval.ms': 900000,
        })

        # subscribe a topic
        consumer.subscribe([config['kafka']['topicid']])

    except Exception as e:
        app_logger = write_log("logs", "main", "bug")
        app_logger.error(f"Fatal error in consumer service: {str(e)}")
        print(f"Fatal error in consumer service: {str(e)}")
        return
    return consumer


def _stop_process_group(proc, interrupt_first=False, wait_seconds=5):
    if proc is None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return

    signals = []
    if interrupt_first:
        signals.append(signal.SIGINT)
    signals.extend([signal.SIGTERM, signal.SIGKILL])

    for sig in signals:
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return

        try:
            proc.wait(timeout=wait_seconds)
            return
        except subprocess.TimeoutExpired:
            continue


def _handle_sigint(signum, frame):
    global ACTIVE_JAVA_PROC, ACTIVE_CONSUMER
    print("\n[INFO] Ctrl+C received. Terminating Python and Java processes now.")
    if ACTIVE_CONSUMER is not None:
        try:
            ACTIVE_CONSUMER.close()
        except Exception:
            pass
        ACTIVE_CONSUMER = None
    _stop_process_group(ACTIVE_JAVA_PROC, interrupt_first=True, wait_seconds=1)
    ACTIVE_JAVA_PROC = None
    raise SystemExit(130)

def driver(config):
    global ACTIVE_JAVA_PROC, ACTIVE_CONSUMER
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
                "candidate_enumeration",
                "calculating_similarity",
                "decision_making",
                ]
            sub_tasks = {k: TASKS[k] for k in stages}
            # check
            if not data_store['state_manager'].cache:
                data_store['state_manager'].load(config, stages)
            # start kafka
            poll_timeout = 5
            max_empty_polls = 5
            empty_poll_count = 0

            consumer = kafka_driver(config)

            java_path = os.path.abspath(config["simulator_path"])
            csv_path = config.get("data_source_B", "")
            csv_path = os.path.abspath(csv_path) if csv_path else ""
            kafka_bootstrap = f'{config["kafka"]["bootstrap_servers"]}:{config["kafka"]["port"]}'
            spring_args = shlex.join([
                f"--csv.file.path={csv_path}",
                f"--spring.kafka.producer.topic-id={config['kafka']['topicid']}",
                f"--spring.kafka.bootstrap-servers={kafka_bootstrap}",
            ])
            java_proc = subprocess.Popen(
                ["mvn", f"-Dspring-boot.run.arguments={spring_args}", "spring-boot:run"],
                cwd=java_path,
                start_new_session=True
            )
            ACTIVE_JAVA_PROC = java_proc

            ACTIVE_CONSUMER = consumer
            previous_sigint_handler = signal.getsignal(signal.SIGINT)

            data_buffer = []
            try:
                signal.signal(signal.SIGINT, _handle_sigint)
                while True:
                    msg = consumer.poll(poll_timeout)  # Non-blocking batch pull

                    # no new message
                    if msg is None:  
                        empty_poll_count += 1
                        print(f'# empty poll count: {empty_poll_count}')
                        if empty_poll_count >= max_empty_polls:
                            print("[INFO] No new messages for a while. Exiting consumer loop.")
                            if data_buffer:
                                data_store['raw_data'] = pd.DataFrame(data_buffer)
                                data_store = run_pipeline(config, sub_tasks, data_store)
                                state_manager = data_store['state_manager']
                                state_manager.save(config, stages)
                            break
                        continue
                    if msg.error():
                        if msg.error().code() == KafkaError._PARTITION_EOF:
                            # end of partition
                            continue
                        else:
                            raise KafkaException(msg.error())
                    
                    # new message
                    empty_poll_count = 0  # reset counter
                    # get message
                    metadata = json.loads(msg.value().decode('utf-8'))
                    data_buffer.append(metadata)
                    if len(data_buffer) >= config["kafka"]["window_count"]:
                        data_store['raw_data'] = pd.DataFrame(data_buffer)
                        data_store = run_pipeline(config, sub_tasks, data_store)
                        data_buffer = []
                state_manager = data_store['state_manager']
                state_manager.save(config, stages)
            finally:
                signal.signal(signal.SIGINT, previous_sigint_handler)
                if ACTIVE_CONSUMER is not None:
                    try:
                        ACTIVE_CONSUMER.close()
                    except Exception:
                        pass
                    ACTIVE_CONSUMER = None
                _stop_process_group(java_proc, interrupt_first=True)
                ACTIVE_JAVA_PROC = None

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
    total_start = time.perf_counter()
    args = parse_args()
    config_path=args.config_file

    # load config file
    config_file = os.path.abspath(config_path)
    yaml = YAML()
    with open(config_file, 'r') as f:
        config = yaml.load(f)
    
    # check data path 
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    print(f'file path: {base_dir}')
    workpath = os.getcwd()
    print(f'work dir: {workpath}')

    # create folders
    os.makedirs('tmp/', exist_ok=True)
    os.makedirs('logs/', exist_ok=True)
    os.makedirs('data/', exist_ok=True)

    print('#' * 46)
    print(f'###  Energy-Aware Energy Resolution System ###')
    print('#' * 46)
    print(f'# Preperation is done')
    print(f'# Configuration file path: {config_file}')
    print(f'# Executuin mode chosen: {config["mode"]}')
    print(f'# The program will start soon')

    try:
        driver(config)
    finally:
        total_duration = time.perf_counter() - total_start
        print(f"[TIME] Total runtime: {total_duration:.3f}s")

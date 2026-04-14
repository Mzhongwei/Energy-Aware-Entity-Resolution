import argparse
import json
import os
import argparse
import shlex
import signal
import subprocess
import sys
import time
from typing import Any
from pandas import DataFrame
from ruamel.yaml import YAML
import pandas as pd
import socket
from confluent_kafka import Consumer, KafkaException, KafkaError
from confluent_kafka.admin import AdminClient, NewTopic

from governance import StateManager
from utils.write_log import write_log

# =========================
# Payload Serialization
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
        return obj

# =========================
# TASKS
# =========================

TASKS = {
    "normalization": {
        "deps": [],
        "input": ["raw_data"],
        "output": ["processed_data"],
        "service": "service-normalization",
        "listen-port": 5000
    },
    "graph_construction": {
        "deps": ["normalization"],
        "input": ["processed_data", "state_manager"],
        "output": [],
        "service": "service-graph-construction",
        "listen-port": 5001
    },
    "random_walk": {
        "deps": ["graph_construction"],
        "input": ["state_manager"],
        "output": ["sequences"],
        "service": "service-random-walk",
        "listen-port": 5002
    },
    "embedding_training": {
        "deps": ["random_walk"],
        "input": ["sequences", "state_manager"],
        "output": [],
        "service": "service-embedding-training",
        "listen-port": 5003
    },
    "bert_training": {
        "deps": ["normalization"],
        "input": ["processed_data", "state_manager"],
        "output": ["completed"],
        "service": "service-bert",
        "listen-port": 5004
    },
    "cg_feature_extraction": {
        "deps": ["normalization"],
        "input": ["processed_data"],
        "output": ["cg_feature"],
        "service": "service-cg-feature-extraction",
        "listen-port": 5005
    },
    "feature_index_construction": {
        "deps": ["cg_feature_extraction"],
        "input": ["cg_feature", "state_manager"],
        "output": [],
        "service": "service-feature-index-construction",
        "listen-port": 5006
    },
    "candidate_enumeration": {
        "deps": ["cg_feature_extraction"],
        "input": ["cg_feature", "state_manager"],
        "output": ["candidate_pairs"],
        "service": "service-candidate-enumeration",
        "listen-port": 5007
    },
    "calculating_similarity": {
        "deps": ["candidate_enumeration"],
        "input": ["candidate_pairs", "state_manager"],
        "output": ["matching_pairs"],
        "service": "service-calculating-similarity",
        "listen-port": 5008
    },
    "decision_making": {
        "deps": ["calculating_similarity"],
        "input": ["matching_pairs", "state_manager"],
        "output": [],
        "service": "service-decision-making",
        "listen-port": 5009
    },
    "bert_inference": {
        "deps": ["normalization"],
        "input": ["processed_data", "state_manager"],
        "output": ["completed"],
        "service": "service-bert",
        "listen-port": 5010
    },
    "evaluation": {
        "deps": [],
        "input": ["state_manager"],
        "output": [],
        "service": "service-evaluation",
        "listen-port": 5011
    },
    "bert_evaluation": {
        "deps": ["normalization"],
        "input": ["processed_data", "state_manager"],
        "output": ["completed"],
        "service": "service-bert",
        "listen-port": 5012
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


def _estimate_cache_item_bytes(value):
    if value is None:
        return 0

    if isinstance(value, pd.DataFrame):
        return int(value.memory_usage(index=True, deep=True).sum())

    if isinstance(value, dict):
        size = sys.getsizeof(value)
        for key, item in value.items():
            size += sys.getsizeof(key)
            size += _estimate_cache_item_bytes(item)
        return size

    if isinstance(value, (list, tuple, set)):
        size = sys.getsizeof(value)
        for item in value:
            size += _estimate_cache_item_bytes(item)
        return size

    if isinstance(value, StateManager):
        return sys.getsizeof(value) + _estimate_cache_item_bytes(value.cache)

    return sys.getsizeof(value)

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

            # print(f"[RUN] {name}")

            # func = task["func"]

            # 3. execute the missions
            task_start = time.perf_counter()
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.connect((service_name, service_port))
                    s.sendall((json.dumps(request) + "\n").encode("utf-8"))
                    print(f"Sent request for task '{name}' to {service_name}:{service_port}")
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
            
            elif output_keys == ["completed"]:
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

            cache_summary = ", ".join(
                f"{key}={_estimate_cache_item_bytes(value)}B"
                for key, value in data_store.items()
            )
            print(f"[CACHE] after {name}: {cache_summary}")

            finished.add(name)
            progress = True

        if not progress:
            unfinished = [t for t in tasks if t not in finished]
            raise RuntimeError(f"DAG stuck! Unfinished tasks: {unfinished}. Check dependencies or inputs.")

    return data_store

# =========================
# Driver
# =========================

def safe_read_csv(path):
    if not path:
        return pd.DataFrame()

    if not os.path.exists(path):
        print(f"CSV not found: {path}")
        return pd.DataFrame()
    print(f"Reading CSV from: {path}")
    print(pd.read_csv(path).head())
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
            'max.poll.interval.ms': 1800000,
        })

        # subscribe a topic
        consumer.subscribe([config['kafka']['topicid']])

    except Exception as e:
        app_logger = write_log("logs", "main", "bug")
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

def daemon():
    while True:
        time.sleep(60)

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

    try:
        driver(config)
    finally:
        total_duration = time.perf_counter() - total_start
        print(f"[TIME] Total runtime: {total_duration:.3f}s")

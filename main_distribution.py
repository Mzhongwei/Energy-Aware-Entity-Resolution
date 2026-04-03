import argparse
import json
import os
import argparse
from ruamel.yaml import YAML
import pandas as pd
import socket
from confluent_kafka import Consumer, KafkaException, KafkaError

from governance import StateManager
from utils.write_log import write_log

# =========================
# Payload Serialization
# =========================

def serialize_for_json(obj):
    """Convert non-JSON-serializable objects to JSON-safe format."""
    if isinstance(obj, pd.DataFrame):
        return {"__dataframe__": True, "data": obj.to_dict(orient="records")}
    elif isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [serialize_for_json(item) for item in obj]
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

            # # 3. execute the missions
            # try:
            #     output = func(config=config, **inputs)
            # except Exception as e:
            #     raise RuntimeError(f"Task {name} failed: {e}")
            
            # Socket connection to the service (new protocol: single JSON envelope)
            service_name = task.get("service")
            if not service_name:
                raise ValueError(f"Task {name} has no service configured")

            service_port = task.get("service-port", 80) # default port 80 for services

            callback_port = task.get("listen-port") if task.get("output") else None

            # Exclude StateManager from JSON payload; it stays in manager
            serializable_inputs = {k: v for k, v in inputs.items() if k != "state_manager"}
            serializable_inputs = serialize_for_json(serializable_inputs)

            request = {
                "protocol_version": "2.0",
                "task": name,
                "config": config,
                "inputs": serializable_inputs,
                "callback": {
                    "host": "service-manager",
                    "port": callback_port,
                } if callback_port else None,
            }

            print(f"Connecting to service: {service_name}:{service_port}")
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.connect((service_name, service_port))
                    s.sendall((json.dumps(request) + "\n").encode("utf-8"))
                    print(f"Sent request for task '{name}' to {service_name}:{service_port}")
            except Exception as e:
                raise RuntimeError(f"Failed to connect to {service_name}:{service_port}: {e}")

            # Listen for the service response if output is expected (JSON envelope protocol v2)
            if task.get("output"):
                listen_port = task.get("listen-port")
                if listen_port:
                    print(f"Listening for response on port: {listen_port}")
                    try:
                        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                            s.bind(("0.0.0.0", listen_port))
                            s.listen(1)
                            s.settimeout(300)

                            conn, addr = s.accept()
                            with conn:
                                print(f"Connected by {addr}")

                                chunks = []
                                while True:
                                    chunk = conn.recv(4096)
                                    if not chunk:
                                        break
                                    chunks.append(chunk)
                                    if b"\n" in chunk:
                                        break

                                raw_response = b"".join(chunks).decode("utf-8").strip()
                                print(f"Received response: {raw_response}")

                                try:
                                    response_obj = json.loads(raw_response)
                                except json.JSONDecodeError as e:
                                    raise RuntimeError(f"Invalid JSON response for task '{name}': {e}")

                                if response_obj.get("protocol_version") != "2.0":
                                    raise RuntimeError(
                                        f"Unsupported protocol version for task '{name}': "
                                        f"{response_obj.get('protocol_version')}"
                                    )

                                if response_obj.get("task") != name:
                                    raise RuntimeError(
                                        f"Task mismatch in response. Expected '{name}', "
                                        f"got '{response_obj.get('task')}'"
                                    )

                                if response_obj.get("status") != "ok":
                                    raise RuntimeError(
                                        f"Service returned error for task '{name}': "
                                        f"{response_obj.get('result')}"
                                    )

                                output = response_obj.get("result")
                    except socket.timeout:
                        raise RuntimeError(f"Timeout waiting for response from task '{name}'")
                else:
                    print(f"No listen port specified for service {service_name}, skipping response handling.")


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
    args = parse_args()
    config_path=args.config_file

    # load config file
    config_file = os.path.abspath(config_path)
    yaml = YAML()
    with open(config_file, 'r') as f:
        config = yaml.load(f)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    print(f'file path: {base_dir}')
    workpath = os.getcwd()
    print(f'work dir: {workpath}')
    
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

    import time
    time.sleep(10)
    driver(config)
    # Enter daemon mode to keep the pod running
    daemon()
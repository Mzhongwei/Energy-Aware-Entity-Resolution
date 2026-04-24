import argparse
import json
import os
import signal
import subprocess
import time

import urllib.error
import urllib.parse
import urllib.request
from ruamel.yaml import YAML
import pandas as pd
from confluent_kafka import Consumer, KafkaException, KafkaError, Producer as KafkaProducer
from confluent_kafka.admin import AdminClient, NewTopic

from utils.write_log import write_log

ACTIVE_JAVA_PROC = None
ACTIVE_CONSUMER = None
CONFIG_PATH = os.environ.get("EAER_CONFIG_PATH", "/app/config/examples/config-embedding.yaml")

# =========================
# endpoints
# =========================

def _as_bool(value, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "y"}
    return default

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

def _ensure_pipeline_ready(config: dict):
    """Block until expected consumer pods are in phase Running."""
    chain_cfg = config.get("kafka_chain", {}) if isinstance(config, dict) else {}
    readiness_cfg = chain_cfg.get("readiness", {}) if isinstance(chain_cfg, dict) else {}

    timeout_seconds = float(readiness_cfg.get("timeout_seconds", 60))
    poll_interval_seconds = float(readiness_cfg.get("poll_interval_seconds", 1))
    require_consumer_pods = _as_bool(readiness_cfg.get("require_consumer_pods", True), default=True)
    allow_without_pod_rbac = _as_bool(readiness_cfg.get("allow_without_pod_rbac", True), default=True)

    stages = chain_cfg.get("stages", {}) if isinstance(chain_cfg, dict) else {}

    def _in_cluster_context() -> dict:
        namespace = str(readiness_cfg.get("kafka_namespace") or "").strip()
        if not namespace:
            namespace = os.environ.get("POD_NAMESPACE", "")
        if not namespace:
            namespace_path = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"
            if os.path.exists(namespace_path):
                with open(namespace_path, "r", encoding="utf-8") as ns_file:
                    namespace = ns_file.read().strip()

        token_path = "/var/run/secrets/kubernetes.io/serviceaccount/token"
        ca_path = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
        api_host = os.environ.get("KUBERNETES_SERVICE_HOST")
        api_port = os.environ.get("KUBERNETES_SERVICE_PORT", "443")

        return {
            "api_host": api_host,
            "api_port": api_port,
            "namespace": namespace,
            "token_path": token_path,
            "ca_path": ca_path,
            "in_cluster": bool(api_host and api_port and os.path.exists(token_path) and os.path.exists(ca_path)),
        }

    cluster_ctx = _in_cluster_context()

    def _k8s_request_json(path: str) -> dict:
        with open(cluster_ctx["token_path"], "r", encoding="utf-8") as token_file:
            token = token_file.read().strip()

        url = f"https://{cluster_ctx['api_host']}:{cluster_ctx['api_port']}{path}"
        request = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/json",
            },
        )

        context = None
        if os.path.exists(cluster_ctx["ca_path"]):
            import ssl

            context = ssl.create_default_context(cafile=cluster_ctx["ca_path"])

        with urllib.request.urlopen(request, timeout=5, context=context) as response:
            return json.loads(response.read().decode("utf-8"))

    def _normalize_task_token(value: str) -> str:
        return (value or "").strip().lower().replace("_", "-")

    expected_consumer_tasks = []
    configured_tasks = readiness_cfg.get("consumer_task_names", [])
    if isinstance(configured_tasks, list) and configured_tasks:
        expected_consumer_tasks = [_normalize_task_token(str(item)) for item in configured_tasks if str(item).strip()]
    else:
        # Source consumer should only wait for the first downstream stage by default.
        normalization_stage = stages.get("normalization", {}) if isinstance(stages, dict) else {}
        if isinstance(normalization_stage, dict) and normalization_stage.get("input_topic"):
            task_name = normalization_stage.get("task_name") or "normalization"
            expected_consumer_tasks = [_normalize_task_token(str(task_name))]
        elif isinstance(stages, dict):
            for stage_name, stage_cfg in stages.items():
                if not isinstance(stage_cfg, dict):
                    continue
                if stage_cfg.get("input_topic"):
                    task_name = stage_cfg.get("task_name") or stage_name
                    expected_consumer_tasks.append(_normalize_task_token(str(task_name)))

    expected_consumer_tasks = sorted({task for task in expected_consumer_tasks if task})

    def _consumer_pods_running() -> tuple[bool, str]:
        if not require_consumer_pods or not expected_consumer_tasks:
            return True, ""
        if not cluster_ctx["in_cluster"]:
            if allow_without_pod_rbac:
                return True, "skipped consumer-pod readiness (not in kubernetes cluster context)"
            return False, "not in kubernetes cluster context"

        namespace = cluster_ctx["namespace"]
        if not namespace:
            return False, "pod namespace not available"

        workflow_name = os.environ.get("ARGO_WORKFLOW_NAME") or os.environ.get("WORKFLOW_NAME")
        if not workflow_name:
            pod_name = os.environ.get("HOSTNAME", "")
            if pod_name:
                try:
                    own_pod = _k8s_request_json(f"/api/v1/namespaces/{namespace}/pods/{pod_name}")
                    labels = own_pod.get("metadata", {}).get("labels", {}) if isinstance(own_pod, dict) else {}
                    workflow_name = labels.get("workflows.argoproj.io/workflow", "")
                except (urllib.error.URLError, OSError, json.JSONDecodeError, TimeoutError):
                    workflow_name = ""

        if not workflow_name:
            if allow_without_pod_rbac:
                return True, "skipped consumer-pod readiness (workflow name unavailable)"
            return False, "workflow name unavailable for consumer-pod check"

        selector = urllib.parse.quote(f"workflows.argoproj.io/workflow={workflow_name}", safe="=,")
        pods_path = f"/api/v1/namespaces/{namespace}/pods?labelSelector={selector}"

        try:
            pods_payload = _k8s_request_json(pods_path)
        except urllib.error.HTTPError as ex:
            if ex.code == 403 and allow_without_pod_rbac:
                return True, "skipped consumer-pod readiness (RBAC denied pod listing)"
            return False, f"failed to query workflow pods (HTTP {ex.code})"
        except (urllib.error.URLError, OSError, json.JSONDecodeError, TimeoutError) as ex:
            return False, f"failed to query workflow pods ({ex})"

        items = pods_payload.get("items", []) if isinstance(pods_payload, dict) else []
        running_pod_names = []
        for pod in items:
            if not isinstance(pod, dict):
                continue
            phase = pod.get("status", {}).get("phase", "")
            name = pod.get("metadata", {}).get("name", "")
            if phase == "Running" and name:
                running_pod_names.append(name.lower())

        if not running_pod_names:
            return False, "no running pods found in workflow"

        missing = []
        for task_token in expected_consumer_tasks:
            if not any(task_token in pod_name for pod_name in running_pod_names):
                missing.append(task_token)

        if missing:
            return False, f"consumer pods not running yet: {', '.join(missing)}"
        return True, ""

    deadline = time.time() + max(timeout_seconds, 1.0)
    last_pods_error = ""

    while time.time() < deadline:
        pods_ready, pods_error = _consumer_pods_running()
        if not pods_ready:
            last_pods_error = pods_error

        if pods_ready:
            if pods_error:
                print(f"[WARN] {pods_error}")
            return
        print(f"[INFO] Waiting for Kafka consumer pods to be ready... {last_pods_error}")
        time.sleep(max(poll_interval_seconds, 0.1))

    if require_consumer_pods and last_pods_error:
        raise TimeoutError(f"Kafka pipeline readiness check timed out. {last_pods_error}")
    raise TimeoutError("Kafka pipeline readiness check timed out.")


# =========================
# Driver
# =========================

def safe_read_csv(path):
    if not path:
        return pd.DataFrame()

    if not os.path.exists(path):
        return pd.DataFrame()

    return pd.read_csv(path)


def _kafka_chain_enabled(config: dict) -> bool:
    chain_cfg = config.get("kafka_chain", {}) if isinstance(config, dict) else {}
    return _as_bool(chain_cfg.get("enabled", False), default=False)


def _normalization_input_topic(config: dict) -> str:
    kafka_cfg = config.get("kafka", {}) if isinstance(config, dict) else {}
    chain_cfg = config.get("kafka_chain", {}) if isinstance(config, dict) else {}
    stages = chain_cfg.get("stages", {}) if isinstance(chain_cfg, dict) else {}
    normalization_stage = stages.get("normalization", {}) if isinstance(stages, dict) else {}
    if isinstance(normalization_stage, dict) and normalization_stage.get("input_topic"):
        return str(normalization_stage["input_topic"])
    return str(kafka_cfg.get("topicid", ""))


def _serialize_for_json(obj):
    if isinstance(obj, pd.DataFrame):
        return {"__dataframe__": True, "data": obj.to_dict(orient="records")}
    if isinstance(obj, dict):
        return {k: _serialize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize_for_json(item) for item in obj]
    return obj


def _encode_message(stage_name: str, payload, message_type: str = "data", upstream_stage: str = "") -> bytes:
    envelope = {
        "message_type": message_type,
        "stage": stage_name,
        "upstream_stage": upstream_stage,
        "payload": _serialize_for_json(payload),
    }
    return json.dumps(envelope).encode("utf-8")


def _publish_buffer_to_normalization(config: dict, producer, data_buffer: list[dict]):
    if producer is None or not data_buffer:
        return

    output_topic = _normalization_input_topic(config)
    if not output_topic:
        return

    payload = {"data": pd.DataFrame(data_buffer)}
    producer.produce(
        output_topic,
        key="source_consumer",
        value=_encode_message("source_consumer", payload, message_type="data", upstream_stage=""),
    )
    producer.flush()
    print(f"[INFO] Published {len(data_buffer)} records to normalization topic '{output_topic}'.")


def _publish_end_of_stream_to_normalization(config: dict, producer):
    if producer is None:
        return

    output_topic = _normalization_input_topic(config)
    if not output_topic:
        return

    eos_type = str((config.get("kafka_chain", {}) if isinstance(config, dict) else {}).get("end_of_stream", "end_of_stream"))
    producer.produce(
        output_topic,
        key="source_consumer",
        value=_encode_message(
            "source_consumer",
            {"reason": "source_consumer_idle_timeout"},
            message_type=eos_type,
            upstream_stage="",
        ),
    )
    producer.flush()
    print(f"[INFO] Published EOS to normalization topic '{output_topic}'.")

def kafka_driver(config):
    try:
        _ensure_kafka_topic_ready(config)
        _ensure_pipeline_ready(config)


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


def start_consumer(config):
    global ACTIVE_JAVA_PROC, ACTIVE_CONSUMER
    data_store = {
        "raw_data": None,
        "is_training": False,
    }
    
    # start kafka
    chain_cfg = config.get("kafka_chain", {}) if isinstance(config, dict) else {}
    poll_timeout = float(chain_cfg.get("source_poll_timeout_seconds", chain_cfg.get("poll_timeout", 5)))
    max_empty_polls = int(chain_cfg.get("source_max_empty_polls", chain_cfg.get("max_empty_polls", 10)))
    startup_grace_seconds = float(chain_cfg.get("source_startup_grace_seconds", 120))
    empty_poll_count = 0

    consumer = kafka_driver(config)
    if consumer is None:
        raise RuntimeError("Kafka consumer initialization failed.")
    producer = None
    if _kafka_chain_enabled(config):
        bootstrap = f'{config["kafka"]["bootstrap_servers"]}:{config["kafka"]["port"]}'
        producer = KafkaProducer({"bootstrap.servers": bootstrap})

    ACTIVE_CONSUMER = consumer
    previous_sigint_handler = signal.getsignal(signal.SIGINT)

    data_buffer = []
    has_received_messages = False
    started_at = time.time()
    try:
        signal.signal(signal.SIGINT, _handle_sigint)
        while True:
            msg = consumer.poll(poll_timeout)  # Non-blocking batch pull

            # no new message
            if msg is None:  
                empty_poll_count += 1
                print(f'# empty poll count: {empty_poll_count}')
                if empty_poll_count >= max_empty_polls:
                    elapsed = time.time() - started_at
                    if not has_received_messages and elapsed < max(startup_grace_seconds, 0.0):
                        print(
                            "[INFO] Still waiting for first source message "
                            f"({elapsed:.1f}s < {startup_grace_seconds:.1f}s). Continuing..."
                        )
                        empty_poll_count = 0
                        continue
                    print("[INFO] No new messages for a while. Exiting consumer loop.")
                    if data_buffer:
                        data_store['raw_data'] = pd.DataFrame(data_buffer)
                        print("[DEBUG]: Final data before exit:", data_store['raw_data'])
                        _publish_buffer_to_normalization(config, producer, data_buffer)
                        data_buffer = []
                    if has_received_messages:
                        _publish_end_of_stream_to_normalization(config, producer)
                    else:
                        print("[INFO] No source messages received; skipping EOS publish.")
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
            has_received_messages = True
            # get message
            metadata = json.loads(msg.value().decode('utf-8'))
            data_buffer.append(metadata)
            if len(data_buffer) >= config["kafka"]["window_count"]:
                data_store['raw_data'] = pd.DataFrame(data_buffer)
                print("[DEBUG]: Data:", data_store['raw_data'])
                _publish_buffer_to_normalization(config, producer, data_buffer)
                data_buffer = []
    finally:
        signal.signal(signal.SIGINT, previous_sigint_handler)
        if ACTIVE_CONSUMER is not None:
            try:
                ACTIVE_CONSUMER.close()
            except Exception:
                pass
            ACTIVE_CONSUMER = None
        if producer is not None:
            try:
                producer.flush()
            except Exception:
                pass
        ACTIVE_JAVA_PROC = None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--config_file', type=str, required=True)
    return parser.parse_args()


# =========================
# Main
# =========================
if __name__ == '__main__':
    total_start = time.perf_counter()

    # load config file
    config_file = os.path.abspath(CONFIG_PATH)
    yaml = YAML()
    with open(config_file, 'r') as f:
        config = yaml.load(f)
    try:
        start_consumer(config)
    except Exception as e:
        app_logger = write_log("logs", "main", "bug")
        app_logger.error(f"Fatal error in consumer service: {str(e)}")
        print(f"Fatal error in consumer service: {str(e)}")
    total_end = time.perf_counter()
    print(f"Total execution time: {total_end - total_start:.2f} seconds")
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable

import pandas as pd
from confluent_kafka import Consumer, KafkaError, Producer
from confluent_kafka.admin import AdminClient, NewTopic
from ruamel.yaml import YAML

CONFIG_PATH = os.environ.get("EAER_CONFIG_PATH", "/app/config/examples/config-embedding.yaml")


def load_config(config_path: str = CONFIG_PATH):
    if not os.path.exists(config_path):
        return {}
    yaml = YAML(typ="safe")
    with open(config_path, "r", encoding="utf-8") as file_handle:
        loaded = yaml.load(file_handle) or {}
    return loaded if isinstance(loaded, dict) else {}


def serialize_for_json(obj):
    if isinstance(obj, pd.DataFrame):
        return {"__dataframe__": True, "data": obj.to_dict(orient="records")}
    if isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [serialize_for_json(item) for item in obj]
    return obj


def deserialize_from_json(obj):
    if isinstance(obj, dict):
        if obj.get("__dataframe__"):
            return pd.DataFrame(obj.get("data", []))
        return {k: deserialize_from_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [deserialize_from_json(item) for item in obj]
    return obj


def _as_bool(value, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "y"}
    return default


def _kafka_bootstrap(config: dict) -> str:
    kafka_cfg = config.get("kafka", {}) if isinstance(config, dict) else {}
    host = kafka_cfg.get("bootstrap_servers", "localhost")
    port = kafka_cfg.get("port", 9092)
    return f"{host}:{port}"


def _stage_config(config: dict, stage_name: str) -> dict:
    chain_cfg = config.get("kafka_chain", {}) if isinstance(config, dict) else {}
    stages = chain_cfg.get("stages", {}) if isinstance(chain_cfg, dict) else {}
    stage_cfg = stages.get(stage_name, {}) if isinstance(stages, dict) else {}
    return stage_cfg if isinstance(stage_cfg, dict) else {}


def ensure_topic(config: dict, topic_name: str, timeout: float = 10.0):
    if not topic_name:
        return

    kafka_cfg = config.get("kafka", {}) if isinstance(config, dict) else {}
    admin_client = AdminClient({"bootstrap.servers": _kafka_bootstrap(config)})
    partitions = int(kafka_cfg.get("partitions", 3))
    replication_factor = int(kafka_cfg.get("replication_factor", 1))

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
            [NewTopic(topic_name, num_partitions=partitions, replication_factor=replication_factor)]
        )
        try:
            futures[topic_name].result(timeout=timeout)
        except Exception:
            pass

    deadline = time.time() + timeout
    while time.time() < deadline:
        metadata = admin_client.list_topics(topic=topic_name, timeout=timeout)
        topic_metadata = metadata.topics.get(topic_name)
        if topic_metadata is not None and topic_metadata.error is None and topic_metadata.partitions:
            return
        time.sleep(0.5)


def resolve_stage_topics(config: dict, stage_name: str) -> dict:
    kafka_cfg = config.get("kafka", {}) if isinstance(config, dict) else {}
    chain_cfg = config.get("kafka_chain", {}) if isinstance(config, dict) else {}
    stage_cfg = _stage_config(config, stage_name)
    input_topic = stage_cfg.get("input_topic") or kafka_cfg.get("topicid", "")
    output_topic = stage_cfg.get("output_topic", "")
    group_id = stage_cfg.get("groupid") or f'{kafka_cfg.get("groupid", "er_group")}-{stage_name}'
    poll_timeout = int(
        stage_cfg.get(
            "poll_timeout",
            chain_cfg.get("poll_timeout", kafka_cfg.get("poll_timeout", 5)),
        )
    )
    max_empty_polls = int(
        stage_cfg.get(
            "max_empty_polls",
            chain_cfg.get("max_empty_polls", kafka_cfg.get("max_empty_polls", 5)),
        )
    )
    eos_message_type = str((config.get("kafka_chain", {}) if isinstance(config, dict) else {}).get("end_of_stream", "end_of_stream"))
    return {
        "input_topic": input_topic,
        "output_topic": output_topic,
        "group_id": group_id,
        "poll_timeout": poll_timeout,
        "max_empty_polls": max_empty_polls,
        "eos_message_type": eos_message_type,
    }


def load_message_payload(raw_value: bytes):
    try:
        loaded = json.loads(raw_value.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return raw_value.decode("utf-8", errors="ignore")
    return deserialize_from_json(loaded)


def encode_message(stage_name: str, payload: Any, message_type: str = "data", upstream_stage: str = "") -> bytes:
    envelope = {
        "message_type": message_type,
        "stage": stage_name,
        "upstream_stage": upstream_stage,
        "payload": serialize_for_json(payload),
    }
    return json.dumps(envelope).encode("utf-8")


def is_end_of_stream(message: dict, eos_message_type: str) -> bool:
    return isinstance(message, dict) and message.get("message_type") == eos_message_type


def kafka_chain_enabled(config: dict, stage_name: str) -> bool:
    chain_cfg = config.get("kafka_chain", {}) if isinstance(config, dict) else {}
    if not _as_bool(chain_cfg.get("enabled", False), default=False):
        return False
    return bool(_stage_config(config, stage_name))


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

    def _pod_identifiers(pod: dict) -> list[str]:
        if not isinstance(pod, dict):
            return []
        metadata = pod.get("metadata", {}) if isinstance(pod.get("metadata"), dict) else {}
        labels = metadata.get("labels", {}) if isinstance(metadata.get("labels"), dict) else {}

        raw_values = [
            metadata.get("name", ""),
            labels.get("workflows.argoproj.io/display-name", ""),
            labels.get("workflows.argoproj.io/template", ""),
            labels.get("workflows.argoproj.io/node-name", ""),
        ]

        identifiers: list[str] = []
        for raw in raw_values:
            token = _normalize_task_token(str(raw))
            if token:
                identifiers.append(token)
        return identifiers

    def _token_matches_identifiers(task_token: str, identifiers: list[str]) -> bool:
        if not task_token:
            return False
        for identifier in identifiers:
            if not identifier:
                continue
            if task_token == identifier:
                return True
            if task_token in identifier or identifier in task_token:
                return True
        return False

    expected_consumer_tasks = []
    configured_tasks = readiness_cfg.get("consumer_task_names", [])
    if isinstance(configured_tasks, list) and configured_tasks:
        expected_consumer_tasks = [_normalize_task_token(str(item)) for item in configured_tasks if str(item).strip()]
    else:
        if isinstance(stages, dict):
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
        running_identifiers: list[list[str]] = []
        all_identifiers: list[list[str]] = []

        for pod in items:
            if not isinstance(pod, dict):
                continue
            phase = pod.get("status", {}).get("phase", "")
            name = pod.get("metadata", {}).get("name", "")
            identifiers = _pod_identifiers(pod)
            if identifiers:
                all_identifiers.append(identifiers)
            if phase == "Running" and name:
                running_pod_names.append(name.lower())
                if identifiers:
                    running_identifiers.append(identifiers)

        if not running_pod_names:
            return False, "no running pods found in workflow"

        # Only enforce tokens that correspond to tasks/pods that exist in this
        # workflow run. This avoids waiting forever on stages not scheduled in
        # the current DAG path (e.g. training vs inference branches).
        applicable_expected = []
        for task_token in expected_consumer_tasks:
            if any(_token_matches_identifiers(task_token, identifiers) for identifiers in all_identifiers):
                applicable_expected.append(task_token)

        missing = []
        for task_token in applicable_expected:
            if not any(_token_matches_identifiers(task_token, identifiers) for identifiers in running_identifiers):
                missing.append(task_token)

        if missing:
            running_preview = ", ".join(sorted(running_pod_names)[:12])
            return (
                False,
                "consumer pods not running yet: "
                f"missing={', '.join(missing)}; "
                f"expected_tokens={', '.join(expected_consumer_tasks)}; "
                f"applicable_tokens={', '.join(applicable_expected)}; "
                f"workflow={workflow_name}; "
                f"running_pods(sample)={running_preview}",
            )
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


def run_kafka_stage(config: dict, stage_name: str, process_payload: Callable[[Any, dict, dict], Any]):
    print(f"[INFO] Starting Kafka stage '{stage_name}'...")
    chain_cfg = config.get("kafka_chain", {}) if isinstance(config, dict) else {}
    readiness_cfg = chain_cfg.get("readiness", {}) if isinstance(chain_cfg, dict) else {}
    # Source producer/consumer already handle startup readiness. For stage workers,
    # pod readiness checks can create circular waits, so keep them opt-in.
    if _as_bool(readiness_cfg.get("enforce_in_stage_workers", False), default=False):
        _ensure_pipeline_ready(config)
    topics = resolve_stage_topics(config, stage_name)
    if not topics["input_topic"]:
        raise ValueError(f"Missing input_topic for Kafka stage '{stage_name}'.")

    ensure_topic(config, topics["input_topic"])
    if topics["output_topic"]:
        ensure_topic(config, topics["output_topic"])

    kafka_servers = _kafka_bootstrap(config)
    consumer = Consumer(
        {
            "bootstrap.servers": kafka_servers,
            "group.id": topics["group_id"],
            "auto.offset.reset": "latest",
            "enable.auto.commit": False,
            "max.poll.interval.ms": 1800000,
        }
    )
    producer = Producer({"bootstrap.servers": kafka_servers})

    consumer.subscribe([topics["input_topic"]])
    empty_poll_count = 0

    try:
        while True:
            msg = consumer.poll(topics["poll_timeout"])

            if msg is None:
                empty_poll_count += 1
                if empty_poll_count >= topics["max_empty_polls"]:
                    return {"status": "idle_timeout", "stage": stage_name}
                continue

            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                raise RuntimeError(str(msg.error()))

            empty_poll_count = 0
            message = load_message_payload(msg.value())
            if is_end_of_stream(message, topics["eos_message_type"]):
                if topics["output_topic"]:
                    producer.produce(
                        topics["output_topic"],
                        key=stage_name,
                        value=encode_message(
                            stage_name,
                            None,
                            message_type=topics["eos_message_type"],
                            upstream_stage=message.get("stage", "") if isinstance(message, dict) else "",
                        ),
                    )
                    producer.flush()
                consumer.commit(message=msg, asynchronous=False)
                return {"status": "end_of_stream", "stage": stage_name}

            payload = message.get("payload", message) if isinstance(message, dict) else message
            result = process_payload(payload, message if isinstance(message, dict) else {}, config)

            if topics["output_topic"] and result is not None:
                producer.produce(
                    topics["output_topic"],
                    key=stage_name,
                    value=encode_message(stage_name, result, upstream_stage=stage_name),
                )
                producer.flush()

            consumer.commit(message=msg, asynchronous=False)

    finally:
        try:
            consumer.close()
        except Exception:
            pass
        try:
            producer.flush()
        except Exception:
            pass
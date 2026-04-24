import argparse
import glob
import json
import os
import shutil
import signal
import subprocess
import sys
import time

import urllib.error
import urllib.parse
import urllib.request
from confluent_kafka import Producer as KafkaProducer
from ruamel.yaml import YAML
import pandas as pd

from utils.write_log import write_log

ACTIVE_JAVA_PROC = None
ACTIVE_CONSUMER = None
CONFIG_PATH = os.environ.get("EAER_CONFIG_PATH", "/app/config/examples/config-embedding.yaml")

# =========================
# Driver
# =========================

def safe_read_csv(path):
    if not path:
        return pd.DataFrame()

    if not os.path.exists(path):
        return pd.DataFrame()

    return pd.read_csv(path)


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


def _resolve_simulator_jar(java_path: str) -> str:
    jar_candidates = sorted(
        candidate
        for candidate in glob.glob(os.path.join(java_path, "target", "*.jar"))
        if not candidate.endswith(".original")
    )
    if not jar_candidates:
        raise FileNotFoundError(
            f"No built simulator jar found under {os.path.join(java_path, 'target')}. "
            "Build the image so Maven packages the application first."
        )
    return jar_candidates[0]


def _as_bool(value, default=False):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "y"}
    return default


def _kafka_chain_enabled(config: dict) -> bool:
    chain_cfg = config.get("kafka_chain", {}) if isinstance(config, dict) else {}
    return _as_bool(chain_cfg.get("enabled", False), default=False)


def _source_topic_for_chain(config: dict) -> str:
    kafka_cfg = config.get("kafka", {}) if isinstance(config, dict) else {}
    chain_cfg = config.get("kafka_chain", {}) if isinstance(config, dict) else {}
    stages = chain_cfg.get("stages", {}) if isinstance(chain_cfg, dict) else {}
    normalization_stage = stages.get("normalization", {}) if isinstance(stages, dict) else {}
    if isinstance(normalization_stage, dict) and normalization_stage.get("input_topic"):
        return str(normalization_stage["input_topic"])
    return str(kafka_cfg.get("topicid", ""))


def _publish_end_of_stream(config: dict):
    if not _kafka_chain_enabled(config):
        return

    kafka_cfg = config.get("kafka", {}) if isinstance(config, dict) else {}
    bootstrap = f'{kafka_cfg.get("bootstrap_servers", "localhost")}:{kafka_cfg.get("port", 9092)}'
    topic_name = _source_topic_for_chain(config)
    if not topic_name:
        return

    eos_type = str((config.get("kafka_chain", {}) if isinstance(config, dict) else {}).get("end_of_stream", "end_of_stream"))
    message = {
        "message_type": eos_type,
        "stage": "source_producer",
        "upstream_stage": "",
        "payload": {
            "reason": "source_completed"
        },
    }

    producer = KafkaProducer({"bootstrap.servers": bootstrap})
    producer.produce(topic_name, key="source_producer", value=json.dumps(message).encode("utf-8"))
    producer.flush()
    print(f"[INFO] Published EOS marker to topic '{topic_name}'.")

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
        # Source producer should only wait for the first downstream stage by default.
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


def start_producer(config):
    global ACTIVE_JAVA_PROC
    java_path = os.path.abspath(config["simulator_path"])
    csv_path = config.get("data_source_B", "")
    csv_path = os.path.abspath(csv_path) if csv_path else ""
    kafka_bootstrap = f'{config["kafka"]["bootstrap_servers"]}:{config["kafka"]["port"]}'
    java_exec = shutil.which("java")
    if not java_exec:
        raise FileNotFoundError("'java' executable was not found in PATH.")

    _ensure_pipeline_ready(config)


    jar_path = _resolve_simulator_jar(java_path)
    java_proc = subprocess.Popen(
        [
            java_exec,
            "-jar",
            jar_path,
            f"--csv.file.path={csv_path}",
            f"--spring.kafka.producer.topic-id={config['kafka']['topicid']}",
            f"--spring.kafka.bootstrap-servers={kafka_bootstrap}",
        ],
        cwd=java_path,
        start_new_session=True
    )
    ACTIVE_JAVA_PROC = java_proc

    print(f"[INFO] Started Java producer process with PID {java_proc.pid}.")

    previous_sigint_handler = signal.getsignal(signal.SIGINT)
    completed_successfully = False
    try:
        signal.signal(signal.SIGINT, _handle_sigint)
        print(f"[INFO] Java producer is running. Press Ctrl+C to stop.")
        java_proc.wait()
        completed_successfully = java_proc.returncode == 0
    finally:
        signal.signal(signal.SIGINT, previous_sigint_handler)
        if completed_successfully:
            try:
                _publish_end_of_stream(config)
            except Exception as exc:
                print(f"[WARN] Failed to publish EOS marker: {exc}")
        print(f"[INFO] Java producer process with PID {java_proc.pid} has stopped.")
        _stop_process_group(java_proc, interrupt_first=True)
        ACTIVE_JAVA_PROC = None

# =========================
# Main
# =========================
if __name__ == '__main__':
    # load config file
    config_file = os.path.abspath(CONFIG_PATH)
    yaml = YAML()
    with open(config_file, 'r') as f:
        config = yaml.load(f)

    try:
        print("[INFO] Starting Kafka producer service...")
        producer_process = start_producer(config)
    except Exception as e:
        app_logger = write_log("logs", "main", "bug")
        app_logger.error(f"Fatal error in producer service: {str(e)}")
        print(f"Fatal error in producer service: {str(e)}")
        sys.exit(1)
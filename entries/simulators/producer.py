import glob
import os
import shutil
import signal
import subprocess
import sys
import time

from confluent_kafka import Producer as KafkaProducer
from ruamel.yaml import YAML
import pandas as pd

from utils.write_log import write_log
from utils import pipeline_io as _pipeline_io  # Registers per-step I/O metrics at exit.

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
    # Exit with code 0 to indicate graceful shutdown
    sys.exit(0)


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


def start_producer(config):
    global ACTIVE_JAVA_PROC
    java_path = os.path.abspath(config["simulator_path"])
    csv_path = config.get("data_source_B", "")
    csv_path = os.path.abspath(csv_path) if csv_path else ""
    kafka_bootstrap = f'{config["kafka"]["bootstrap_servers"]}:{config["kafka"]["port"]}'
    java_exec = shutil.which("java")
    if not java_exec:
        raise FileNotFoundError("'java' executable was not found in PATH.")

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
    previous_sigterm_handler = signal.getsignal(signal.SIGTERM)
    try:
        # Kubernetes sends SIGTERM (not SIGINT) when stopping a pod, so both need to route
        # through the same handler or the Java subprocess is left orphaned on pod teardown.
        signal.signal(signal.SIGINT, _handle_sigint)
        signal.signal(signal.SIGTERM, _handle_sigint)
        print(f"[INFO] Java producer is running. Press Ctrl+C to stop.")
        java_proc.wait()
    finally:
        signal.signal(signal.SIGINT, previous_sigint_handler)
        signal.signal(signal.SIGTERM, previous_sigterm_handler)
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

import argparse
import json
import os
import signal
import subprocess
import time

from ruamel.yaml import YAML
import pandas as pd
from confluent_kafka import Consumer, KafkaException, KafkaError, Producer as KafkaProducer
from confluent_kafka.admin import AdminClient, NewTopic

from utils.write_log import write_log
from utils.buffers import _clear_buffer_directory, _write_buffer, _write_eos


ACTIVE_JAVA_PROC = None
ACTIVE_CONSUMER = None
CONFIG_PATH = os.environ.get("EAER_CONFIG_PATH", "/app/config/examples/config-embedding.yaml")
BUFFER_DIR = "/app/data/buffers/raw_data"

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


# =========================
# Driver
# =========================

def safe_read_csv(path):
    if not path:
        return pd.DataFrame()

    if not os.path.exists(path):
        return pd.DataFrame()

    return pd.read_csv(path)

def _serialize_for_json(obj):
    if isinstance(obj, pd.DataFrame):
        return {"__dataframe__": True, "data": obj.to_dict(orient="records")}
    if isinstance(obj, dict):
        return {k: _serialize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize_for_json(item) for item in obj]
    return obj

def kafka_driver(config):
    try:
        _ensure_kafka_topic_ready(config)
        offset_reset = str(config.get("kafka", {}).get("auto_offset_reset", "earliest")).strip().lower()
        if offset_reset not in {"earliest", "latest"}:
            offset_reset = "earliest"
        print(f"[kafka] consumer auto.offset.reset={offset_reset}", flush=True)

        # prepare kafka consumer
        consumer = Consumer({
            'bootstrap.servers': f'{config["kafka"]["bootstrap_servers"]}:{config["kafka"]["port"]}',
            'group.id': config['kafka']["groupid"],
            'auto.offset.reset': offset_reset,
            'enable.auto.commit': False,
            'max.poll.interval.ms': 1800000,
        })

        # subscribe a topic
        consumer.subscribe([config['kafka']['topicid']])

    except Exception as e:
        app_logger = write_log("logs", "main", "bug")
        app_logger.error(f"Fatal error in consumer service: {str(e)}")
        print(f"Fatal error in consumer service: {str(e)}", flush=True)
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

def _daemon():
    while True:
        time.sleep(1)


def _commit_processed_offsets(consumer: Consumer, msg, reason: str):
    if consumer is None or msg is None:
        return
    try:
        consumer.commit(message=msg, asynchronous=False)
        print(f"[kafka] committed offsets after {reason}", flush=True)
    except KafkaException as exc:
        print(f"[WARNING] Failed to commit Kafka offsets after {reason}: {exc}", flush=True)


def start_consumer(config):
    global ACTIVE_JAVA_PROC, ACTIVE_CONSUMER
    data_store = {
        "raw_data": None,
        "is_training": False,
    }
    
    # start kafka
    poll_timeout = 5
    max_empty_polls = 5
    startup_grace_seconds =  120
    empty_poll_count = 0

    consumer = kafka_driver(config)
    if consumer is None:
        raise RuntimeError("Kafka consumer initialization failed.")
    producer = None
    
    ACTIVE_CONSUMER = consumer
    previous_sigint_handler = signal.getsignal(signal.SIGINT)

    data_buffer = []
    last_valid_msg = None
    _clear_buffer_directory(BUFFER_DIR)
    has_received_messages = False
    started_at = time.time()
    try:
        signal.signal(signal.SIGINT, _handle_sigint)
        while True:
            msg = consumer.poll(poll_timeout)  # Non-blocking batch pull

            # no new message
            if msg is None:  
                empty_poll_count += 1
                print(f'# empty poll count: {empty_poll_count}', flush=True)
                if empty_poll_count >= max_empty_polls:
                    elapsed = time.time() - started_at
                    if not has_received_messages and elapsed < max(startup_grace_seconds, 0.0):
                        print(
                            "[INFO] Still waiting for first source message "
                            f"({elapsed:.1f}s < {startup_grace_seconds:.1f}s). Continuing...",
                            flush=True,
                        )
                        empty_poll_count = 0
                        continue
                    print("[INFO] No new messages for a while. Exiting consumer loop.", flush=True)
                    if data_buffer:
                        data_store['raw_data'] = pd.DataFrame(data_buffer)
                        _write_buffer(data_buffer, BUFFER_DIR, extension="csv")
                        _commit_processed_offsets(consumer, last_valid_msg, "buffer flush on idle")
                        data_buffer = []
                        _write_eos(BUFFER_DIR, reason="source_consumer_idle_timeout")
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
            payload = msg.value()
            if payload is None:
                print("[WARNING] Received null Kafka payload; skipping message.", flush=True)
                continue
            try:
                decoded_payload = payload.decode("utf-8").strip()
            except UnicodeDecodeError as exc:
                print(f"[WARNING] Failed to decode Kafka payload as UTF-8: {exc}; skipping message.", flush=True)
                continue
            if not decoded_payload:
                print("[WARNING] Received empty Kafka payload; skipping message.", flush=True)
                continue
            try:
                metadata = json.loads(decoded_payload)
            except json.JSONDecodeError as exc:
                preview = decoded_payload[:200]
                print(
                    f"[WARNING] Invalid JSON payload from Kafka: {exc}. payload_preview={preview!r}; skipping message.",
                    flush=True,
                )
                continue
            if not isinstance(metadata, dict):
                print(
                    f"[WARNING] Kafka payload is not a JSON object (type={type(metadata).__name__}); skipping message.",
                    flush=True,
                )
                continue
            data_buffer.append(metadata)
            last_valid_msg = msg
            if len(data_buffer) >= config["kafka"]["window_count"]:
                data_store['raw_data'] = pd.DataFrame(data_buffer)
                _write_buffer(data_buffer, BUFFER_DIR, extension="csv")
                _commit_processed_offsets(consumer, last_valid_msg, "window flush")
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
        _daemon()

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
        print(f"Fatal error in consumer service: {str(e)}", flush=True)
    total_end = time.perf_counter()
    print(f"Total execution time: {total_end - total_start:.2f} seconds", flush=True)
import argparse
import json
import os
import argparse
import signal
import subprocess
import time
from ruamel.yaml import YAML
import pandas as pd
from confluent_kafka import Consumer, KafkaException, KafkaError
from confluent_kafka.admin import AdminClient, NewTopic

from utils.write_log import write_log

ACTIVE_JAVA_PROC = None
ACTIVE_CONSUMER = None
CONFIG_PATH = os.environ.get("EAER_CONFIG_PATH", "/app/config/examples/config-embedding.yaml")

# =========================
# endpoints
# =========================

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
    poll_timeout = 5
    max_empty_polls = 5
    empty_poll_count = 0

    consumer = kafka_driver(config)

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
                        # Lancer le pipeline argo "embdding-inference"
                        print("[DEBUG]: Final data batch before exit:", data_store['raw_data'])
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
                # Lancer le pipeline argo "embdding-inference"
                print("[DEBUG]: Data:", data_store['raw_data'])
                data_buffer = []
    finally:
        signal.signal(signal.SIGINT, previous_sigint_handler)
        if ACTIVE_CONSUMER is not None:
            try:
                ACTIVE_CONSUMER.close()
            except Exception:
                pass
            ACTIVE_CONSUMER = None
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
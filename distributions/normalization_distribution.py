from pandas import DataFrame
import pandas as pd
import socket
import json

from pipeline.normalization import sequence_generating_m1, index_normalization

listener_host = "0.0.0.0"
listener_port = 8080
manager_service = "manager-service"

# =========================
# Payload Serialization/Deserialization
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

def deserialize_from_json(obj):
    """Reconstruct non-JSON-serializable objects from JSON-safe format."""
    if isinstance(obj, dict):
        if obj.get("__dataframe__"):
            return pd.DataFrame(obj.get("data", []))
        else:
            return {k: deserialize_from_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [deserialize_from_json(item) for item in obj]
    else:
        return obj

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

                # Deserialize inputs
                inputs = deserialize_from_json(request.get("inputs", {}))
                output = function_dispatcher(task, config=request.get("config"), **inputs)

                callback = request.get("callback") or {}
                callback_host = callback.get("host")
                callback_port = callback.get("port")

                if callback_host and callback_port:
                    # Serialize output before sending
                    serialized_output = serialize_for_json(output) if output is not None else None
                    response = {
                        "protocol_version": "2.0",
                        "task": task,
                        "status": "ok" if output is not None else "error",
                        "result": serialized_output if output is not None else f"Function '{task}' not found",
                    }
                    send(json.dumps(response), callback_host, int(callback_port))
                    print(f"Sent response to callback at {callback_host}:{callback_port}")
                else:
                    print("No callback provided; skipping response send.")

            except json.JSONDecodeError as e:
                print(f"Invalid JSON request: {e}")
            except Exception as e:
                print(f"Listener error: {e}")

def send(payload, service, port):
    try:
        with socket.create_connection((service, port), timeout=5) as sock:
            data = (payload + "\n").encode("utf-8")
            sock.sendall(data)
    except Exception as e:
        print(f"Error sending callback to {service}:{port}: {e}")

def function_dispatcher(function_name: str, config, **kwargs):
    function_map = {
        "normalization": normalization,
    }
    func = function_map.get(function_name)
    if func is not None:
        return func(config=config, **kwargs)
    else:
        print(f"Function {function_name} not found in dispatcher.")
        return ""

if __name__ == '__main__':
    listener()
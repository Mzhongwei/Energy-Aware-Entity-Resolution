import ast
import csv
from io import StringIO
import math
import os
import re
import string
import warnings
import pathlib
import numpy as np
import pandas as pd
from copy import deepcopy
from datetime import datetime

from utils.pipeline_io import write_eos
from utils.pipeline_io import serialize_for_json, write_step_output

try:
    from sklearn.decomposition import PCA
except ModuleNotFoundError:
    PCA = None

try:
    from ruamel.yaml import YAML
except ModuleNotFoundError:
    YAML = None

TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
OUTPUT_FORMAT = "# {:.<60} {}"

POSSIBLE_TASKS = ["smatch", "batch", "evaluation"]
POSSIBLE_CG_METHODS = {"fullindexing", "key-blocking", "token-blocking", "minhash-lsh"}
CG_METHOD_ALIASES = {
    "key": "key-blocking",
    "token": "token-blocking",
    "minhash": "minhash-lsh",
}

OUTPUT_CODE = "%Y%m%d_%H%M%S"

### check configuration ###
def _merge_with_defaults(user_config, default_config):
    """
    The default configuration is recursively merged into the user configuration, which takes precedence.
    """
    if not isinstance(default_config, dict):
        return user_config 
    
    merged = deepcopy(default_config)
    for key in user_config:
        if isinstance(user_config[key], dict) and key in merged:
            merged[key] = _merge_with_defaults(user_config[key], merged[key])
        else:
            merged[key] = user_config[key]
    return merged

def _verify_gwe(config):
    if config["graph"]["smoothing_method"] not in ["log", "no", "IDF", "ICF"]:
        raise ValueError("Unknown smoothing_method {}".format(config["smoothing_method"]))

    if config["embeddings"]["training_algorithm"] not in ["word2vec", "fasttext"]:
        raise ValueError(
            "Unknown training algorithm {}.".format(config["training_algorithm"])
        )
    if config["embeddings"]["learning_method"] not in ["skipgram", "CBOW"]:
        raise ValueError("Unknown learning method {}".format(config["learning_method"]))

    return config
                
def _verify_sk(config):            
    if config["kafka"]["window_strategy"] not in ["count", "time"]:
        raise ValueError("Expected sliding window strategy, pls choose between [\"count\", \"time\"]")
    elif config["kafka"]["window_strategy"] == "count":
        if "window_count" not in config["kafka"]:
            raise ValueError("Expected window_count value.")
    elif config["kafka"]["window_strategy"] == "time":
        if "window_time" not in config["kafka"]:
            raise ValueError("Expected window_time value.")
    
    if config["similarity_list"]["output_format"] not in ["db", "json", "parquet", "graphml"]:
        raise ValueError("output_format must be one of ['db', 'json', 'parquet']")
    if config["similarity_list"]["strategy_suppl"] not in ["faiss", "basic"]:
        raise ValueError('''strategy_suppl must be one of ["faiss", "basic"]''')
    
    return config

def _check_file(config, files):
    for file in files:
        if file not in config:
            raise ValueError(f"please specify the file {file}")
        else:
            file_path = config[file]
            if file_path == "" or (file_path != "" and not os.path.exists(file_path)):
                raise IOError("File {} not found. ".format(file_path))


def _check_positive_int(value, field_name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")


def _check_field_config(field_value, field_name):
    if field_value is None:
        return
    if isinstance(field_value, str):
        return
    if isinstance(field_value, (list, tuple, set)):
        if not all(isinstance(item, str) and item.strip() for item in field_value):
            raise ValueError(f"{field_name} must contain non-empty strings.")
        return
    raise ValueError(f"{field_name} must be a string or a list/set of strings.")


POSSIBLE_CG_METHODS = {"fullindexing", "key-blocking", "token-blocking", "minhash-lsh"}
CG_METHOD_ALIASES = {
    "key": "key-blocking",
    "token": "token-blocking",
    "minhash": "minhash-lsh",
}

def _verify_candidate_generation(config: dict) -> dict:
    """
    Validate candidate_generation config.
    """
    if not isinstance(config, dict):
        raise ValueError("config must be a dictionary.")

    cg_cfg = config.get("candidate_generation")
    if cg_cfg is None:
        raise ValueError("Missing required block: candidate_generation (defaults should provide it).")
    if not isinstance(cg_cfg, dict):
        raise ValueError("candidate_generation must be a dictionary.")

    # 1) method
    raw_method = cg_cfg.get("method", config.get("cg_method", "fullindexing"))
    method = str(raw_method).strip().lower()
    method = CG_METHOD_ALIASES.get(method, method)
    if method not in POSSIBLE_CG_METHODS:
        raise ValueError(
            f"candidate_generation.method must be one of {sorted(POSSIBLE_CG_METHODS)}, got '{method}'."
        )
    cg_cfg["method"] = method

    # 2) top_k, random_seed
    if "top_k" in cg_cfg and cg_cfg["top_k"] is not None:
        _check_positive_int(cg_cfg["top_k"], "candidate_generation.top_k")

    if "random_seed" in cg_cfg and cg_cfg["random_seed"] is not None:
        if isinstance(cg_cfg["random_seed"], bool) or not isinstance(cg_cfg["random_seed"], int):
            raise ValueError("candidate_generation.random_seed must be an integer.")

    # 3) method-specific required fields (no defaults injected here)
    if method == "key-blocking":
        kb = cg_cfg.get("key_blocking")
        if not isinstance(kb, dict):
            raise ValueError("candidate_generation.key_blocking must be a dictionary.")
        keys = kb.get("keys")
        if not isinstance(keys, list) or not keys or not all(isinstance(k, str) and k.strip() for k in keys):
            raise ValueError("candidate_generation.key_blocking.keys must be a non-empty list of strings.")

    if method == "token-blocking":
        tb = cg_cfg.get("token_blocking")
        if not isinstance(tb, dict):
            raise ValueError("candidate_generation.token_blocking must be a dictionary.")
        field = tb.get("field")
        _check_field_config(field, "candidate_generation.token_blocking.field")
        if field in (None, ""):
            raise ValueError("candidate_generation.token_blocking.field is required.")

    if method == "minhash-lsh":
        mh = cg_cfg.get("minhash_lsh")
        if not isinstance(mh, dict):
            raise ValueError("candidate_generation.minhash_lsh must be a dictionary.")
        field = mh.get("field")
        _check_field_config(field, "candidate_generation.minhash_lsh.field")
        if field in (None, ""):
            raise ValueError("candidate_generation.minhash_lsh.field is required.")
        for field_name in ["num_perm", "bands", "rows_per_band", "shingle_size"]:
            if field_name in mh and mh[field_name] is not None:
                _check_positive_int(int(mh[field_name]), f"candidate_generation.minhash_lsh.{field_name}")

        if "seed" in mh and mh["seed"] is not None:
            if isinstance(mh["seed"], bool) or not isinstance(mh["seed"], int):
                raise ValueError("candidate_generation.minhash_lsh.seed must be an integer.")

    # write back
    config["candidate_generation"] = cg_cfg
    return config


def validate_candidate_generation_config(config: dict) -> dict:
    return _verify_candidate_generation(config)
            
def check_config_validity(config):
    if YAML is None:
        raise ModuleNotFoundError("ruamel.yaml is required to validate YAML configs.")
    yaml = YAML()
    #### Set default values
    if config["task"] not in POSSIBLE_TASKS:
        raise ValueError("Task {} not supported.".format(config["task"]))
 
    if "evaluation" in config["task"]:
       
        defaul_path = "config/default/default-evaluation.yaml"
        with open(defaul_path, 'r') as f:
            default = yaml.load(f)
        config = _merge_with_defaults(config, default)
        _check_file(config, ['similarity_file', 'match_file'])
        config["output_format"] = pathlib.Path(config["similarity_file"]).suffix[1:]
    elif "smatch" in config["task"]:
        defaul_path = "config/default/default-stream.yaml"
        with open(defaul_path, 'r') as f:
            default = yaml.load(f)
        config = _merge_with_defaults(config, default)
        config = _verify_gwe(config)
        config = _verify_sk(config)
    elif "batch" in config["task"]:
        defaul_path = "config/default/default-batch.yaml"
        with open(defaul_path, 'r') as f:
            default = yaml.load(f)
        config = _merge_with_defaults(config, default)
        config = _verify_gwe(config)
        _check_file(config, ['dataset_file'])

    config = validate_candidate_generation_config(config)

    if not os.path.exists(config['log']['path']) or config['log']['path'] == "":
        config['log']['path'] = "pipeline/logging" 
        print('!!! Invalid log path, change to path "pipeline/logging"')

    return config

### data preparation ###
def convert_token_value(original_value):
    """
    Convert a cell value into a clean string. Try to evaluate literals using ast.literal_eval first
    - If it's NaN, ***None***

    - If it's list, determine elements in the list are numeric or not
    - If it's dict, for each attribute, concat and reform as a string
    - If it's a float, round to int and convert to string
    - Otherwise, just ***str()*** the value
    - then put all values into a list except None

    : return1: list of value or None
    : return2: bool (true: numeric; false: str)

    Modify this function if we need to treat other data types
    """
    if isinstance(original_value, np.ndarray):
        original_value = original_value.tolist()
    elif isinstance(original_value, (tuple, set)):
        original_value = list(original_value)

    if original_value in ("", None):
        return None, False

    if isinstance(original_value, list):
        cleaned_values = []
        is_numeric = True
        for el in original_value:
            if el in ("", None):
                continue
            if pd.isna(el):
                continue
            cleaned_values.append(clean_str(str(el)))
            is_numeric = is_numeric and isinstance(el, (int, float, np.integer, np.floating))
        return (cleaned_values or None), (is_numeric if cleaned_values else False)

    try:
        # Try to safely evaluate a literal value (e.g., "123", "[1, 2]", etc.)
        cell_value = ast.literal_eval(str(original_value))

        # Handle float or int
        if isinstance(cell_value, (int, float)):
            if isinstance(cell_value, float) and math.isnan(cell_value):
                return None, False
            return [str(int(cell_value))], True
        # Handle list
        elif isinstance(cell_value, list):
            if cell_value:
                is_numeric = all(isinstance(el, (int, float)) for el in cell_value)
            else: 
                is_numeric = False
            return [clean_str(str(el)) for el in cell_value], is_numeric
        # Handle dict
        elif isinstance(cell_value, dict):
            return [f"{clean_str(str(key))}_{clean_str(str(value))}" for key, value in cell_value.items()], False
        # other object (like list, dict, string)
        return [clean_str(str(cell_value))], False

    except (ValueError, SyntaxError, OverflowError):
        return [clean_str(str(original_value))], False
    
def clean_str(value):
    value = value.lower().strip()
    # Replace all non-alphanumeric characters with “_”
    value = re.sub(r'[^a-z0-9]+', '_', value)
    # Remove the underscores at the beginning and end
    value = value.strip('_')
    return value

def clean_date(value):
    date_formats = [
            ("%Y-%m-%d", "%Y%m%d"),
            ("%Y/%m/%d", "%Y%m%d"),
            ("%d-%m-%Y", "%Y%m%d"),
            ("%d/%m/%Y", "%Y%m%d"),
            ("%Y-%m", "%Y%m"),
            ("%Y/%m", "%Y%m"),
        ]

    for fmt_in, fmt_out in date_formats:
        try:
            dt = datetime.strptime(value, fmt_in)
            return str(dt.strftime(fmt_out))
        except ValueError:
            continue
    return value

def data_cleaning(input):
    if isinstance(input, str):
        value = clean_str(input)
        res = clean_date(value)
        return res
    else:
        return str(input)

def parse_idx_suffix(word: str, prefix: str = "idx__"):
    """
    Parse 'idx__<number>' -> <number> as int.
    Be tolerant to '123.0' etc. Return None on failure.
    """
    if not isinstance(word, str) or not word.startswith(prefix):
        return None
    try:
        suffix = word.split("__", 1)[1]
        return int(float(suffix))
    except Exception:
        return None


def state_config(config: dict) -> dict:
    if not isinstance(config, dict):
        return {}
    value = config.get("state_management", {}) or config.get("state_config", {}) or {}
    return value if isinstance(value, dict) else {}

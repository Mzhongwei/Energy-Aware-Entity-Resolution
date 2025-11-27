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
from sklearn.decomposition import PCA
from copy import deepcopy
from ruamel.yaml import YAML
from datetime import datetime

TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
OUTPUT_FORMAT = "# {:.<60} {}"

POSSIBLE_TASKS = ["smatch", "batch", "evaluation"]

OUTPUT_CODE = "%Y%m%d_%H%M%S"

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
            
def check_config_validity(config):
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

    if not os.path.exists(config['log']['path']) or config['log']['path'] == "":
        config['log']['path'] = "pipeline/logging" 
        print('!!! Invalid log path, change to path "pipeline/logging"')

    return config

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
    if original_value in ("", None):
        return None, False

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
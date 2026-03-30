import pandas as pd
import numpy as np
import os
import re
import secrets
import string
try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable, **kwargs):
        return iterable
    
from utils.utils import convert_token_value, data_cleaning


_ID_COLUMN_RE = re.compile(r"(^id$|_id$|\.id$)", re.IGNORECASE)

def sequence_generating_m1(df):
    """
    Generate comparable sequences for llm
    label(s) value: 0 or 1
    text_n value: Concatenate the property name and property value (ex. title-title_value_1 name-name_value_1)

    :param df: [_id, label, table1.id1, table2.id, table1.title, table2.title, table1.name, table2.name, ...]
    :return df_result: [text1, text2, labels]
    """
    df_result = pd.DataFrame(columns=["text1", "text2", "labels"])
    for _, df_row in tqdm(df.iterrows(), total=len(df), desc="# Reading data"):
        text1 = ""
        text2 = ""
        labels = 0
        for col in df.columns:
            col_name_list = col.split(".")
            if col == "label":
                labels = df_row[col]
            elif len(col_name_list) > 1 and col_name_list[1] != "id":
                if int(col_name_list[0][-1]) == 1:
                    text1 = text1 + str(col_name_list[1]) + str(df_row[col])
                elif int(col_name_list[0][-1]) == 2:
                    text2 = text2 + str(col_name_list[1]) + str(df_row[col])
        text1 = data_cleaning(text1)
        text2 = data_cleaning(text2)
        df_result.loc[len(df_result)] = [text1, text2, labels]
    return df_result



def index_normalization(config, raw_data):
    """
    incremental mode: index and normalization
    * 'rid' should appear in meta_path if it is setted
    * note: change directly raw_data dataFrame - may be risky
    
    :param config: 
    :param raw_data: pd.DataFrame
    :return: pd.DataFrame
    """
    def _random_id(prefix="idx__", length=12):
        alphabet = string.ascii_letters + string.digits
        return prefix + "".join(secrets.choice(alphabet) for _ in range(length))

    def _generate_unique_rids(size, prefix="idx__", length=12):
        seen = set()
        rids = []
        while len(rids) < size:
            rid = _random_id(prefix=prefix, length=length)
            if rid in seen:
                continue
            seen.add(rid)
            rids.append(rid)
        return rids

    def _original_id_columns(df):
        return [col for col in df.columns if col != "rid" and _ID_COLUMN_RE.search(str(col))]
    
    # ===== for evaluation =====
    def _mapping_output_path(cfg):
        state_cfg = cfg.get("state_management", {}) if isinstance(cfg, dict) else {}
        save_dir = state_cfg.get("id_mapping-dir", "data/id_mapping")
        name = state_cfg.get("id_mapping-name") or cfg.get("version_name", "test")
        os.makedirs(save_dir, exist_ok=True)
        return os.path.join(save_dir, f"{name}.csv")

    # ===== index =====
    id_cols = _original_id_columns(raw_data)
    raw_data["rid"] = _generate_unique_rids(len(raw_data))

    if id_cols:
        mapping_df = raw_data[["rid"] + id_cols].copy()
        mapping_df.to_csv(_mapping_output_path(config), index=False)

    meta_path = config.get("meta_path", [])
    result = None
    if meta_path:
        # with meta_path 
        paths = meta_path if isinstance(meta_path[0], list) else [meta_path]

        meta_node = set()
        for path in paths:
            meta_node.update(path)
        cols = [col for col in raw_data.columns if col in meta_node]
    else:
        # without meta_path 
        cols = [col for col in raw_data.columns]
    
    # ===== normalization ===== 
    # mapping
    col_to_idx = {col: idx for idx, col in enumerate(cols)}

    # Convert to a list of tuples
    data_tuples = list(raw_data[cols].itertuples(index=False, name=None))
    result_rows = []
    for row_data in tqdm(data_tuples, total=len(raw_data), desc="# Indexing/Normalizing meta_path"):
        row_result = {}

        for col in cols:
            col_idx = col_to_idx[col]
            cell_value = row_data[col_idx]

            if col != 'rid':
                token_list, _ = convert_token_value(cell_value)
            else:
                token_list = [cell_value]
          
            row_result[col] = token_list

        result_rows.append(row_result)

    result = pd.DataFrame(result_rows, columns=cols)
    return result

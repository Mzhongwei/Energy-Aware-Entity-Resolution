import copy
import string
import time
import pandas as pd
import numpy as np
from collections import defaultdict
from igraph import Graph
from concurrent.futures import ThreadPoolExecutor
from utils.utils import data_cleaning  # 保留你原有的导入路径

# 全局字符索引映射缓存
CHAR_INDEX = {c: i for i, c in enumerate(string.ascii_lowercase + string.digits)}

def timeit(func):
    from functools import wraps
    @wraps(func)
    def wrapper(*args, **kwargs):
        st = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} executed in {time.time() - st:.4f}s")
        return result
    return wrapper

def tokenize_record(row, cols):
    tokens = set()
    for col in cols:
        cleaned = data_cleaning(row[col])
        tokens.update(cleaned.split('_'))
    return frozenset(token for token in tokens if token and token != 'nan')

def compute_bitmask_int(tokens: set[str]) -> int:
    mask = 0
    for token in tokens:
        for c in token.lower():
            if c in CHAR_INDEX:
                mask |= 1 << CHAR_INDEX[c]
    return mask

def update_pairs_list(id_a, id_b, list_pairs, index_pairs):
    updated = False
    for pair in list_pairs:
        if id_a in pair or id_b in pair:
            pair.update([id_a, id_b])
            updated = True
            break
    if not updated:
        list_pairs.append({id_a, id_b})
    index_pairs.update([id_a, id_b])
    return list_pairs, index_pairs

def compare_group(group):
    local_pairs = []
    local_index = set()
    n = len(group)
    for i in range(n - 1):
        rid_i, tokens_i = group[i]
        for j in range(i + 1, n):
            rid_j, tokens_j = group[j]
            if tokens_i == tokens_j:
                local_pairs, local_index = update_pairs_list(rid_i, rid_j, local_pairs, local_index)
    return local_pairs, local_index

def comparison_internal(list_to_compare: pd.DataFrame):
    cols_to_compare = [col for col in list_to_compare.columns if col != "rid"]
    grouped = defaultdict(list)
    for _, row in list_to_compare.iterrows():
        rid = row["rid"]
        tokens = tokenize_record(row, cols_to_compare)
        bitmask = compute_bitmask_int(tokens)
        grouped[bitmask].append((rid, tokens))

    pairs = []
    pairs_index = set()

    with ThreadPoolExecutor() as executor:
        results = executor.map(compare_group, grouped.values())
        for local_pairs, local_index in results:
            pairs.extend(local_pairs)
            pairs_index.update(local_index)

    return pairs, pairs_index

def preprocessing_batch(list_to_compare: pd.DataFrame):
    pairs, pairs_index = comparison_internal(list_to_compare)
    pairs_copy = [set(s) for s in pairs]
    if pairs:
        to_delete = set()
        for el in pairs:
            el_copy = el.copy()
            el_copy.discard(next(iter(el_copy)))
            to_delete.update(el_copy)
        drop_indices = list_to_compare[list_to_compare['rid'].isin(to_delete)].index
        list_to_compare.drop(index=drop_indices, inplace=True)
    return list_to_compare, pairs_copy, pairs_index

def preprocessing_incremental(list_targets: pd.DataFrame, list_candidats: dict):
    cols_to_compare = [col for col in list_targets.columns if col != "rid"]
    list_deduplication, pairs_list, pairs_index = preprocessing_batch(list_targets)

    candidats_token_map = {}
    for rid_j, record_j in list_candidats.items():
        tokens_j = frozenset(
            token
            for item in record_j
            for token in data_cleaning(item).split('_')
            if token and token != 'nan'
        )
        bitmask_j = compute_bitmask_int(tokens_j)
        candidats_token_map[rid_j] = (tokens_j, bitmask_j)

    token_cache = {}
    bitmask_cache = {}
    pairs_dict = {}
    to_delete = []

    for idx, row in list_deduplication.iterrows():
        rid_i = row["rid"]
        if rid_i not in token_cache:
            tokens_i = tokenize_record(row, cols_to_compare)
            bitmask_i = compute_bitmask_int(tokens_i)
            token_cache[rid_i] = tokens_i
            bitmask_cache[rid_i] = bitmask_i
        else:
            tokens_i = token_cache[rid_i]
            bitmask_i = bitmask_cache[rid_i]

        for rid_j, (tokens_j, bitmask_j) in candidats_token_map.items():
            if bitmask_i == bitmask_j and tokens_i == tokens_j:
                to_delete.append(idx)
                if pairs_index:
                    found = False
                    for pair in pairs_list:
                        if rid_i in pair:
                            pairs_dict[rid_j] = pair
                            pairs_list.remove(pair)
                            found = True
                            break
                    if not found:
                        pairs_dict[rid_j] = rid_i
                else:
                    pairs_dict[rid_j] = rid_i
                break

    list_deduplication = list_deduplication.drop(index=to_delete)
    return list_deduplication, pairs_dict, pairs_list, pairs_index

# 示例主函数
@timeit
def run_test():
    data = {
        "rid": ["001", "002", "003"],
        "country": ["france", "France", ""],
        "name": ["Amie", np.nan, "Amie France"],
        "date": ["1999-03-03", "2025", "03/03/1999"],
        "num": [50, 40, 50]
    }

    data_1 = {
        "rid": ["002", "003"],
        "country": ["France", ""],
        "name": [np.nan, "Amie France"],
        "date": ["2025", "03/03/1999"],
        "num": [40, 50]
    }

    data_2 = {
        "rid": ["001", "002", "003", "004", "018"],
        "country": ["france", "France", "", 2025, "USA"],
        "name": ["Amie", np.nan, "Amie France", "France", "English"],
        "date": ["1999-03-03", "2025", "03/03/1999", "nan", "03/03/2015"],
        "num": [50, 40, 50, 40, 6]
    }

    candidats = {
        "005": ["France", 50, "aMIE", "1999-03-03"],
        "006": ["France", 40, "Italie", "2025"],
        "007": ["Italie", 50, "Italie", "1999-03-03"],
        "009": ["Chine", 6, "France", "1999-03-03"],
        "012": ["USA", 6, "English", "2015-03-03"],
    }

    df_bat = pd.DataFrame(data)
    df_inc = pd.DataFrame(data)
    print(preprocessing_incremental(df_inc, candidats))
    print(preprocessing_incremental(pd.DataFrame(data_1), candidats))
    print(preprocessing_incremental(pd.DataFrame(data_2), candidats))

if __name__ == '__main__':
    run_test()

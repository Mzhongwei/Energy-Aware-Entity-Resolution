import copy
import string
import time
import pandas as pd
from collections import defaultdict
from igraph import Graph

from utils.utils import *

def tokenize_record(row, cols):
    tokens = set()
    for col in cols:
        cleaned = data_cleaning(row[col])
        tokens.update(cleaned.split('_'))
        
        tokens = {x for x in tokens if x and x != 'nan'}
        
    return frozenset(tokens)

def compute_bitmask_int(tokens: set[str]) -> int:
    chars = string.ascii_lowercase + string.digits
    char_index = {c: i for i, c in enumerate(chars)}
    mask = 0
    for token in tokens:
        for c in token.lower():
            if c in char_index:
                mask |= 1 << char_index[c] 
    return mask  # 返回整数

def comparison_internal(list_to_compare: pd.DataFrame):
    cols_to_compare = [col for col in list_to_compare.columns if col != "rid"]
    grouped = defaultdict(list)
    pairs = []
    pairs_index = set()

    prepared_list = {}
    for _, row in list_to_compare.iterrows():
        rid = row["rid"]
        
        tokens = tokenize_record(row, cols_to_compare)
        bitmask = compute_bitmask_int(tokens)
        grouped[bitmask].append((rid, tokens))

    for group in grouped.values():
        n = len(group)
        for i in range(n - 1):
            rid_i, tokens_i = group[i]
            for j in range(i + 1, n):
                rid_j, tokens_j = group[j]
                if tokens_i == tokens_j:
                    # print("get exact matchs")
                    # print(f"in em_1: {rid_i}, {tokens_i}")
                    # print(f"in em_2: {rid_j}, {tokens_j}")
                    pairs, pairs_index = update_pairs_list(rid_i, rid_j, pairs, pairs_index)
    return pairs, pairs_index

def preprocessing_batch(list_to_compare: pd.DataFrame):
    pairs, pairs_index = comparison_internal(list_to_compare)
    pairs_copy = copy.deepcopy(pairs)
    if pairs:
        to_delete = set()
        for el in pairs:
            el_copy = el.copy()
            el_copy.discard(next(iter(el_copy)))
            to_delete.update(el_copy)
        drop_indices = list_to_compare[list_to_compare['rid'].isin(to_delete)].index
        list_to_compare.drop(index=drop_indices, inplace=True)
    return list_to_compare, pairs_copy, pairs_index

def update_pairs_list(id_a, id_b, list_pairs, index_pairs):
    """
    : return list_pairs: a list of set 
    : return index_pairs: a list of str
    """
    if id_a in index_pairs or id_b in index_pairs:
        for el in list_pairs:
            if id_a in el or id_b in el:
                el.add(id_a)
                el.add(id_b)
        index_pairs.add(id_a)
        index_pairs.add(id_b)
    else:
        list_pairs.append({id_a, id_b})
        index_pairs.add(id_a)
        index_pairs.add(id_b)
    return list_pairs, index_pairs

def preprocessing_incremental(list_targets: pd.DataFrame, list_candidats: dict): # TODO: 可能需要改进嵌套结构
    cols_to_compare = [col for col in list_targets.columns if col != "rid"]
    list_deduplication, pairs_list, pairs_index = preprocessing_batch(list_targets)

    candidats_token_map = {
                            rid_j: frozenset(
                                token
                                for item in record_j
                                for token in data_cleaning(item).split('_')
                                if token and token != 'nan'
                            )
                            for rid_j, record_j in list_candidats.items()
                        }

    pairs_dict = {}
    to_delete = []
    for idx, row in list_deduplication.iterrows():
        rid_i = row["rid"]
        tokens_i = tokenize_record(row, cols_to_compare)
        bitmask_i = compute_bitmask_int(tokens_i)
        for rid_j, tokens_j in candidats_token_map.items():
            if compute_bitmask_int(tokens_j) == bitmask_i and tokens_i == tokens_j:
                # print("get exact matchs")
                # print(f"ex em_1: {rid_i}, {tokens_i}")
                # print(f"ex em_2: {rid_j}, {tokens_j}")
                to_delete.append(idx)
                pairs_dict[rid_j] = rid_i
                if pairs_index : # and r_df in pairs_index
                    for pair in pairs_list:
                        if rid_i in pair:
                            pairs_dict[rid_j] = pair
                            pairs_list.remove(pair)
                    
                break
    list_deduplication.drop(index=set(to_delete), inplace=True)
    return list_deduplication, pairs_dict, pairs_list, pairs_index


if __name__ == '__main__':

    # data = {
    # "rid": ["001", "002", "003"],
    # "country": ["france", "France", ""],
    # "name": ["Amie", np.nan, "Amie France"],
    # "date": ["1999-03-03", "2025", "03/03/1999"],
    # "num": [50, 40, 50]
    # }

    # data_1 = {
    # "rid": [ "002", "003"],
    # "country": [ "France", ""],
    # "name": [ np.nan, "Amie France"],
    # "date": [ "2025", "03/03/1999"],
    # "num": [ 40, 50]
    # }

    # data_2 = {
    # "rid": ["001", "002", "003", "004", "018"],
    # "country": ["france", "France", "", 2025, "USA"],
    # "name": ["Amie", np.nan, "Amie France", "France", "English"],
    # "date": ["1999-03-03", "2025", "03/03/1999", "nan", "03/03/2015"],
    # "num": [50, 40, 50, 40, 6]
    # }

    # df_bat = pd.DataFrame(data)
    # df_inc = pd.DataFrame(data)

    # candidats = {
    #     "005": ["France", 50, "aMIE", "1999-03-03"],
    #     "006": ["France", 40, "Italie", "2025"],
    #     "007": ["Italie", 50, "Italie", "1999-03-03"],
    #     "009": ["Chine", 6, "France", "1999-03-03"],
    #     "012": ["USA", 6, "English", "2015-03-03"],
    #     }
    
    # st = time.time()

    # print(comparison_internal(df_bat))
    # # print(comparison_external(df_inc, candidats))

    # print(preprocessing_batch(df_bat))
    # print(preprocessing_incremental(df_inc, candidats))
    # print()
    # print(preprocessing_incremental(pd.DataFrame(data_1), candidats))
    # print()
    # print(preprocessing_incremental(pd.DataFrame(data_2), candidats))

    # et = time.time()
    # print('Execution time:', et - st, 'seconds')
 
    # str_1 = { "RUE", "DE", "LA", "LIBERTE", "12"}
    str_1 = { "Rue", "du", "pommeret"

}
    print(bin(compute_bitmask_int(str_1)))
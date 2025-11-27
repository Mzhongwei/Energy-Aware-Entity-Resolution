import pandas as pd
from itertools import combinations
from collections import Counter
from Levenshtein import ratio

from utils.utils import *

def deterministe_comparison(r_a, r_b):
    '''compare if r_a and r_b is totally the same
    '''
    r_a = [x for x in r_a if pd.notna(x) and x != '' and x != 'nan']
    r_b = [x for x in r_b if pd.notna(x) and x != '' and x != 'nan']
    return Counter(r_a) == Counter(r_b)

def levenshtein_sim(r_a, r_b):
    sim_degree = ratio(r_a, r_b)
    return sim_degree

def jaccard_sim_deduplication(r_a, r_b):
    set1 = set(r_a)
    set2 = set(r_b)
    intersection = set1 & set2
    union = set1 | set2
    return len(intersection) / len(union)

def jaccard_sim_multise(r_a, r_b):
    c1 = Counter(r_a)
    c2 = Counter(r_b)

    all_items = set(c1.keys()).union(set(c2.keys()))

    intersection = sum(min(c1[x], c2[x]) for x in all_items)
    union = sum(max(c1[x], c2[x]) for x in all_items)

    return intersection / union if union != 0 else 1.0

def n_gram(r_a, r_b):
    # unigram
    set_a = unigram(r_a)
    set_b = unigram(r_b)
    if deterministe_comparison(set_a, set_b):
        return True
    else: 
        return False

def unigram(r):
    res = []
    for item in r:
        if type(item) == str:
            t = item.split('_')
        else:
            t = [item]
        res = res + t
    return res

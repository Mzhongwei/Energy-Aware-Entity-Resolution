from itertools import combinations
import json
import logging
import os
from pathlib import Path
from igraph import Graph
import sqlite3
import math
import pandas as pd
from utils.write_log import write_log
from scipy.stats import kendalltau

from dataprocessing.similarity_anlysis import similarity_analysis

def _get_ground_truth(ground_truth_file):
    matches = {}
    n_lines = 0
    with open(ground_truth_file, 'r', encoding='utf-8') as fp:
        for n, line in enumerate(fp.readlines()):
            if len(line.strip()) > 0:
                item, match = line.replace('_', '__').split(',')
                if item not in matches:
                    matches[item] = [match.strip()]
                else:
                    matches[item].append(match.strip())
                n_lines = n
        if n_lines == 0:
            raise IOError('Matches file is empty. ')
    return matches


def _get_similarity_list(similarity_file, output_format):
    sim_list = {}
    if output_format == "db":
        try:
            if Path(similarity_file).exists:
                conn = sqlite3.connect(similarity_file)
                cursor = conn.cursor()

                table = "matchinglist"
                primary_key = "id"
                value = "similarity"

                cursor.execute(f'SELECT {primary_key}, {value} FROM {table}')
                rows = cursor.fetchall()

                sim_list = {row[0]: json.loads(row[1]) for row in rows}
                conn.close()
            else:
                raise ValueError("[ERROR] Do not find similarity file.")
        except Exception as e:
            logging.error(f"Error processing message: {str(e)}")
    elif output_format == "json":
        print("parsing json file...")
        try:
            if Path(similarity_file).exists:
                with open(similarity_file, "r", encoding="utf-8") as f:
                    sim_list = json.load(f)
            else:
                raise ValueError("[ERROR] Do not find similarity file.")
        except Exception as e:
            logging.error(f"Error processing message: {str(e)}")
    elif output_format == "parquet":
        try:
            if Path(similarity_file).exists:
                data = pd.read_parquet(similarity_file)
                sim_list = dict(zip(data['key'], data['value'].apply(json.loads)))
            else:
                raise ValueError("[ERROR] Do not find similarity file.")
        except Exception as e:
            logging.error(f"Error processing message: {str(e)}")
    elif output_format == "graphml":
        try:
            if Path(similarity_file).exists():
                g = Graph.Read_GraphML(similarity_file)

                for v in g.vs:
                    node_name = v["name"]
                    members = v["members"].split(",") if "members" in v.attributes() else [node_name]

                    neighbors = g.neighbors(v.index, mode="OUT")
                    similarity_list = []
                    for n_idx in neighbors:
                        neighbor_name = g.vs[n_idx]["name"]
                        edge_id = g.get_eid(v.index, n_idx, directed=False, error=False)
                        if edge_id != -1:
                            sim = g.es[edge_id]["weight"]
                            similarity_list.append((sim, neighbor_name))

                    # Create the same similarity list for all members of the node
                    if similarity_list != []:
                        for member in members:
                            
                            sim_list[member] = similarity_list
                            # print(f"member: {member}, simialrity_list: {similarity_list}")
                    
                # Step 2: Add all pairwise combinations with internal similarity of 1.0 for the merged nodes
                for v in g.vs:
                    members = v["members"].split(",") if "members" in v.attributes() else []
                    if len(members) >= 2:
                        for a, b in combinations(members, 2):
                            # Bidirectionally join sim_list
                            sim_list.setdefault(a, []).append((1.0, b))
                            # print(f"member: {a}, simialrity_list: (1, {b})")

            else:
                raise ValueError("[ERROR] Do not find similarity file.")
        except Exception as e:
            logging.error(f"Error processing graphml: {str(e)}")
    return sim_list


def _get_match_pairs(data_dict, source_a):
    '''
    :param item: {'key': ['match1', 'match2'], 'key2': ['match1', 'match2']}
    :return: ex. (item, match1), (item, match2)
    '''
    matchpair_set = set()
    for key in data_dict:
        item_matched = int(key.split('__')[1])
        for value in data_dict[key]:
            item_matching = None
            # value ex. ['idx__27463', 'idx__19499', 'idx__18194', 'idx__14572', 'idx__64547', 'idx__35496', 'idx__22535']
            if isinstance(value, str):
                item_matching = value
            item_matching = int(item_matching.split('__')[1])

            # sort: facilitate deplication
            if item_matching > item_matched:
                el = (item_matched, item_matching)
            else:
                el = (item_matching, item_matched)
            
            # because of dataset constraint, limit the comparison between A and B
            if source_a == 0:
                print(f"[WARNING] calculate metrics without defining the records number of source A")
            elif source_a > 0:
                for id in el:
                    if id < source_a:
                        matchpair_set.add(el)
    return matchpair_set

def _get_similar_pairs(data_dict, n, appr, seuil):
    '''
    :param data_dict: {'key': [(sim_degree, 'match1'), (sim_degree, 'match2')], 'key2': [(sim_degree, 'match1'), (sim_degree, 'match2')]}
    :param n: select top n similarities
    :param appr: number of decimal places to be retained for similarity
    :return: ex. (item, match1), (item, match2)
    '''
    matchpair_set = set()
    s = ""
    for key in data_dict:
        # print(key)
        item_matched = int(key.split('__')[1])
        values = [[round(degree, appr), _] for degree, _ in data_dict[key]]
        if isinstance(n, int) and n>0:
            # find the n most matching scores
            scores = list(set(degree for degree, _ in values))
            scores = sorted(scores, reverse=True)[:n]
            values = [value for value in values if value[0] >= scores[len(scores)-1]]
        for item in values:
            item_matching = None
            # value ex. [[0.8194172382354736, 'idx__38260'], [0.8130440711975098, 'idx__51652']]
            if isinstance(item, list):
                item_matching = item[1]
                item_matching = int(item_matching.split('__')[1])

            if item[0] >= seuil and item_matched != item_matching:
                # sort: facilitate deplication
                if item_matching > item_matched:
                    el = (item_matched, item_matching)
                else:
                    el = (item_matching, item_matched)
                matchpair_set.add(el)
                res = f"pair: {el}, score: {item[0]} \n"
                # print(f"pair: {el}, score: {item[0]}")
                s = s + res
    return matchpair_set, s

def _get_similarity_pairs_with_degree(data_dict, n, appr):
    sim_list = {}
    for key in data_dict:
        item_matched = int(key.split('__')[1])
        values = [[round(degree, appr), _] for degree, _ in data_dict[key]]
        if isinstance(n, int) and n>0:
            # find the n most matching scores
            scores = list(set(degree for degree, _ in values))
            scores = sorted(scores, reverse=True)[:n]
            values = [value for value in values if value[0] >= scores[len(scores)-1]]
            sim_list[item_matched] = values
    return sim_list


def check_representation_quality(configuration):
    """
    Test the accuracy of matches by precisin, recall, f1
    :param configuration:
    """
    ground_truth_file = configuration['match_file']
    similarity_file = configuration['similarity_file']
    output_format = configuration['output_format']
    k = configuration['eva']['n_first']
    appr = configuration['eva']['approximate']
    source_a = configuration['source_a']
    # similarity_list example: {item: [matches]}
    similarity_list = _get_similarity_list(similarity_file, output_format)
    
    # matches example: {item: [matches]}
    matches = _get_ground_truth(ground_truth_file)
    actual_matches = _get_match_pairs(matches, source_a)
    total_relevant_matches = len(actual_matches)
    f_total_relevant_matches = total_relevant_matches
    print(f'ground truth: {total_relevant_matches}')

    f_seuil = 0
    f_total_predicted_matches = 0
    f_correct_matches = 0

    f_precision = 0
    f_recall = 0
    f_f1_score = 0
    f_k = 0
    for seuil in [i / 100 for i in range(5, 100, 5)]:
        for k in range (1, 11):
            correct_matches = 0
            predicted_matches, s = _get_similar_pairs(similarity_list, k, appr, seuil)
            
            total_predicted_matches = len(predicted_matches)
            
            correct_matches =  len(set(predicted_matches) & set(actual_matches)) 

            # Precision: Number of correct matches / Total predicted matches
            precision = correct_matches / total_predicted_matches if total_predicted_matches != 0 else 0.0
            # Recall: Number of correct matches / Total relevant matches
            recall = correct_matches / total_relevant_matches if total_relevant_matches != 0 else 0.0
            f1_score = 2*precision*recall / (precision + recall) if (precision + recall) != 0 else 0.0

            # if recall > f_recall:
            if f1_score > f_f1_score:
                f_seuil = seuil
                f_total_predicted_matches = total_predicted_matches
                f_correct_matches = correct_matches
                f_precision = precision
                f_recall = recall
                f_f1_score = f1_score
                f_k = k

    sim_list = _get_similarity_pairs_with_degree(similarity_list, k, appr)
    similarity_analysis(sim_list, actual_matches, configuration['output_file_name'], k)
    

    # print results
    print(f'''Evaluation result for {similarity_file}: \n threshold: {f_seuil} \n top k records: {f_k} \n correct matches: {f_correct_matches} \n total number of predicted matches: {f_total_predicted_matches} \n total number of matches in groud truth file: {f_total_relevant_matches} \n \n precision: {f_precision} \n recall: {f_recall} \n f1 score: {f_f1_score}''')

    # output results to log file
    dir_name = "evaluation"
    Path(f'''{configuration['log']['path']}/{dir_name}''').mkdir(parents=True, exist_ok=True)
    logger = write_log(f'''{configuration['log']['path']}''', dir_name, dir_name)
    
    logger.info(f'''[RESULTS] Evaluation result of similarity list in file [{similarity_file}] :\n threshold: {f_seuil} \n top k records: {f_k} \n decimal places retaining for the similarity: {appr} : \n correct matches: {f_correct_matches} \n total number of predicted matches: {f_total_predicted_matches} \n total number of matches in groud truth file: {f_total_relevant_matches} \n \n precision: {f_precision} \n recall: {f_recall} \n f1 score: {f_f1_score}''')
    logger.info(s)
    # return precision, recall, f1_score
import numpy as np
import matplotlib
matplotlib.use('Agg') 
from matplotlib import pyplot as plt
from scipy import stats
from utils.write_log import write_log

def _distribution_analysis(prop, valus_list, graph_path, output_file_name):
    # getting access frequency data
    values = list(valus_list)

    # calculate Histogram
    plt.figure(figsize=(10, 5))
    if prop == "idx":
        count, bins, ignored = plt.hist(values, bins=30, density=True, alpha=0.6, color='blue', edgecolor='black')
    elif prop == "token":
        count, bins, ignored = plt.hist(values, bins=30, density=True, alpha=0.6, color='red', edgecolor='black')

    # calculate the fitted curve of the normal distribution
    mu, sigma = np.mean(values), np.std(values)  # the mean and standard deviation
    x = np.linspace(min(values), max(values), 1000)
    pdf = stats.norm.pdf(x, mu, sigma)  # the normal distribution curve
    # drawing a Normal Distribution Curve
    plt.plot(x, pdf, 'r-', label=f'Normal Fit\n mu={mu:.2f}, sigma={sigma:.2f}')

    plt.xlabel("Visit Frequency")
    plt.ylabel("Count")
    plt.title(f"Histogram of Visit Frequency with Normal Fit {prop}")
    plt.legend()
    plt.savefig(f"{graph_path}/distribution {prop}-{output_file_name}.png", dpi=300)

def random_walk_analysis(graph, output_file_name):
    exp_logger = write_log("pipeline/logging", "walks", "stat_random_walk")
    graph = graph.get_graph()
    # visualization
    vis_token = {}
    freq_token = 0
    vis_idx = {}
    freq_idx = 0
    vis_cid = {}
    freq_cid = 0
    freqs = {}

    i = 0
    i_idx = 0
    i_token = 0
    i_cid = 0
    never_visited=0
    most_visited_freq=0
    most_visited_type=""
    most_visited_num = 0
    m_pretraining = 0
    n_pretraining = 0
    m_dyntraining = 0
    n_dyntraining = 0
    for v in graph.vs:
        i = i + 1
        node_name = v["name"]
        freq = v["appearing_frequency"]
        node_type = v["type"]

        if node_type == 'idx':
            vis_idx[i_idx] = freq
            freq_idx = freq_idx + freq
            i_idx = i_idx + 1
        elif node_type == 'tt' or node_type == 'tn':
            vis_token[i_token] = freq
            freq_token = freq_token + freq
            i_token = i_token + 1
        elif node_type == "cid":
            vis_cid[i_cid] = freq
            freq_cid = freq_cid + freq
            i_cid = i_cid + 1

        if freq == 0: 
            never_visited = never_visited+1
        if freq > most_visited_freq:
            most_visited_freq = freq
            most_visited_type = node_type
            if node_type == 'idx':
                most_visited_num = i_idx
            elif node_type == 'cid': 
                most_visited_num = i_cid
            else:
                most_visited_num = i_token
        # if freq >= 2000:
        freqs[node_name]={"freq": freq, "type": node_type}
        
        if v["test_neighbors_freq"] and node_type == 'cid':
            test_neighbors_freq = v["test_neighbors_freq"]
            # exp_logger.info(f'''neighbors of cid {node_name}: {test_neighbors_freq}''')

            for neighbor_name in test_neighbors_freq:         
                if graph.vs.find(name=neighbor_name)['test_pretraining']:
                    n_pretraining = n_pretraining + test_neighbors_freq[neighbor_name]
                else:
                    n_dyntraining = n_dyntraining + test_neighbors_freq[neighbor_name]
                
                if v["test_pretraining"]:
                    m_pretraining = m_pretraining + test_neighbors_freq[neighbor_name]
                else: 
                    m_dyntraining = m_dyntraining + test_neighbors_freq[neighbor_name]
            
            exp_logger.info(f'''random walk for pretraining nodes and dyntraning nodes for cid {node_name}: {m_pretraining}, {n_pretraining}, {m_dyntraining}, {n_dyntraining}''')
                

    print(f"total nodes number {i+1}")
    print(f"average fraquency for idx: {(freq_idx/len(vis_idx))}")
    print(f"average fraquency for token: {(freq_token/len(vis_token))}")
    print(f"average fraquency for cid: {(freq_cid/len(vis_cid))}")
    print(f"average fraquency for all: {(freq_idx + freq_token + freq_cid)/i}")
    print(f"random walk for pretraining nodes and dyntraning nodes: {m_pretraining}, {n_pretraining}, {m_dyntraining}, {n_dyntraining}")
    print(f"nomber of nodes never visited : {never_visited}")
    print(f"the highest nomber of visit : {most_visited_freq}")
    print(f"type of node most visited : {most_visited_type}")
    exp_logger.info(f'''total nodes number {i+1}
                        average fraquency for idx: {(freq_idx/len(vis_idx))}
                        average fraquency for token: {(freq_token/len(vis_token))}
                        average fraquency for cid: {(freq_cid/len(vis_cid))}
                        average fraquency for all: {(freq_idx + freq_token)/(i+1)}
                        random walk for pretraining nodes and dyntraning nodes: {m_pretraining}, {n_pretraining}, {m_dyntraining}, {n_dyntraining}
                        nomber of nodes never visited : {never_visited}
                        info of most visited node: frequency-{most_visited_freq}, node-type-{most_visited_type}, number-{most_visited_num}
                        ''')
    # exp_logger.info('''nodes whose frenquency is above ():''')
    # for node_name in freqs:
    #     node_type = freqs[node_name]["type"]
    #     node_freq = freqs[node_name]["freq"]
    #     exp_logger.info(f"# node name: {node_name}, node type: {node_type}, node frequency: {node_freq}")

    graph_path = "pipeline/stat"
    print(graph_path)
    # clipped_vis_idx = {k: min(v, 2000) for k, v in vis_idx.items()}
    # clipped_vis_token = {k: min(v, 2000) for k, v in vis_token.items()}
    clipped_vis_idx = {k:v for k, v in vis_idx.items()}
    clipped_vis_token = {k:v for k, v in vis_token.items()}
    _distribution_analysis('idx', vis_idx.values(), graph_path, output_file_name)
    _distribution_analysis('token', vis_token.values(), graph_path, output_file_name)

    plt.figure() # plt.figure(figsize=(10, 5))
    plt.bar(clipped_vis_idx.keys(), clipped_vis_idx.values(), width=1.0 ,color='blue')
    plt.xlabel("nodes")
    plt.ylabel("visit frequency")
    plt.title("distribution of IDX nodes visited by random walk")
    plt.legend()
    plt.savefig(f"{graph_path}/idx-{output_file_name}.png", dpi=300) 

    plt.figure()
    plt.bar(clipped_vis_token.keys(), clipped_vis_token.values(), width=1.0, color='red')
    plt.xlabel("nodes")
    plt.ylabel("visit frequency")
    plt.title("distribution of TOKEN nodes visited by random walk")
    plt.legend()
    plt.savefig(f"{graph_path}/token-{output_file_name}.png", dpi=300) 

    # plt.show(block=False)
    plt.close()

import random
from tqdm import tqdm
import traceback

from utils.utils import *
from utils.write_log import write_log

app_debug = write_log("pipeline/logging", "debug", "random_walk")
walk_info = write_log("pipeline/logging", "walks", "random_walk_record")
class RamdomRow:
    def __init__(
        self,
        graph,  
        row_id_index,
        sentence_len
    ):
        i_graph = graph.get_graph()
        self.walk = []
        row_id = i_graph.vs[row_id_index]['name']
        sampler = graph.get_sampler(row_id_index)
        app_debug.info(f"neighbors of {row_id}: {i_graph.neighbors(row_id_index, mode='OUT') }")
        while len(self.walk) < sentence_len:
            app_debug.info(f'{row_id}, {row_id_index}')
            node_index = sampler.sample()
            if node_index is None:
                raise ValueError(f'No neighbors')
            
            node = i_graph.vs[node_index]['name']
            self.walk.append(node)
            if len(self.walk) < sentence_len:
                self.walk.append(row_id)

    def get_walk(self):
        return self.walk

    def get_reversed_walk(self):
        return self.walk[::-1]
        
class RandomWalk:
    def __init__(
        self,
        graph,
        starting_node_index,
        sentence_len,
        backtrack
    ):
        i_graph = graph.get_graph()
        self.walk = []
        
        # find node and its neighbors
        starting_node = i_graph.vs[starting_node_index]
        starting_node_name = starting_node['name']
        # first step 
        if starting_node['node_class']['isfirst']:
            self.walk = [starting_node_name]
        else:
            try:
                sampler = graph.get_sampler(starting_node_index)
                first_node_indice = sampler.sample_firstnode()
                # first_node_indice = starting_node['sampler'].sample_firstnode()
                if first_node_indice is not None:
                    first_node_name = i_graph.vs[first_node_indice]['name']
                    self.walk = [first_node_name, starting_node_name]
                else:
                    raise ValueError(f"The first node of the sentence could not be found. Please check your node_types settings.")
            except Exception:
                print(f"The first node of the sentence could not be found. Please check node {starting_node_name}.")
                app_debug.error(f"The first node of the sentence could not be found. Please check node {starting_node_name}, index {starting_node_index}.")
                   
        if self.walk != []:
            current_node_indice = starting_node_index
            current_node_name = starting_node_name
            current_node = starting_node
            sentence_step = len(self.walk)
        else:
            return

        # the next steps
        while sentence_step < sentence_len:
            previous_node = current_node
            previous_node_index = current_node_indice
            sampler = graph.get_sampler(previous_node_index)
            current_node_indice = sampler.sample()
            # current_node_indice = current_node['sampler'].sample()
            if current_node_indice is None:
                raise ValueError(f'No neighbors')
            current_node = i_graph.vs[current_node_indice]
            current_node_name = current_node['name'] 

            if not backtrack and current_node_name == self.walk[-1]:
                continue
            if not current_node["node_class"]["isappear"]:
                continue
          
            self.walk.append(current_node_name)
            previous_node['appearing_frequency'] = previous_node['appearing_frequency'] + 1
            if current_node_name in previous_node["test_neighbors_freq"]:
                previous_node["test_neighbors_freq"][current_node_name] = previous_node["test_neighbors_freq"][current_node_name] + 1
            else:
                previous_node["test_neighbors_freq"][current_node_name] = 1
            sentence_step += 1

    def get_walk(self):
        return self.walk

    def get_reversed_walk(self):
        return self.walk[::-1]

class RandomWalk_MetaPath:
    def __init__(
        self,
        graph,
        starting_node_index,
        sentence_len,
        meta_path
    ):
        i_graph = graph.get_graph()
        self.walk = []
        # find node and its neighbors
        current_node = i_graph.vs[starting_node_index]
        current_node_indice = starting_node_index
        self.walk.append(current_node['name'])
        # init for the second node
        meta_index = 1 
        
        # the next steps
        sentence_step = len(self.walk)
        while sentence_step < sentence_len:
            try:
                next_type = meta_path[meta_index % len(meta_path)]
                sampler = graph.get_sampler(current_node_indice)
                next_node_indice = sampler[next_type].sample()
                next_node = i_graph.vs[next_node_indice]
                self.walk.append(next_node['name'])
                current_node = next_node
                current_node_indice = next_node_indice
            except Exception:
                    print(f"NO neighbors found for node: {current_node_indice}, name: {current_node['name']}. Looking for node type: {next_type}")
                    app_debug.error(f"NO neighbors found for node: {current_node}, name: {current_node['name']}. Looking for node type: {next_type}")
                    break
            sentence_step += 1
            meta_index += 1

    def get_walk(self):
        return self.walk

    def get_reversed_walk(self):
        return self.walk[::-1]

def start_walk_multiscale(roots_index, graph, walks_number, walk_length, write_walks, walk_rules):
    ''' 
    random walk with different walk length
    '''
    sentences = []
    sentence_counter = 0
    if roots_index == 0 or roots_index is None:
        return
    wl = [15, 30, 60]
    dis = [2, 2, 1]
    for wl_i in range(len(wl)):
        wn = int(walks_number * (dis[wl_i]/sum(dis)))
        pbar = tqdm(desc="# Sentence generation progress: ", total=len(roots_index)*wn)
        for root in roots_index:
            # random walk for each node
            walks = []
            for _r in range(wn):
                try: 
                    if isinstance(walk_rules, bool):
                        w = RandomWalk(
                            graph,
                            root,
                            wl[wl_i],
                            walk_rules
                        )
                    else: 
                        w = RandomWalk_MetaPath(
                            graph,
                            root,
                            wl[wl_i],
                            walk_rules
                        )
                except Exception as e:
                    print("node: ", _r)
                    print(e)
                    print(traceback.print_exc())
                    break
                
                if w.get_walk() != []:
                    walks.append(w.get_walk())
                else:
                    raise ValueError(f"random walk anormal")

            if write_walks:
                if len(walks) > 0:
                    ws = [" ".join(_) for _ in walks]
                    s = "\n".join(ws) + "\n"
                    walk_info.info(s)
                else:
                    pass
            sentences += walks
            sentence_counter += wn

            pbar.update(wn)
    pbar.close()
    return sentences

def start_walk(roots_index, graph, walks_number, walk_length, write_walks, walk_rules, row):
    ''' 
    random walk with fixed walk length
    '''
    sentences = []
    sentence_counter = 0
    if roots_index == 0 or roots_index is None:
        return
   
    pbar = tqdm(desc="# Sentence generation progress: ", total=len(roots_index)*walks_number)

    for root in roots_index:
        ######## random walk for each node
        walks = []

        root_id = graph.get_graph().vs[root]['name']
        if row:
            walks_number_basic = int(walks_number * 0.2)
            walks_number_row = int(walks_number - walks_number_basic)
            if root_id.startswith("idx"):
                for _r in range(walks_number_row):
                    w = RamdomRow(
                        graph,
                        root,
                        walk_length
                    )
                    # app_debug.info(f'for token {root}')
                    if w.get_walk() != []:
                        walks.append(w.get_walk())
                        app_debug.info(f'walks of token {w.get_walk()}')
            else:
                for _r in range(walks_number_basic):
                    w = RandomWalk(
                        graph,
                        root,
                        walk_length,
                        walk_rules
                    )
                    # app_debug.info(f'for token {root}')
                    if w.get_walk() != []:
                        walks.append(w.get_walk())
                        app_debug.info(f'walks of token {w.get_walk()}')
        else:
            for _r in range(walks_number):
                try: 
                    
                    if isinstance(walk_rules, bool):
                        w = RandomWalk(
                            graph,
                            root,
                            walk_length,
                            walk_rules
                        )
                    else: 
                        w = RandomWalk_MetaPath(
                            graph,
                            root,
                            walk_length,
                            walk_rules
                        )
                except Exception as e:
                    print("node: ", _r)
                    print(e)
                    print(traceback.print_exc())
                    break
                
                if w.get_walk() != []:
                    walks.append(w.get_walk())
                else:
                    if not row:
                        raise ValueError(f"random walk anormal")
                    else:
                        pass

        if write_walks:
            if len(walks) > 0:
                ws = [" ".join(_) for _ in walks]
                s = "\n".join(ws) + "\n"
                walk_info.info(s)
            else:
                pass
        sentences += walks
        sentence_counter += walks_number

        pbar.update(walks_number)
    pbar.close()
    return sentences

def dynrandom_walks_generation(configuration, graph, walk_nums):
    """
    Traverse the graph using different random walks strategies.
    :param configuration: run parameters to be used during the generation
    :param graph: graph generated starting from the input dataframe
    :return: the collection of random walks
    """
    
    walk_length = int(configuration['walks']['walk_length'])
    backtrack = configuration['walks']['backtrack']
    meta_path = configuration['graph']['meta_path']
    write_walks = configuration['walks']['write_walks']

    if walk_nums > 0:
        ############ Random walks ############
        sentences = []
        if not meta_path:
            roots_index = graph.dyn_roots
            # sentences = start_walk(roots_index, graph, walk_nums, walk_length, write_walks, backtrack, row=False)
            # for test
            sentences = start_walk(roots_index, graph, walk_nums, walk_length, write_walks, backtrack, row=False)
            graph.dyn_roots.clear()
        else:
            if isinstance(meta_path, list):
                if isinstance(meta_path[0], list):
                    for path in meta_path:
                        roots_index = graph.dyn_roots[path[0]]
                        sentences += start_walk(roots_index, graph, walk_nums, walk_length, write_walks, path, row=False)
                        graph.dyn_roots[path[0]].clear()
                else:
                    roots_index = graph.dyn_roots[meta_path[0]]
                    sentences = start_walk(roots_index, graph, walk_nums, walk_length, write_walks, meta_path, row=False)
                    graph.dyn_roots[meta_path[0]].clear()
    return sentences
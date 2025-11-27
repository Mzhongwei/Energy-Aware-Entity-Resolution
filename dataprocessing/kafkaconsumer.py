import json
import os
import queue
import signal
import socket
import sys
from threading import Timer
from datetime import datetime
import threading
import pandas as pd
import time
from collections import deque
import traceback

from prometheus_client import start_http_server
from tqdm import tqdm
from comparison_approaches.exact_matching_copy import preprocessing_incremental
from dataprocessing.similaritygraph import SimilarityGraph
from dynamic_embedding.dynamic_sampling import dynrandom_walks_generation
from dynamic_embedding.dynamic_entity_resolution import IdxMatrix, dynentity_resolution, FaissIndex, filter_result
from dataprocessing.similaritylist import SimilarityList
from dataprocessing.metrics import Metrics
from dataprocessing.random_walk_analysis import random_walk_analysis
from utils.write_log import write_log
from utils.utils import TIME_FORMAT
from confluent_kafka import Consumer, KafkaException, KafkaError

task_queue = queue.Queue()
consumer = None
class ConsumerService:
    def __init__(self, configuration, graph, model, output_file_name, metrics):
        self.consumer = None
        self.config = configuration
        self.graph = graph
        self.model = model
        self.strategy_suppl = configuration["similarity_list"]["strategy_suppl"]
        self.strategy_model = None
        self.output_file_name = output_file_name
        
        self.window_data = deque()
        self.last_update_time = time.time()
        self.data_buffer = []
        self.flag_running = False
        self.write_timer = None
        self.time_interval = 120 # 300 seconds = 5 minutes
        self.app_logger = None
        self.kafka_logger = None
        self.debug_logger = None
        self.metrics = metrics
        
        self.timeout = 20 # (s)
        self.timer = None
        self.t_start_time = None
        self.t_end_time = None

        if configuration["similarity_list"]["sim_structure"] == "list":
            self.sim_list = SimilarityList(
                            configuration["similarity_list"]["simlist_n"],
                            configuration["similarity_list"]["output_format"]
                        )
        elif configuration["similarity_list"]["sim_structure"] == "graph":
            self.sim_list = SimilarityGraph(
                            configuration["similarity_list"]["simlist_n"],
                            configuration["similarity_list"]["output_format"]
                            )
        self._setup_logging()
    def _setup_logging(self):
        self.app_logger = write_log(self.config["log"]["path"], "app", "app")
        self.kafka_logger = write_log(self.config["log"]["path"], "kafka", "kafka")
        self.debug_logger = write_log(self.config["log"]["path"], "debug", "debug")

        self.sim_list.set_logger(self.app_logger)

    def trigger_file_write(self):
        self.sim_list.check_output_path(self.output_file_name)
        if self.flag_running:
            self.write_timer = Timer(self.time_interval, self.trigger_file_write)  
            self.write_timer.daemon = True  # Make it a daemon thread
            self.write_timer.start()
        self.sim_list.update_file()

    def _remove_expired_data(self, current_time):
        """Remove data older than the window size"""
        while self.window_data and (current_time - self.window_data[0]["timestamp"] > self.config["window_time"]):
            self.window_data.popleft()

    def _prepare_data(self, window_data):
        # construct data structucre
        data_list = list(window_data)
        df = pd.DataFrame.from_records(data_list)
        return df
    
    def build_sim_graph(self):
        """
        最后一轮：对模型中所有 'idx__*' 一次性批量计算 Top-K，
        1) 写入 self.sim_list（按 sim_structure 处理）
        2) 同时生成 records 列表并返回
        非最后一轮：不计算，直接返回空列表
        """
        records = []

        try:
            # === 1) 全量取 'idx__*' + 归一化 ===
            keys, E = IdxMatrix.build_idx_matrix(self.model.wv, prefix="idx__")
            if not keys:
                self.app_logger.warning("No 'idx__*' keys found in model; skip building similarity graph.")
                return records

            # === 2) 一次性 Top-K（精确余弦=点积），内存预算可配 ===
            topk = int(self.config["similarity_list"]["top_k"])
            cal_k = topk*2
            budget_mb = 512
            D, I = IdxMatrix.topk_all_cosine(E, k=cal_k, budget_mb=budget_mb)
            out_k = min(cal_k, max(0, len(keys) - 1))

            # === 3) 写图 + 生成 records ===
            for t_idx, target in enumerate(keys):
                # 已按相似度降序
                neighs = [(keys[j], float(D[t_idx, r])) for r, j in enumerate(I[t_idx, :out_k])]

                # 过滤（优先用你类里的 _filter_list，其次按 config['source_num'] 的跨域过滤）
                if neighs:
                    if hasattr(self, "_filter_list"):
                        neighs = self._filter_list(neighs)
                    elif int(self.config.get("source_num", 0)) > 0:
                        # 需确保 filter_result_from_config 在同一模块中（我们之前给过实现）
                        neighs = filter_result(target, neighs, source_num=int(self.source_num))[:topk]

                if not neighs:
                    continue

                # 写入图：target -> neighbors
                self.sim_list.add_similarity(target, neighs)

                # 生成 records
                for rank, (nid, sim) in enumerate(neighs[:topk], start=1):
                    records.append({
                        "round": self.count,
                        "target_id": target,
                        "neighbor_rank": rank,
                        "neighbor_id": nid,
                        "similarity": sim
                    })

            print(f"[Final] Built similarity graph for {len(keys)} nodes and generated {len(records)} records.\n")
            df_new = pd.DataFrame(records)
            output_file = f"pipeline/stat/sim_changement-{self.output_file_name}.csv"
            if not os.path.exists(output_file):
                df_new.to_csv(output_file, index=False)
            else:
                df_new.to_csv(output_file, mode="a", header=False, index=False)
            self.count += 1

            print(f"{datetime.now()}: Top-{topk} similar data for {len(records)} records are exported to {output_file}")  

        except Exception as e:
            self.app_logger.error(f"Error similarity building (final batch): {str(e)}")
            import traceback
            self.debug_logger.error(traceback.format_exc())
            print(f"Error similarity building (final batch): {str(e)}")
            return records

    # def build_matching_list(self, df, last_win):
    #     '''build sim list and output to db file'''
    #     # get similar words 
    #     for target in tqdm(df.loc[:,"rid"], desc= "# build similarity list. "):
    #         # print(f"[ExecTime] probabilistic comparison starts ....................{time_start.strftime(TIME_FORMAT)}")
    #         try:
    #             if self.strategy_suppl == "basic":
    #                 similar = dynentity_resolution(self.model, target, self.config["similarity_list"]["top_k"])
    #             elif self.strategy_suppl == "faiss":
    #                 similar = self.strategy_model.get_similar_words([self.model.wv[target]], target, self.config["similarity_list"]["top_k"])

    #             if int(self.config['source_num']) > 0 and similar != []:
    #                 similar = self._filter_list(similar)

    #             if similar != [] and similar is not None:
    #                 self.sim_list.add_similarity(target, similar)

    #                 # if self.sim_list.output_format == "db":
    #                 #     self.sim_list.insert_data(target)
    #                 if self.config["similarity_list"]["sim_structure"] == "list":
    #                     for word, score in similar:
    #                         self.sim_list.add_similarity(word, [(target, score)])
    #                         # if self.sim_list.output_format == "db":
    #                         #     self.sim_list.insert_data(word)
    #                 if self.config["similarity_list"]["sim_structure"] == "graph" and self.sim_list.output_format == "db":
    #                     for word, score in similar:
    #                         self.sim_list.insert_data(word)
    #             else:
    #                 pass
    #             time_end_1 = datetime.now()
    #             # print(f"[ExecTime] building list/graph starts ....................{time_end_1.strftime(TIME_FORMAT)}")
    #             # print(f"[ExecTime] building list/graph time-------------------{time_end_1 - time_end}")
    #         except Exception as e:
    #             self.app_logger.error(f"Error similarity building: {str(e)}")
    #             self.debug_logger.error(traceback.print_exc())
    #             print(f"Error similarity building: {str(e)}")

    #     if not last_win:
    #         print(f"Waiting for the next data window...")
    #     print()

    def _record_similarity_changement(self):
        records = []
        topk_remain = self.config["similarity_list"]["top_k"]

        # get 'idx__*' and normalization
        keys, E = IdxMatrix.build_idx_matrix(self.model.wv, prefix="idx__")
        # calculate Top-K (Adjustable memory budget：budget_mb)
        D, I = IdxMatrix.topk_all_cosine(E, k=topk_remain*2, budget_mb=512)

        # get records；按需选择过滤器：用 source_num / 用 config
        records = IdxMatrix.to_records(
            keys, D, I,
            round_cnt=self.count,
            topk_remain=topk_remain,
            filter_fn=(lambda t, sims: filter_result(t, sims, source_num=int(self.source_num)))
        )

        df_new = pd.DataFrame(records)
        output_file = f"pipeline/stat/sim_changement-{self.output_file_name}.csv"
        if not os.path.exists(output_file):
            df_new.to_csv(output_file, index=False)
        else:
            df_new.to_csv(output_file, mode="a", header=False, index=False)
        self.count += 1

        print(f"{datetime.now()}: Top-{topk_remain} similar data for {len(records)} records are exported to {output_file}")

        match_rows, match_cols, match_scores = IdxMatrix.pipeline_ratio_rnn_triangle_one2one(
            E,
            ratio=None,            # 比值检验阈值
            delta=0.1,           # 若想更稳，可加差值阈值（例如 0.05）
            enforce_rnn=True,     # 互为最近邻
            triangle_alpha=0.9,   # 三角一致性强度
            triangle_undirected=True,
            triangle_max_deg=100, # 控复杂度的度截断
        )

        pairs = [(keys[i], keys[j], float(s)) for i, j, s in zip(match_rows, match_cols, match_scores)]
        for A, B, score in pairs:
            target = A
            similarity_list = [(B, score)]
            sim_list =  filter_result(target, similarity_list, int(self.source_num), "idx__")
            self.debug_logger.info(f'target, : {target}, sim list: {sim_list}')
            self.sim_list.add_similarity(target, sim_list)

    def process_window_data(self):
        """Process data of the current window """
        print("processing window data...")
        # self.timer.cancel()
        
        # data preparation
        df = self._prepare_data(self.window_data)
        # preprocessing exact matching for increments
        time_start = datetime.now()
        print(f"[ExecTime] preprocessing starts ....................{time_start.strftime(TIME_FORMAT)}")
        df = self._em_inc(df)
        # print(df)
        time_end = datetime.now()
        print(f"[ExecTime] preprocessing ends ....................{time_end.strftime(TIME_FORMAT)}")
        print(f"[ExecTime] preprcessing time-------------------{time_end - time_start}")
        # print(df)
        if not df.empty:
            # add new node to outputfile
            time_start = datetime.now()
            print(f"[ExecTime] graph construction starts ....................{time_start.strftime(TIME_FORMAT)}")
            self.graph.build_relation(df)
            time_end = datetime.now()
            print(f"[ExecTime] graph construction ends ....................{time_end.strftime(TIME_FORMAT)}")
            print(f"[ExecTime] graph construction time---------------{time_end - time_start}")
            print(f"# roots numbers: {len(self.graph.dyn_roots)}")

            # start random walk for new data
            time_start = datetime.now()
            print(f"[ExecTime] random walk starts ....................{time_start.strftime(TIME_FORMAT)}")
            
            walks_number = self.config['walks']['walks_number']
            # dyn_roots = self.graph.dyn_roots
            # walks_number = walks_number % 3
            # for wn in range(3):
            #     walks = dynrandom_walks_generation(self.config, self.graph, walk_nums=walks_number)
            #     self.graph.dyn_roots = dyn_roots
            walks = dynrandom_walks_generation(self.config, self.graph, walk_nums=walks_number)
            time_end = datetime.now()
            print(f"[ExecTime] random walk ends ....................{time_end.strftime(TIME_FORMAT)}")
            print(f"[ExecTime] random walk time---------------------{time_end - time_start}")

            if walks == []:
                print("[Warning] Random walk do not executed ...")
                raise ValueError(f"Random walk anomaly")

            try:
                print("# Retraining embeddings model by window data...")
                if len(self.model.wv.key_to_index) == 0:
                    # initiate model
                    time_start = datetime.now()
                    print(f"[ExecTime] retraining embedding model starts ....................{time_start.strftime(TIME_FORMAT)}")
                    self.model.build_vocab(walks)
                    self.model.train(walks, total_examples=self.model.corpus_count, epochs=self.config["embeddings"]["inc_epochs"])
                    if self.strategy_suppl == "faiss":
                        self.strategy_model = FaissIndex(self.model)
                    time_end = datetime.now()
                    print(f"[ExecTime] retraining embedding model ends....................{time_end.strftime(TIME_FORMAT)}")
                    print(f"[ExecTime] retraining embedding model time----------------------{time_end - time_start}")
                else:
                    # Update model
                    time_start = datetime.now()
                    print(f"[ExecTime] retraining embedding model starts ....................{time_start.strftime(TIME_FORMAT)}")
                    self.model.build_vocab(walks, update=True)
                    self.model.train(walks, total_examples=len(walks), epochs=self.config['embeddings']['inc_epochs']) # An epoch is one complete pass through the entire training data.
                    if self.strategy_suppl == "faiss":
                        if self.strategy_model == None:
                            self.strategy_model = FaissIndex(self.model)
                        else:
                            self.strategy_model.rebuild_index(self.model, self.graph.get_id_nums())
                            # self.strategy_model.update_index(self.model)
                    time_end = datetime.now()
                    print(f"[ExecTime] retraining embedding model ends....................{time_end.strftime(TIME_FORMAT)}")
                    print(f"[ExecTime] retraining embedding model time----------------------{time_end - time_start}")
            except Exception as e:
                print("[ERROR]: ", e)
                self.app_logger.error(f"[ERROR]: {str(e)}")
        return df

    def _reset_timer(self):
        if self.timer:
            self.timer.cancel()
        self.timer = Timer(self.timeout, self._timerout_function)
        self.timer.start()

    def _timerout_function(self):
        print("Last window of this experiment")
        self.debug_logger.info("### last window of this experiment... ")

        if len(self.window_data) != 0:
            self.sim_list.check_output_path(self.output_file_name)
            df = self.process_window_data()
            if not df.empty:
                time_start = datetime.now()
                print(f"[ExecTime] sim structure building starts................{time_start.strftime(TIME_FORMAT)}")
                # self.build_matching_list(df, True)
                # self.build_sim_graph()
                self._record_similarity_changement()
                time_end = datetime.now()
                print(f"[ExecTime] sim structure building ends.................{time_end.strftime(TIME_FORMAT)}")
                print(f"[ExecTime] sim structure building time---------------------{time_end - time_start}")
            self.window_data.clear()
            self.t_end_time = time.time()
            print(f'[Finished] the test finished, output file name: {self.output_file_name}, execution time(s): {self.t_end_time - self.t_start_time}')
            self.app_logger.info(f'[Finished] the test finished, output file name: {self.output_file_name}, execution time (s): {round(self.t_end_time - self.t_start_time - self.timeout)}')

            print(f"Saving model in binary format... Embedding file: pipeline/embeddings/{self.output_file_name}.embin")
            self.model.save(f"pipeline/embeddings/{self.output_file_name}.embin") # Saves in binary format by default
            # self.model.wv.save_word2vec_format(f"pipeline/embeddings/{self.output_file_name}.emb", binary=False) # Saves in binary format by default
            print("Model saved!")

            self.sim_list.update_file()
            print(f'output similarity relationship in file')

            if self.config["walks"]["rw_stat"] == True:
                print(f"Drawing... Distribution of random walk visit values: {self.output_file_name}")
                random_walk_analysis(self.graph, self.output_file_name)
                print(f"Display statistical results in graphs. The folder for statiqtical results: pipeline/stat/")

            print(f"Saving graph with attributes... Graph file path: pipeline/graph/graph-{self.output_file_name}.graphml")
            g = self.graph.clean_attributes()
            g.write_graphml(f"pipeline/graph/graph-{self.output_file_name}.graphml")
            print("Graph saved!")
            print("Process over!!!!!!!!!!!!!!!! The program will be closed automatically in few secondes")
            # print("Press 'ctrl+C' to stop listening of kafka consumer")

            

    # def _filter_list(self, similarity_list):
    #     result = []
    #     if similarity_list is not None and similarity_list != []:
    #         for t in similarity_list:
    #             if int(float(t[0].split('__')[1])) <= int(self.config['source_num']):
    #                 result.append(t)
    #                 # print(t)
    #     return result
    
    # def _filter_result(self, target, similarity_list):
    #     result = []
    #     if similarity_list is not None and similarity_list != []:
    #         if int(float(target.split('__')[1])) <= int(self.config['source_num']):
    #             for t in similarity_list:
    #                 if int(float(t[0].split('__')[1])) > int(self.config['source_num']):
    #                     result.append(t)
    #         else:
    #             for t in similarity_list:
    #                 if int(float(t[0].split('__')[1])) <= int(self.config['source_num']):
    #                     result.append(t)

    #     return result



    def run(self):
        try:
            # prepare kafka consumer
            self.consumer = Consumer({
                'bootstrap.servers': f'{self.config["kafka"]["bootstrap_servers"]}:{self.config["kafka"]["port"]}',
                'group.id': self.config['kafka']["groupid"],
                'auto.offset.reset': 'latest',   # latest / earliest
                'enable.auto.commit': False
            })

            # subscribe a topic
            self.consumer.subscribe([self.config['kafka']['topicid']])

        except Exception as e:
            self.app_logger.error(f"Fatal error in consumer service: {str(e)}")
            print(f"Fatal error in consumer service: {str(e)}")
            return

        self.app_logger.info("Start Kafka consumer...")

        empty_poll_count = 0
        max_empty_polls = 30  # 最多空转30次 相当于约 30*1 秒内没消息就退出（poll 每次 100ms）
        poll_timeout = 1.0  # 1s
        last_poll_time = time.time()


        while True:
            msg = self.consumer.poll(poll_timeout)  # Non-blocking batch pull
            now = time.time()
            print(f"[DEBUG] time since last poll: {now - last_poll_time:.2f}s")
            last_poll_time = now

            if msg is None:  # no new message
                empty_poll_count += 1
                if empty_poll_count >= max_empty_polls:
                    print("[INFO] No new messages for a while. Exiting consumer loop.")
                    break
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    # end of partition
                    continue
                else:
                    raise KafkaException(msg.error())
            empty_poll_count = 0  # reset counter
            
            # get message
            metadata = json.loads(msg.value().decode('utf-8'))
            if self.t_start_time is None:
                self.app_logger.info("[STARTED] Receiving records...")
                print("[STARTED] Receiving records...")
                self.t_start_time = time.time()
                # sim_changement
                self.source_num=self.graph.get_id_nums()
                self.count=0
            try:
                # Update lag
                try:
                    low, high = self.consumer.get_watermark_offsets(msg.topic(), msg.partition())
                    lag = high - msg.offset()
                    self.metrics.update_lag_metrics(lag)
                except Exception as lag_err:
                    self.app_logger.warning(f"[Lag fetch error]: {lag_err}")

                self.metrics.update_message_consumed()

                ## prepare data structure
                self.graph.accum_id_nums()
                id_num = self.graph.get_id_nums()

                metadata = msg.value
                metadata["rid"] = f"idx__{id_num}"
                current_time = time.time()

                self.data_buffer.append(metadata)

                ## Handle (time or count based) windowing
                if self.config["kafka"]["window_strategy"] == "time":
                    self._handle_time_window(metadata, current_time)
                elif self.config["kafka"]["window_strategy"] == "count":
                    self._handle_count_window(metadata)
                

            except Exception as e:
                self.app_logger.error(f"Error processing message: {str(e)}")
                traceback.print_exc()
                    
                    # self._reset_timer()

            # prevent tight loop
            time.sleep(0.01)  
            self.consumer.commit()
        self._timerout_function()  # 处理最后窗口
        self.consumer.close()
        print("[INFO] Consumer shutdown complete.")

    def _handle_time_window(self, metadata, current_time):
        metadata["timestamp"] = current_time
        self.window_data.append(metadata)

        if current_time - self.last_update_time >= self.config["kafka"]["update_frequency"]:
            self._process_window()
            self.last_update_time = current_time
        
            # clear window
            if self.config["kafka"]["update_frequency"] != 0:
                self._remove_expired_data(current_time)
            else: 
                self.window_data.clear()

    def _handle_count_window(self, metadata):
        self.window_data.append(metadata)

        if len(self.window_data) >= self.config["kafka"]["window_count"]:

            self._process_window()

            # clear window
            if self.config["kafka"]["update_frequency"] != 0:
                for _ in range(self.config["kafka"]["update_frequency"]):
                    if self.window_data:
                        self.window_data.popleft()
            else:
                self.window_data.clear()

    def _process_window(self):
        '''
        update metric for window data processing time 
        '''        
        start = time.time()
        df = self.process_window_data()
        if not df.empty:
            # self.build_matching_list(df, False)
            self._record_similarity_changement()
        end = time.time()
        self.metrics.update_window_data_processing_time((end - start) / len(self.window_data))

    def _extract_candidates_from_graph(self, graph) -> dict:
        """
        从图中提取所有 type='idx' 的节点，并获取其邻居的 name 字段
        """
        candidats = {}
        for v in graph.vs:
            if v.attributes().get("type") == "idx" and "name" in v.attributes():
                rid = v["name"]
                candidats[rid] = self.graph.get_record(v)
        return candidats

    def _em_inc(self, df):
        print('preprocessing')
        graph = self.graph.get_graph()
        # rids = [v.index for v in graph.vs if v["type"] == "idx"]
        canididats = self._extract_candidates_from_graph(graph)
        # for rid_index in rids:
        #     canididats[graph.vs[rid_index]["name"]] = self.graph.get_record(rid_index)
        df, existing_nodes, new_nodes, new_nodes_index = preprocessing_incremental(df, canididats)
        if new_nodes:
            print("testing")

            for pair in new_nodes:
                for record_i in pair:
                    # if self.sim_list.output_format == "db":
                    #     self.sim_list.insert_data(record_i)
                    # print(self.sim_list.get_similarity_words_with_score(target, self.config["simlist_show"]))
                    for record_j in pair:
                        if record_i != record_j:
                            self.sim_list.add_similarity(record_j, [(record_i, int(1))])
                            if self.config["similarity_list"]["sim_structure"] != "graph":
                                self.sim_list.add_similarity(record_i, [(record_j, int(1))])
                                
        if existing_nodes:
            print("testing")               
            for id_in_graph in existing_nodes:
                em_ids = existing_nodes[id_in_graph]
                if type(em_ids) == str:
                    em_ids = [em_ids]

                if self.config["similarity_list"]["sim_structure"] == "graph":
                    sim = []
                    for em_id in em_ids:
                        sim.append((em_id, int(1)))
                    self.sim_list.add_similarity(id_in_graph, sim)

                else:
                    ### find old exact matches 
                    existing_matchs = []
                    heap_list = self.sim_list.get_similarity_words_with_score(id_in_graph, 10)
                    if heap_list:
                        for score, word in heap_list:
                            if int(score) == 1:
                                existing_matchs.append(word)
                    ### add new exact matches for target (1) and add existing matchs for new matchs (2)
                    for em_id in em_ids:
                        ## (1)
                        self.sim_list.add_similarity(id_in_graph, [(em_id, int(1))])
                        ## (2)
                        self.sim_list.add_similarity(em_id, [(id_in_graph, int(1))])
                        if existing_matchs:
                            for exm in existing_matchs:
                                self.sim_list.add_similarity(em_id, [(exm, int(1))])
                    #     if self.sim_list.output_format == "db":
                    #         self.sim_list.insert_data(em_id)
                    # if self.sim_list.output_format == "db":
                    #     self.sim_list.insert_data(id_in_graph)
                    
                    ### add new matchs for existing matchs
                    if existing_matchs:
                        for match in existing_matchs:
                            for em_id in em_ids:
                                self.sim_list.add_similarity(id_in_graph, [(em_id, int(1))])
                            # if self.sim_list.output_format == "db":
                            #     self.sim_list.insert_data(match)
                

        
            # for r in new_nodes_index:
            #     if self.sim_list.output_format == "db":
            #         self.sim_list.insert_data(record_i)
            #         self.sim_list.insert_data(record_j)
        return df
            

def _start_prometheus_port(port=8000):
    def is_port_free(port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("0.0.0.0", port))
                return True
            except OSError:
                return False

    if is_port_free(port):
        print(f"[INFO] Prometheus starting on port {port}...")
        start_http_server(port)
    else:
        print(f"[WARN] Port {port} is already in use. Prometheus server not started.")

def handle_sigint(signum, frame):
    global consumer
    print("Received SIGINT (Ctrl+C), shutting down...")
    if consumer:
        consumer.consumer.close()
        consumer = None
        print(f"-----------------Closing Kafka Consumer-------------------")
    print(f"-----------------Exiting program-------------------")
    sys.exit(0)

def start_kafka_consumer(configuration, graph, model, output_file_name):
    metrics = Metrics()
    # t = threading.Thread(target=_start_prometheus_port, daemon=True)
    # t.start()

    consumer = ConsumerService(configuration, graph, model, output_file_name, metrics)
    # check output path
    consumer.sim_list.check_output_path(output_file_name)
    # if consumer.sim_list.output_format != "db":
    # consumer.flag_running = True
    # consumer.trigger_file_write()
    

    signal.signal(signal.SIGINT, handle_sigint)
    consumer.run()
    
    # try:
    #     consumer.run()
    # except KeyboardInterrupt:
    #         if consumer:
    #             consumer.consumer.close()
    #             consumer = None
    #             print(f"-----------------Closing Kafka Consumer-------------------")
    #         print(f"-----------------Exiting program-------------------")
    #         sys.exit(0)

if __name__ == '__main__':
    # consumer_service(configuration, graph, wv)
    pass

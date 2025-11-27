import heapq
import json
import sqlite3
import os

import pandas as pd

class SimilarityList:
    def __init__(self, most_similar_num, output_format):
        self.similarity_dict = {}
        self.most_similar_num = most_similar_num
        self.output_format = output_format
        self.name = ""
        self.file_path = ""
        self.db_conn = None
        self.db_cursor = None
        self.app_logger = None
    
    def set_logger(self, logger):
        self.app_logger = logger

    def check_output_path(self, name):
      
        similarity_file_path = "pipeline/similarity"
        os.makedirs('pipeline/similarity', exist_ok=True)

        self.name = name

        if self.output_format == "db":
            print('connect to db ...')
            try:
                conn = sqlite3.connect(f"{similarity_file_path}/{self.name}.db")
                print(f"db output: {similarity_file_path}/{self.name}.db")
                self.db_conn = conn
                cursor = conn.cursor()
                print("connexion created!")
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS matchinglist (
                        id TEXT PRIMARY KEY,
                        similarity TEXT
                    )
                    """)
                print("cursor created!")
                self.app_logger.info("cursor created!")
                self.db_cursor = cursor
            except Exception:
                print(f"[ERROR] Database connexion failed: {Exception}")
                self.app_logger.error(f"[ERROR] Database connexion failed: {Exception}")
        else:
            if self.file_path == "":
                self.file_path = os.path.join(similarity_file_path, f'{self.name}.{self.output_format}')
                print(self.file_path)
            

    def insert_data(self, key):
        """
        write in a database 
        """
        # print('insert data.. ')
        if self.output_format=="db":
            try:
                if self.db_cursor is None:
                    self.check_output_path(self.name)
                value = self.similarity_dict.get(key)
                if value is not None and value != []:
                    self.db_cursor.execute(f"""
                        INSERT INTO matchinglist (id, similarity) VALUES (?, ?)
                        ON CONFLICT(id) DO UPDATE SET similarity = excluded.similarity;
                        """, (key, json.dumps(value)))
                    # print(f"""
                    #     INSERT INTO matchinglist (id, similarity) VALUES ({key}, {json.dumps(value)})
                    #     ON CONFLICT(id) DO UPDATE SET similarity = excluded.similarity;
                    #     """)
                    # print("[Great] data inserted!")
                    # self.app_logger.info("[Great] data inserted!")
                    self.db_conn.commit()
            except Exception as e:
                print(f"[ERROR] Insert data failed: {e}")
                self.app_logger.error(f"[ERROR] Insert data failed: {e}")

    def update_db(self):
        """
        update all records in a database 
        """
        # print('insert data.. ')
        if self.output_format=="db":
            try:
                if self.db_cursor is None:
                    self.check_output_path(self.name)

                for key in self. similarity_dict:
                    value = self.similarity_dict.get(key)
                    if value is not None and value != []:
                        self.db_cursor.execute(f"""
                            INSERT INTO matchinglist (id, similarity) VALUES (?, ?)
                            ON CONFLICT(id) DO UPDATE SET similarity = excluded.similarity;
                            """, (key, json.dumps(value)))
                    # print(f"""
                    #     INSERT INTO matchinglist (id, similarity) VALUES ({key}, {json.dumps(value)})
                    #     ON CONFLICT(id) DO UPDATE SET similarity = excluded.similarity;
                    #     """)
                    # print("[Great] data inserted!")
                    # self.app_logger.info("[Great] data inserted!")
                        self.db_conn.commit()
            except Exception as e:
                print(f"[ERROR] Insert data failed: {e}")
                self.app_logger.error(f"[ERROR] Insert data failed: {e}")

    def update_file(self):
        """
        write in a file 
        """
        print("update..")
        try:
            if self.file_path == "":
                self.check_output_path(self.name)
            if self.output_format=="parquet":
                # Convert dictionary to DataFrame
                df = pd.DataFrame(list(self.similarity_dict.items), columns=['key', 'value'])
                # Save as Parquet
                df["value"].apply(json.dumps)
                df.to_parquet(self.file_path, engine="pyarrow", index=False)
            elif self.output_format=="json":
                with open(self.file_path, "w") as f:
                    json.dump(self.similarity_dict, f)
            elif self.output_format == "db":
                self.update_db()
        except Exception:
            print("[ERROR]: can not update file ", Exception)
            self.app_logger.error("[ERROR]: can not update file ", Exception)


    def add_similarity(self, target, new_similarities):
        '''
        : param new_similarities: [(word, score)]
        '''
        # add to word and score to dictionary
        # If the target word already exists, take out the current similarity list, otherwise create an empty list
        
        existing_similarities = self.similarity_dict.get(target, [])
        existing_id = {id for _,id in existing_similarities}

        # Store the first n are the most similar
        for word, score in new_similarities:
            if word not in existing_id:
                heapq.heappush(existing_similarities, (score, word))
                # if number of candidates is greater than n, remove the one with the smallest sim score
                if len(existing_similarities) > self.most_similar_num:  
                    heapq.heappop(existing_similarities)

        self.similarity_dict[target] = existing_similarities

    def add_similarity_batch(self, key, value):
        self.similarity_dict[key] = value
    
    
    def get_similarity_words_with_score(self, word, n):
        """
        get the whole dictionary
        """
        heap_list = self.similarity_dict.get(word, [])
        
        return heapq.nlargest(n, heap_list)
    
    def get_similarity_words(self, word, n):
        """
        only get the similar words
        """
        heap_list = self.get_similarity_words_with_score(word, n)
        words_list = []
        for _, word in heap_list:
            words_list.append[word]
        return words_list

    def display(self):
        """
        export in terminal
        """
        for word, similar_words in self.similarity_dict.items():
            print(f"{word}: {', '.join(similar_words)}")
    
    def validate_similarity(self, word1, word2):
        """
        Verify that two words have common neighbors (similar words)
        """
        neighbors1 = set(self.similarity_dict.get(word1, []))
        neighbors2 = set(self.similarity_dict.get(word2, []))
        
        # Get the intersection of two word neighbors
        common_neighbors = neighbors1.intersection(neighbors2)
        
        # If there are common neighbors, it means that their neighborhoods are similar
        if common_neighbors:
            return True, common_neighbors
        else:
            return False, None
        
if __name__ == '__main__':
    sim = SimilarityList(3, "db")
    sim.add_similarity("apple", [("banana", 0.8), ("cherry", 0.9), ("grape", 0.85)])
    sim.add_similarity("apple", [("orange", 0.7), ("pear", 0.95), ("watermelon", 0.92)])

    print(sim.similarity_dict.get("apple"))  
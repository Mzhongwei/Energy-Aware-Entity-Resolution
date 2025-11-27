# ER for stream
This project enables **entity resolution (ER)** for streaming data. We use **embedding techniques** to represent the data in a relational table with vectors and computes similarities between records to identify matching entities.

## 🛠️ INSTALLATION
### Install dependencies
```bash
pip install -r requirements.txt 
pip install git+https://github.com/dpkp/kafka-python.git
```
### Usage
To run the application with a configuration file:
```bash
python main.py -f [config_file_path]
```
### Notes
> ⚠️ **Python Compatibility Notice**:  
> Python 3.13 is **not currently compatible** with the `gensim` package. It is **not recommended** to use Python 3.13 for this project.  

> ✅ **For streaming process**:  
> Make sure Kafka is properly configured and activated if running process with real-time data sources

## ⚙️ CONFIGURATION

There are 3 example configuration files provided in the `/config` directory.

Each configuration file contains several blocks:

- **mandatory**: Required settings needed to run the program
- **optional**: You can customize these values as needed; otherwise, the program will fall back to default settings (the default configuration values are attached at the end of the article).

> 📌 **Attention**:  
> - Please select a appropriate task type. `batch` for pre-training models, "smatch" for er on streaming data, `evaluation` for evaluating results;
> - Make sure to update the configuration file paths (e.g., dataset files, task type, etc.) according to your environment.

## 🧪 PIPELINES AND EVALUATION

### Pre-training Pipeline
This process constructs a graph according to a relational table formatted as a csv, then converts it into an embedding model. 
- Input file: dataset file, setted up with variable name `dataset_file`.
- Output files: graph file `.graphml` and embedding `.emb`, stored respectively in path `pipeline/graph` and `pipeline/embeddings`, which can be used in the streaming process as pre-training input variables. 

For testing pre-training process, pls run `python main.py -f Config_example/config-batch` 

### Streaming Pipeline
This process captures real-time data from kafka and performs entity resolution tasks incrementally. The incremental window can be count-based or time-based and is set in config file.
- Input file: 
    - No input files: runs without pre-training (not recommanded, low accurancy) 
    - With input files: uses both graph file `.graphml` and embedding file `.emb` generated from pre-training process
- Output file: similarity results and predictions stored as a `.db` file in the `pipeline/similarity` directory.

For testing streaming process, pls run `python main.py -f Config_example/config-stream` 

### Evaluation
We provide 3 metrics for the evaluation: precision, recall and f1-score. All evaluation results output to the terminal and simultaneously logged in `pipeline/evaluation/evaluation.log`.

For verifying the results of the experiment, pls run `python main.py -f Config_example/config-evaluation`

## ANNEXES
### default values list:
| VARIABLE NAME    | VALUE BY DEFAULT |
| -------- | ------- |
| log_path  | "pipeline/logging"  |
| output_file_name  | "test" |
| node_types | ["7#__tn", "7$__tt", "3$__idx", "1$__cid"] |
| flatten |  "no" | 
| smoothing_method|  "no" | 
| directed| False | 
| write_walks|  True | 
| walk_length|  60 | 
| walks_number|  20 | 
| backtrack|  False | 
| rw_stat| True | 
| learning_method|  "skipgram" | 
| window_size|  3 | 
| n_dimensions|  300 | 
| training_algorithm|  "word2vec" | 
| sampling_factor |  0.001 |
| top_k| 10 | 
| simlist_n| 10 | 
| simlist_show| 5 | 
| strategy_suppl| "basic" | 
| output_format| "db" | 
| kafka_topicid| "entity_resolution_process" | 
| kafka_groupid| "er_group" | 
| bootstrap_servers| "localhost" | 
| port| "9092" | 
| window_strategy| "count" | 
| window_count| 50 | 
| update_frequency| 0| 
| n_first|  3 | 
| approximate|  16 | 
| sim_vis|  True | 

### node types
- "tn": token as number, "tt": token as text, "idx": id of row, "cid": id of column
- "$": string, "#": number
- (1-7): class of this type of value, definition of different classes is written at file README
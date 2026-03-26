import os
import json

from models import SimilarityGraph, CGIndex, RepresentationGraph, EmbeddingModel
from models.bert_model import Model

class StateManager:

    def __init__(self):
        self.cache = {}

    def get(self, key):
        return self.cache.get(key)

    def update(self, key, value):
        self.cache[key] = value

    def save(self, config, stages):
        state_config = config.get("state_management", {})

        def _has_stage(*names):
            return any(name in stages for name in names)

        def _artifact_path(dir_key, name_key, default_dir, default_ext):
            save_dir = state_config.get(dir_key, default_dir)
            # personalize file name or use version name 
            version_name = config.get("version_name", "test")
            name = state_config.get(name_key) or version_name
            os.makedirs(save_dir, exist_ok=True)
            return os.path.join(save_dir, f"{name}{default_ext}")

        def _write_text(path, value):
            with open(path, "w", encoding="utf-8") as f:
                if isinstance(value, str):
                    f.write(value)
                elif isinstance(value, (list, tuple, set)):
                    f.write("\n".join(map(str, value)))
                elif isinstance(value, dict):
                    json.dump(value, f, ensure_ascii=False, indent=2)
                else:
                    f.write(str(value))

        if _has_stage("bert_training") and "bert_model" in self.cache:
            save_dir = state_config.get("bert-dir", "data/bert_model")
            version_name = config.get('version_name', "test")
            model_dir = os.path.join(save_dir, version_name)
            os.makedirs(model_dir, exist_ok=True)
            bert_mode = self.cache["bert_model"]
            bert_mode["trainer"].save_model(model_dir)
            bert_mode["tokenizer"].save_pretrained(model_dir)

        if _has_stage("graph_construction") and "representation_graph" in self.cache:
            graph_path = _artifact_path("graph-dir", "graph-name", "data/graph", ".graphml")
            graph = self.cache["representation_graph"]
            graph_obj = getattr(graph, "graph", graph)
            if isinstance(graph, RepresentationGraph):
                graph_save = graph.clean_attributes()
                graph_save.write_graphml(graph_path)
            else:
                _write_text(graph_path, graph)

        if _has_stage("embedding_training") and "embedding_model" in self.cache:
            emb_path = _artifact_path("embedding-dir", "embedding_model-name", "data/embedding", ".emb")
            model = self.cache["embedding_model"]
            if hasattr(model, "save"):
                model.save(emb_path)
            else:
                _write_text(emb_path, model)

        if _has_stage("feature_index_construction") and "cg_feature_index" in self.cache:
            index_dir = state_config.get("feature_index-dir", "data/index")
            index_name = state_config.get("feature_index-name") or config.get("version_name", "test")
            save_dir = os.path.join(index_dir, index_name)
            os.makedirs(save_dir, exist_ok=True)
            index = self.cache["cg_feature_index"]
            if isinstance(index, CGIndex):
                if hasattr(index, "index_dir"):
                    index.index_dir = save_dir
                index.persist()
            else:
                _write_text(os.path.join(save_dir, "index.json"), index)

        if _has_stage("decision_making", "bert_inference"):
            predicted = self.cache.get("predicted_matching")
            if predicted is not None:
                if isinstance(predicted, SimilarityGraph):
                    pred_path = _artifact_path("predicted_match-dir", "predicted_match-name", "data/predicted", ".graphml")
                    predicted_save = predicted.graph.clean_attributes()
                    predicted_save.write_graphml(pred_path)
                else:
                    pred_path = _artifact_path("predicted_match-dir", "predicted_match-name", "data/predicted", ".txt")
                    _write_text(pred_path, predicted)

        if _has_stage("evaluation", "bert_evaluation"):
            result = self.cache.get("result") or self.cache.get("evaluation_result")
            if result is not None:
                result_path = _artifact_path("predicted_match-dir", "predicted_match-name", "data/predicted", ".result.txt")
                _write_text(result_path, result)

    def load(self, config, stages):
        # 需要确认再改
        state_config = config.get("state_management", {})

        def _has_stage(*names):
            return any(name in stages for name in names)

        def _artifact_path(dir_key, name_key, default_dir, default_ext):
            save_dir = state_config.get(dir_key, default_dir)
            version_name = config.get("version_name", "test")
            name = state_config.get(name_key) or version_name
            return os.path.join(save_dir, f"{name}{default_ext}")

        def _file_has_content(path):
            return os.path.exists(path) and os.path.getsize(path) > 0

        if _has_stage("graph_construction"):
            graph_path = _artifact_path("graph-dir", "graph-name", "data/graph", ".graphml")
            if _file_has_content(graph_path):
                graph = RepresentationGraph()
                graph.load_graph(graph_path)
                self.cache["representation_graph"] = graph
            else:
                self.cache["representation_graph"] = {}

        if _has_stage("embedding_training"):
            emb_path = _artifact_path("embedding-dir", "embedding_model-name", "data/embedding", ".emb")
            if _file_has_content(emb_path):
                try:
                    model = EmbeddingModel.load(emb_path)
                except Exception:
                    model = None
                self.cache["embedding_model"] = model if model is not None else {}
            else:
                self.cache["embedding_model"] = {}

        if _has_stage("feature_index_construction"):
            index_dir = state_config.get("feature_index-dir", "data/index")
            index_name = state_config.get("feature_index-name") or config.get("version_name", "test")
            save_dir = os.path.join(index_dir, index_name)
            manifest_path = os.path.join(save_dir, "manifest.json")
            if _file_has_content(manifest_path):
                self.cache["cg_feature_index"] = CGIndex.load(save_dir)
            else:
                self.cache["cg_feature_index"] = {}

        if _has_stage("bert_inference", "bert_evaluation"):
            bert_dir = state_config.get("bert-dir", "data/bert_model")
            version_name = config.get("version_name", "test")
            save_dir = os.path.join(bert_dir, version_name)
            config_path = os.path.join(save_dir, "config.json")
            tokenizer_path = os.path.join(save_dir, "tokenizer_config.json")

            if _file_has_content(config_path) and _file_has_content(tokenizer_path):
                try:
                    from transformers import AutoModelForSequenceClassification, AutoTokenizer

                    model_name = config.get("model_name", "bert")
                    bert_model = Model(model_name=model_name)
                    bert_model.model = AutoModelForSequenceClassification.from_pretrained(save_dir)
                    bert_model.tokenizer = AutoTokenizer.from_pretrained(save_dir)
                    self.cache["bert_model"] = {
                        "model": bert_model.get_model(),
                        "tokenizer": bert_model.get_tokenizer(),
                    }
                except Exception:
                    self.cache["bert_model"] = {}
            else:
                self.cache["bert_model"] = {}

            # evaluation --> similarityGraph

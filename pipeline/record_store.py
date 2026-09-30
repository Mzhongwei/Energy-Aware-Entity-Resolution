"""Record texts for the BERT matcher, keyed by record id.

Normalization writes one small JSONL file per window (``{"rid": ..., "text": ...}`` per line):
the batch training stage for the indexed (left) side and the incremental worker for the
query (right) side. The BERT matching stage loads them lazily, so it can judge any pair the
embedding decision emits without reading the graph or the source files.
"""
from __future__ import annotations

import json
import os
from typing import Iterable, Mapping

from utils.utils import data_cleaning

DEFAULT_MAX_TEXT_CHARS = 4096
# Never part of a record's text: ids are not attributes, and cluster_id is a ground-truth
# label (normalization drops the same columns before building the graph).
_EXCLUDED_COLUMNS = {"id", "rid", "cluster_id"}


def serialize_record_text(attributes: Iterable[tuple[str, object]]) -> str:
    """BERT input text for one record: ``name + value`` for each attribute in order, cleaned.

    BERT training (``sequence_generating_m1``) and streaming inference both build texts with
    this function, so a record is serialized identically on both sides.
    """
    return data_cleaning("".join(str(name) + str(value) for name, value in attributes))


def record_texts(config: dict, raw_data, rids, max_chars: int = DEFAULT_MAX_TEXT_CHARS) -> dict[str, str]:
    """Map each row of ``raw_data`` (pre-normalization, in row order) to its record text."""
    source_field = str((config.get("record_ids") or {}).get("source_field", "")).strip().lower()
    columns = [column for column in raw_data.columns
               if str(column).strip().lower() not in _EXCLUDED_COLUMNS | {source_field}]
    texts = {}
    for rid, values in zip(rids, raw_data[columns].itertuples(index=False, name=None)):
        rid = rid[0] if isinstance(rid, (list, tuple)) else rid
        # BERT truncates to max_length tokens anyway; the cap only bounds stored bytes.
        texts[str(rid)] = serialize_record_text(zip(columns, values))[:max_chars]
    return texts


class RecordStore:
    """Append-only directory of per-window record-text files."""

    def __init__(self, directory: str):
        self.directory = directory
        self._texts: dict[str, str] = {}
        self._loaded: set[str] = set()

    def write(self, phase: str, window_index: int, texts: Mapping[str, str]) -> str:
        os.makedirs(self.directory, exist_ok=True)
        path = os.path.join(self.directory, f"{phase}-{int(window_index):06d}.jsonl")
        temporary = f"{path}.{os.getpid()}.tmp"
        with open(temporary, "w", encoding="utf-8") as output:
            for rid, text in texts.items():
                output.write(json.dumps({"rid": rid, "text": text}, ensure_ascii=False) + "\n")
        os.replace(temporary, path)
        return path

    def refresh(self) -> int:
        """Load window files written since the last refresh; returns how many were read."""
        if not os.path.isdir(self.directory):
            return 0
        new_files = sorted(name for name in os.listdir(self.directory)
                           if name.endswith(".jsonl") and name not in self._loaded)
        for name in new_files:
            with open(os.path.join(self.directory, name), encoding="utf-8") as source:
                for line in source:
                    if line.strip():
                        record = json.loads(line)
                        self._texts[record["rid"]] = record["text"]
            self._loaded.add(name)
        return len(new_files)

    def get_many(self, rids: Iterable[str]) -> dict[str, str]:
        rids = list(rids)
        if any(rid not in self._texts for rid in rids):
            self.refresh()
        missing = [rid for rid in rids if rid not in self._texts]
        if missing:
            raise KeyError(f"No record text for ids {missing[:5]} in {self.directory}")
        return {rid: self._texts[rid] for rid in rids}

    def __len__(self):
        return len(self._texts)


def record_store_for(config: dict) -> RecordStore:
    from utils.pipeline_io import get_model_directory

    return RecordStore(get_model_directory(config, "records"))


def write_window_records(config: dict, phase: str, window_index: int, raw_data, processed) -> str:
    """Persist one normalized window's record texts (``processed`` carries the minted rids)."""
    max_chars = int((config.get("bert_matching") or {}).get("max_text_chars", DEFAULT_MAX_TEXT_CHARS))
    texts = record_texts(config, raw_data, processed["rid"].tolist(), max_chars)
    return record_store_for(config).write(phase, window_index, texts)


__all__ = [
    "RecordStore",
    "record_store_for",
    "record_texts",
    "serialize_record_text",
    "write_window_records",
]

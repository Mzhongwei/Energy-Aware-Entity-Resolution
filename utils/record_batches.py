import json
import os
from collections.abc import Iterator

import pandas as pd


SUPPORTED_RECORD_FORMATS = {"csv", "jsonl", "ndjson"}


def _record_format(path: str, configured_format: str | None = None) -> str:
    if not path:
        raise ValueError("A record source path is required.")
    record_format = str(configured_format or os.path.splitext(path)[1].lstrip(".")).lower()
    if record_format not in SUPPORTED_RECORD_FORMATS:
        raise ValueError(
            f"Unsupported record format '{record_format}'; expected one of {sorted(SUPPORTED_RECORD_FORMATS)}."
        )
    return record_format


def _iter_jsonl_batches(path: str, max_rows: int, max_bytes: int) -> Iterator[pd.DataFrame]:
    rows = []
    buffered_bytes = 0
    with open(path, "rb") as source:
        for line_number, raw_line in enumerate(source, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON object at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"JSONL record at {path}:{line_number} must be an object.")
            rows.append(record)
            buffered_bytes += len(raw_line)
            if len(rows) >= max_rows or buffered_bytes >= max_bytes:
                yield pd.DataFrame.from_records(rows)
                rows = []
                buffered_bytes = 0
    if rows:
        yield pd.DataFrame.from_records(rows)


def iter_record_batches(
    path: str,
    *,
    batch_rows: int,
    max_batch_bytes: int,
    record_format: str | None = None,
) -> Iterator[pd.DataFrame]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Record source does not exist: {path}")
    batch_rows = int(batch_rows)
    max_batch_bytes = int(max_batch_bytes)
    if batch_rows <= 0:
        raise ValueError("batch_rows must be greater than zero.")
    if max_batch_bytes <= 0:
        raise ValueError("max_batch_bytes must be greater than zero.")

    resolved_format = _record_format(path, record_format)
    if resolved_format == "csv":
        yield from pd.read_csv(path, chunksize=batch_rows)
        return
    yield from _iter_jsonl_batches(path, batch_rows, max_batch_bytes)


__all__ = ["SUPPORTED_RECORD_FORMATS", "iter_record_batches"]

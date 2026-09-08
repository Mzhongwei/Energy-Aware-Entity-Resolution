import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from utils.record_batches import iter_record_batches


ROOT = Path(__file__).resolve().parents[1]
NORMALIZATION_SPEC = importlib.util.spec_from_file_location(
    "windowed_training_normalization", ROOT / "pipeline" / "normalization.py"
)
NORMALIZATION = importlib.util.module_from_spec(NORMALIZATION_SPEC)
NORMALIZATION_SPEC.loader.exec_module(NORMALIZATION)


class RecordBatchTests(unittest.TestCase):
    def test_configured_source_ids_are_preserved_as_rids(self):
        raw_data = pd.DataFrame.from_records(
            [{"id": 10, "new_id": "A_7", "title": "example"}]
        )
        normalized = NORMALIZATION.index_normalization(
            {
                "record_ids": {
                    "source_field": "new_id",
                    "left_prefix": "A_",
                    "right_prefix": "B_",
                }
            },
            raw_data=raw_data,
            is_training=True,
        )

        self.assertEqual(normalized["rid"].tolist(), [["A_7"]])
        self.assertNotIn("id", normalized.columns)
        self.assertNotIn("new_id", normalized.columns)

    def test_jsonl_batches_are_bounded_and_preserve_objects(self):
        records = [
            {"new_id": f"A_{index}", "title": f"record {index}"}
            for index in range(5)
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

            batches = list(
                iter_record_batches(
                    str(path),
                    batch_rows=2,
                    max_batch_bytes=1024,
                )
            )

        self.assertEqual([len(batch) for batch in batches], [2, 2, 1])
        self.assertEqual(
            [record_id for batch in batches for record_id in batch["new_id"].tolist()],
            [f"A_{index}" for index in range(5)],
        )

    def test_jsonl_rejects_non_object_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            path.write_text("[1, 2, 3]\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be an object"):
                list(iter_record_batches(str(path), batch_rows=2, max_batch_bytes=1024))

    def test_jsonl_batches_honor_the_byte_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            path.write_text(
                json.dumps({"title": "first record"}) + "\n"
                + json.dumps({"title": "second record"}) + "\n",
                encoding="utf-8",
            )
            batches = list(
                iter_record_batches(str(path), batch_rows=100, max_batch_bytes=1)
            )

        self.assertEqual([len(batch) for batch in batches], [1, 1])

    def test_csv_uses_the_same_batch_interface(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.csv"
            path.write_text("id,title\n1,one\n2,two\n3,three\n", encoding="utf-8")
            batches = list(iter_record_batches(str(path), batch_rows=2, max_batch_bytes=1024))

        self.assertEqual([len(batch) for batch in batches], [2, 1])


if __name__ == "__main__":
    unittest.main()

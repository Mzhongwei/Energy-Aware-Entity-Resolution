import importlib
import sys
import types
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import pandas as pd


APP_ROOT = Path(__file__).resolve().parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


class NormalizationTraceFrequencyTest(unittest.TestCase):
    def test_embedding_worker_delegates_incremental_mode(self):
        worker = importlib.import_module("entries.worker.normalization_embedding")
        calls = []

        def fake_run_argo_incremental(mode=None):
            calls.append(mode)

        fake_distributions = types.ModuleType("distributions")
        fake_normalization_distribution = types.ModuleType(
            "distributions.normalization_distribution"
        )
        fake_normalization_distribution.run_argo_incremental = fake_run_argo_incremental

        argv = [
            "normalization_embedding.py",
            "--config",
            "/tmp/test-config.yaml",
            "--mode",
            "embedding-inference-inc",
        ]
        with patch.dict(
            sys.modules,
            {
                "distributions": fake_distributions,
                "distributions.normalization_distribution": fake_normalization_distribution,
            },
        ), patch.object(sys, "argv", argv):
            worker.main()

        self.assertEqual(calls, ["embedding-inference-inc"])

    def test_incremental_normalization_writes_trace_once_per_output_buffer_per_window(self):
        sys.modules.pop("distributions.normalization_distribution", None)
        fake_pipeline = types.ModuleType("pipeline")
        fake_pipeline_normalization = types.ModuleType("pipeline.normalization")
        fake_pipeline_normalization.index_normalization = lambda *args, **kwargs: None
        fake_pipeline_normalization.sequence_generating_m1 = lambda df: df
        with patch.dict(
            sys.modules,
            {
                "pipeline": fake_pipeline,
                "pipeline.normalization": fake_pipeline_normalization,
            },
        ):
            normalization_distribution = importlib.import_module(
                "distributions.normalization_distribution"
            )

        raw_windows = [
            pd.DataFrame({"name": ["a", "b", "c"]}),
            pd.DataFrame({"name": ["d", "e"]}),
        ]
        wait_results = ["ready-0", "ready-1", None]
        load_results = list(raw_windows)
        write_calls = []
        delete_calls = []
        eos_calls = []

        def fake_write_buffer(data, output_dir, window_index, extension="json"):
            write_calls.append((len(data), output_dir, window_index, extension))

        patches = [
            patch.object(normalization_distribution, "BUFFER_PATH", "/tmp/eaer-test/"),
            patch.object(normalization_distribution, "load_config", lambda: {"mode": ""}),
            patch.object(
                normalization_distribution,
                "_maybe_reset_rid_counter",
                lambda config, force=False: "/tmp/counter.txt",
            ),
            patch.object(
                normalization_distribution,
                "wait_for_buffer",
                lambda buffer_dir, timeout_seconds=60: wait_results.pop(0),
            ),
            patch.object(
                normalization_distribution,
                "load_earliest_buffer",
                lambda buffer_dir: load_results.pop(0),
            ),
            patch.object(
                normalization_distribution,
                "get_earliest_window_index",
                lambda buffer_dir: len(delete_calls),
            ),
            patch.object(
                normalization_distribution,
                "normalization",
                lambda config, raw_data, is_training=False: raw_data,
            ),
            patch.object(normalization_distribution, "write_buffer", fake_write_buffer),
            patch.object(
                normalization_distribution,
                "delete_earliest_buffer_file",
                lambda buffer_dir: delete_calls.append(buffer_dir),
            ),
            patch.object(
                normalization_distribution,
                "write_eos",
                lambda output_dir, reason="stream_completed": eos_calls.append(
                    (output_dir, reason)
                ),
            ),
        ]

        with ExitStack() as stack:
            for patcher in patches:
                stack.enter_context(patcher)
            normalization_distribution.run_argo_incremental(mode="embedding-inference-inc")

        self.assertEqual(len(write_calls), 4)
        self.assertEqual([call[0] for call in write_calls], [3, 3, 2, 2])
        self.assertEqual([call[3] for call in write_calls], ["csv", "csv", "csv", "csv"])
        self.assertEqual(
            [call[1] for call in write_calls],
            [
                "/tmp/eaer-test/processed_data",
                "/tmp/eaer-test/processed_data_feature",
                "/tmp/eaer-test/processed_data",
                "/tmp/eaer-test/processed_data_feature",
            ],
        )
        self.assertEqual(len(delete_calls), 2)
        self.assertEqual(len(eos_calls), 2)


if __name__ == "__main__":
    unittest.main()

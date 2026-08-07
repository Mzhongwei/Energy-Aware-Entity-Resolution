import os
import tempfile
import threading
import time
import unittest

from utils.pipeline_io import (
    delete_earliest_buffer_file,
    load_earliest_buffer,
    wait_for_buffer,
    wait_for_embedding_buffer,
    write_buffer,
    write_eos,
)


class IncrementalTimeoutEosTest(unittest.TestCase):
    def test_unbounded_wait_survives_idle_period_until_data_arrives(self):
        with tempfile.TemporaryDirectory() as buffer_dir:
            writer = threading.Thread(
                target=lambda: (time.sleep(0.05), write_buffer([{"id": 1}], buffer_dir, 1)),
                daemon=True,
            )
            writer.start()

            found = wait_for_buffer(buffer_dir, timeout_seconds=None, poll_interval_seconds=0.01)

            writer.join(timeout=1)
            self.assertIsNotNone(found)
            self.assertFalse(os.path.basename(found).startswith("eos_"))

    def test_finite_wait_is_available_for_startup_only(self):
        with tempfile.TemporaryDirectory() as buffer_dir:
            found = wait_for_buffer(buffer_dir, timeout_seconds=0.03, poll_interval_seconds=0.01)
            self.assertIsNone(found)

    def test_data_is_consumed_before_later_eos(self):
        with tempfile.TemporaryDirectory() as buffer_dir:
            write_buffer([["a", "b", 0.9]], buffer_dir, 26)
            write_eos(buffer_dir)

            self.assertEqual(load_earliest_buffer(buffer_dir), [["a", "b", 0.9]])
            self.assertTrue(delete_earliest_buffer_file(buffer_dir))
            self.assertIsNone(load_earliest_buffer(buffer_dir))

    def test_embedding_wait_reports_eos_when_window_will_never_arrive(self):
        with tempfile.TemporaryDirectory() as buffer_dir:
            write_eos(buffer_dir)
            found = wait_for_embedding_buffer(
                buffer_dir,
                window_index=26,
                timeout_seconds=None,
                poll_interval_seconds=0.01,
            )
            self.assertTrue(os.path.basename(found).startswith("eos_"))


if __name__ == "__main__":
    unittest.main()

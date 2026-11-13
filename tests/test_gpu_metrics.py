"""Metrics stay observational and never turn missing observations into idle GPUs."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts.gpu_metrics import GPUMetricsSampler, query_gpus


def devices(utilization=25):
    return [dict(uuid=f"GPU-{index}", index=index, utilization_gpu_pct=utilization,
                 memory_used_mib=100, memory_total_mib=24000) for index in range(8)]


class MetricsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.now = 0.0
        self.query_calls = 0
        self.failed = False

        def query():
            self.query_calls += 1
            if self.failed:
                raise subprocess.TimeoutExpired("nvidia-smi", 5)
            return devices()

        self.sampler = GPUMetricsSampler(self.root, clock=lambda: self.now,
            wall_clock=lambda: 1_800_000_000 + self.now, query=query,
            waiver="The user accepts low utilization for this experiment")

    def tearDown(self):
        self.temporary.cleanup()

    def sample_at(self, now):
        self.now = float(now)
        return self.sampler.maybe_sample(workers=[dict(task="example", state="running")])

    def test_query_reads_all_devices_once_with_five_second_timeout(self):
        output = "\n".join(f"GPU-{i}, {i}, 25, 100, 24000" for i in range(8))
        with patch("scripts.gpu_metrics.subprocess.run", return_value=subprocess.CompletedProcess([], 0, output)) as run:
            self.assertEqual(query_gpus(), devices())
        self.assertEqual(run.call_args.kwargs["timeout"], 5)
        self.assertEqual(run.call_count, 1)
        with patch("scripts.gpu_metrics.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "GPU-0, 0, N/A, 0, 24000")):
            with self.assertRaises(ValueError):
                query_gpus()

    def test_sampling_is_throttled_and_startup_is_not_a_full_window(self):
        summary = self.sample_at(0)
        self.assertEqual(summary["assessment"], "unknown")
        self.assertFalse(summary["complete_30_minute_window"])
        self.assertEqual(summary["observed_mean_utilization_gpu_pct"], 25)
        self.assertIsNone(summary["mean_utilization_gpu_pct"])
        self.assertTrue(summary["low_utilization_waiver"]["accepted"])
        self.assertIsNone(self.sample_at(29))
        self.sample_at(30)
        self.assertEqual(self.query_calls, 2)
        raw = [json.loads(line) for line in (self.root / "gpu_metrics.jsonl").read_text().splitlines()]
        self.assertEqual(len(raw), 2)
        self.assertEqual(raw[0]["workers"][0]["task"], "example")

    def test_complete_window_with_sufficient_coverage_reports_actual_values(self):
        for instant in range(0, 1801, 30):
            summary = self.sample_at(instant)
        self.assertEqual(summary["assessment"], "measured")
        self.assertTrue(summary["complete_30_minute_window"])
        self.assertEqual(summary["valid_coverage"], 1)
        self.assertEqual(summary["mean_utilization_gpu_pct"], 25)
        self.assertEqual(len(summary["per_gpu"]), 8)
        self.assertEqual(summary["unknown_reasons"], [])

    def test_long_interior_gap_is_unknown_despite_above_80_percent_coverage(self):
        for instant in range(0, 1801, 30):
            if 600 <= instant < 780:
                continue
            summary = self.sample_at(instant)
        self.assertGreaterEqual(summary["valid_coverage"], .8)
        self.assertEqual(summary["largest_observation_gap_s"], 210)
        self.assertEqual(summary["assessment"], "unknown")
        self.assertIsNone(summary["mean_utilization_gpu_pct"])
        self.assertEqual(summary["observed_mean_utilization_gpu_pct"], 25)

    def test_full_window_without_80_percent_coverage_is_unknown(self):
        for instant in range(0, 1801, 60):
            summary = self.sample_at(instant)
        self.assertTrue(summary["complete_30_minute_window"])
        self.assertEqual(summary["valid_coverage"], .5)
        self.assertFalse(summary["valid_coverage_at_least_80_percent"])
        self.assertEqual(summary["assessment"], "unknown")
        self.assertIn("valid_coverage_below_80_percent", summary["unknown_reasons"])

    def test_query_failure_has_no_zero_gpu_values_and_stale_data_is_unknown(self):
        for instant in range(0, 1801, 30):
            self.sample_at(instant)
        self.failed = True
        summary = self.sample_at(2000)
        self.assertFalse(summary["latest_sample_valid"])
        self.assertEqual(summary["last_valid_sample_age_s"], 200)
        self.assertEqual(summary["assessment"], "unknown")
        self.assertIsNone(summary["mean_utilization_gpu_pct"])
        self.assertEqual(summary["observed_mean_utilization_gpu_pct"], 25)
        last = json.loads((self.root / "gpu_metrics.jsonl").read_text().splitlines()[-1])
        self.assertEqual(last["gpus"], [])
        self.assertEqual(last["error"]["type"], "TimeoutExpired")

    def test_partial_allocation_is_not_counted_as_valid_coverage(self):
        self.sampler.query = lambda: devices()[:7]
        summary = self.sample_at(1800)
        self.assertEqual(summary["valid_samples"], 0)
        self.assertEqual(summary["valid_coverage"], 0)
        self.assertIsNone(summary["observed_mean_utilization_gpu_pct"])
        self.assertEqual(summary["assessment"], "unknown")

    def test_old_samples_leave_the_rolling_window(self):
        self.sample_at(0)
        self.sample_at(30)
        summary = self.sample_at(4000)
        self.assertEqual(len(self.sampler.samples), 1)
        self.assertEqual(summary["valid_samples"], 1)
        self.assertEqual(len((self.root / "gpu_metrics.jsonl").read_text().splitlines()), 3)
        self.assertEqual(json.loads((self.root / "gpu_metrics_summary.json").read_text()), summary)

    def test_monitoring_io_or_unexpected_query_failure_does_not_escape(self):
        with patch("scripts.gpu_metrics._write_json", side_effect=OSError("disk full")):
            summary = self.sample_at(0)
        self.assertEqual(summary["assessment"], "unknown")
        self.assertEqual(summary["sampler_error"]["type"], "OSError")
        self.sampler.query = lambda: [dict(malformed="row")]
        summary = self.sample_at(30)
        self.assertEqual(summary["assessment"], "unknown")
        self.assertEqual(summary["sampler_error"]["type"], "KeyError")


if __name__ == "__main__":
    unittest.main()

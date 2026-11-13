"""Small, observational GPU sampler for a single multi-worker Pod (stdlib only)."""
import argparse
from collections import deque
import csv
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time


def query_gpus():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,index,utilization.gpu,memory.used,memory.total",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=5)
    devices = []
    for row in csv.reader(io.StringIO(result.stdout)):
        if not row:
            continue
        if len(row) != 5:
            raise ValueError("GPU metrics require five CSV columns")
        identifier, index, utilization, used, total = [value.strip() for value in row]
        index = int(index)
        utilization, used, total = map(float, (utilization, used, total))
        if (not identifier.startswith("GPU-") or index < 0
                or not all(math.isfinite(value) for value in (utilization, used, total))
                or not 0 <= utilization <= 100 or not 0 <= used <= total or total <= 0):
            raise ValueError("GPU metrics contain unavailable or invalid values")
        devices.append(dict(uuid=identifier, index=index, utilization_gpu_pct=utilization,
                            memory_used_mib=used, memory_total_mib=total))
    if (not devices or len({item["uuid"] for item in devices}) != len(devices)
            or len({item["index"] for item in devices}) != len(devices)):
        raise ValueError("GPU metrics require distinct devices")
    return devices


def _timestamp(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat()


def _write_json(path, value):
    descriptor, temporary = tempfile.mkstemp(prefix=".gpu-metrics-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, allow_nan=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class GPUMetricsSampler:
    """Record observations, never terminate workers or substitute missing values.

    Call maybe_sample(workers=compact_states) from the supervisor loop. One query
    is made per interval, with a five-second timeout. JSONL keeps the full audit
    trail; rolling aggregates retain only the last 1,800 seconds. A valid sample
    contributes at most one sampling interval of time coverage. An incomplete
    window, coverage below 80%, or an observation gap above three minutes makes
    the utilization assessment unknown; observed values remain available.
    """
    def __init__(self, output, expected_gpu_count=8, interval_s=30, waiver=None,
                 *, clock=None, wall_clock=None, query=None):
        if (type(expected_gpu_count) is not int or expected_gpu_count < 1
                or not math.isfinite(interval_s) or interval_s <= 0):
            raise ValueError("GPU count and sampling interval must be positive")
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.expected_gpu_count = expected_gpu_count
        self.interval_s, self.window_s, self.max_gap_s = float(interval_s), 1800.0, 180.0
        self.clock, self.wall_clock = clock or time.monotonic, wall_clock or time.time
        self.query = query or query_gpus
        self.started = self.clock()
        self.started_at = _timestamp(self.wall_clock())
        self.last_query = None
        self.devices = None
        self.samples = deque()
        self.waiver = dict(accepted=bool(waiver), reason=waiver)

    def maybe_sample(self, workers=None, *, force=False):
        try:
            return self._maybe_sample(workers, force=force)
        except Exception as error:
            # Monitoring failures, including a full output volume, must not
            # terminate a running experiment. The supervisor can log this result.
            summary = dict(assessment="unknown", sampler_error=dict(
                type=type(error).__name__, message=str(error)[:300]),
                mean_utilization_gpu_pct=None, low_utilization_waiver=self.waiver)
            try:
                _write_json(self.output / "gpu_metrics_summary.json", summary)
            except Exception:
                pass
            return summary

    def _maybe_sample(self, workers=None, *, force=False):
        now = self.clock()
        if not force and self.last_query is not None and now - self.last_query < self.interval_s:
            return None
        self.last_query = now
        sample = dict(timestamp=_timestamp(self.wall_clock()), elapsed_s=now - self.started,
                      valid=False, gpus=[], workers=workers if workers is not None else [])
        try:
            sample["gpus"] = self.query()
            identifiers = {item["uuid"] for item in sample["gpus"]}
            if len(sample["gpus"]) != self.expected_gpu_count or len(identifiers) != self.expected_gpu_count:
                raise ValueError("GPU metrics do not cover the requested GPU count")
            if self.devices is not None and identifiers != self.devices:
                raise ValueError("The observed GPU allocation changed")
            self.devices = identifiers
            sample["valid"] = True
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            # Do not record stderr or replace an unavailable measurement by zero.
            sample["error"] = dict(type=type(error).__name__, message=str(error)[:300])
        self.samples.append((now, sample))
        # Keep one extra interval for coverage overlapping the left window edge.
        while self.samples and self.samples[0][0] + self.interval_s < now - self.window_s:
            self.samples.popleft()
        with (self.output / "gpu_metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(sample, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
        summary = self._summary(now, sample)
        _write_json(self.output / "gpu_metrics_summary.json", summary)
        return summary

    def _summary(self, now, latest):
        left = max(self.started, now - self.window_s)
        valid = [(when, item) for when, item in self.samples if item["valid"]]
        covered, cursor = 0.0, left
        for when, _ in valid:
            start, end = max(left, when), min(now, when + self.interval_s)
            if end > max(start, cursor):
                covered += end - max(start, cursor)
                cursor = end
        in_window = [(when, item) for when, item in self.samples if when >= left]
        observations = [(when, item) for when, item in valid if when >= left]
        observation_times = [max(left, when) for when, _ in valid]
        last_age = now - valid[-1][0] if valid else None
        boundaries = [left, *observation_times, now]
        largest_gap = max((b - a for a, b in zip(boundaries, boundaries[1:])), default=now - left)
        coverage = min(1.0, covered / self.window_s)
        complete = now - self.started >= self.window_s
        reasons = []
        if not complete:
            reasons.append("incomplete_30_minute_window")
        if coverage < .8:
            reasons.append("valid_coverage_below_80_percent")
        if last_age is None:
            reasons.append("no_valid_gpu_samples")
        elif last_age > self.max_gap_s or largest_gap > self.max_gap_s:
            reasons.append("missing_gpu_data_for_more_than_3_minutes")
        per_gpu = []
        for identifier in sorted(self.devices or ()):
            records = [gpu for _, item in observations for gpu in item["gpus"] if gpu["uuid"] == identifier]
            if records:
                per_gpu.append(dict(uuid=identifier, index=records[-1]["index"], samples=len(records),
                    observed_mean_utilization_gpu_pct=sum(r["utilization_gpu_pct"] for r in records) / len(records),
                    latest_utilization_gpu_pct=records[-1]["utilization_gpu_pct"],
                    latest_memory_used_mib=records[-1]["memory_used_mib"],
                    memory_total_mib=records[-1]["memory_total_mib"]))
        values = [gpu["utilization_gpu_pct"] for _, item in observations for gpu in item["gpus"]]
        observed_mean = sum(values) / len(values) if values else None
        return dict(timestamp=latest["timestamp"], started_at=self.started_at,
                    elapsed_s=now - self.started, window_s=self.window_s,
                    sampling_interval_s=self.interval_s, expected_gpu_count=self.expected_gpu_count,
                    complete_30_minute_window=complete, valid_coverage=coverage,
                    valid_coverage_at_least_80_percent=coverage >= .8,
                    valid_samples=len(observations), failed_samples=sum(not item["valid"] for _, item in in_window),
                    last_valid_sample_age_s=last_age, largest_observation_gap_s=largest_gap,
                    assessment="unknown" if reasons else "measured", unknown_reasons=reasons,
                    mean_utilization_gpu_pct=None if reasons else observed_mean,
                    observed_mean_utilization_gpu_pct=observed_mean, per_gpu=per_gpu,
                    latest_sample_valid=latest["valid"], latest_error=latest.get("error"),
                    workers=latest["workers"], low_utilization_waiver=self.waiver)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-gpu-count", type=int, default=8)
    parser.add_argument("--interval", type=float, default=30)
    parser.add_argument("--waiver", help="Record the user's explicit low-utilization acceptance")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    sampler = GPUMetricsSampler(args.output, args.expected_gpu_count, args.interval, args.waiver)
    try:
        while True:
            summary = sampler.maybe_sample()
            if args.once:
                print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
                return 0 if summary.get("latest_sample_valid") else 1
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

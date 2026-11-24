#!/usr/bin/env python3
"""Run a frozen training supervisor, then prepare an unsubmitted evaluation Job."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_policy_evaluation import HandoffNotReady, _read, _sha, _write, verify_source_snapshot


def train_and_prepare(plan_path):
    plan_path = Path(plan_path).resolve()
    plan = _read(plan_path)
    run = Path(plan["run"]).resolve()
    if plan_path != run / "parallel_plan.json" or plan.get("mode", "run") != "run":
        raise HandoffNotReady("Wrapper requires the frozen training parallel_plan.json")
    verify_source_snapshot(run)
    with (run / ".training_evaluation_pipeline.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        pipeline = {"status": "training", "plan": str(plan_path), "job_created": False,
                    "started_at_utc": datetime.now(timezone.utc).isoformat(),
                    "training_returncode": None, "evaluation_submission_performed": False}
        receipt_path = run / "pipeline_receipt.json"
        config = _read(run / "config.json")
        handoff = config.get("evaluation_handoff", {})
        if not handoff.get("template") or not handoff.get("name"):
            raise HandoffNotReady("Training configuration has no evaluation_handoff template/name")
        process, cancelled, old_handlers = None, [], {}

        def stop(signum, _frame):
            cancelled.append(signum)
            if process is not None and process.poll() is None:
                process.send_signal(signum)

        _write(receipt_path, pipeline)
        for signum in (signal.SIGTERM, signal.SIGINT):
            old_handlers[signum] = signal.signal(signum, stop)
        try:
            if not (run / "evaluation_preparation_receipt.json").exists():
                command = [sys.executable, "-u", str(run / "source/scripts/parallel_worker.py"), "--plan", str(plan_path)]
                pipeline["training_command"] = command
                _write(receipt_path, pipeline)
                process = subprocess.Popen(command)
                pipeline["training_returncode"] = process.wait()
            else:
                pipeline["training_supervisor_skipped"] = "Existing preparation receipt; verify without replaying training"
            parallel_receipt = run / "parallel_receipt.json"
            if parallel_receipt.exists():
                saved = _read(parallel_receipt)
                pipeline["training_receipt"] = {"path": str(parallel_receipt), "sha256": _sha(parallel_receipt),
                                                "status": saved.get("status"), "counts": saved.get("counts")}
            if cancelled:
                pipeline.update(status="interrupted", received_signals=cancelled,
                                handoff_skipped="Training stop was requested; preserve interrupted episodes")
            else:
                # Explicitly load the callback from the same training source snapshot.
                target = run / "source/scripts/prepare_policy_evaluation.py"
                spec = importlib.util.spec_from_file_location("_training_frozen_handoff", target)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                prepared = module.prepare_evaluation(run, handoff["template"], handoff["name"],
                    project=handoff.get("project"), runtime=handoff.get("runtime", "/mnt/auwomo-data/yumuzhen/physical-rsi"))
                pipeline.update(status="handoff_ready", evaluation_preparation_receipt=str(run / "evaluation_preparation_receipt.json"),
                    evaluation_preparation_receipt_sha256=_sha(run / "evaluation_preparation_receipt.json"),
                    training_counts=prepared["training_audit"]["counts"], job=prepared["job"],
                    evaluation_run=prepared["evaluation_run"], frozen_library_sha256=prepared["frozen_library"]["state_sha256"])
        except Exception as error:
            pipeline.update(status="handoff_blocked", error={"type": type(error).__name__, "message": str(error)},
                            training_audit=getattr(error, "audit", {}))
        finally:
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)
            pipeline["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            _write(receipt_path, pipeline)
        return pipeline


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    args = parser.parse_args(argv)
    try:
        receipt = train_and_prepare(args.plan)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({"status": "pipeline_blocked", "error": str(error), "job_created": False}), flush=True)
        return 2
    path = Path(args.plan).resolve().parent / "pipeline_receipt.json"
    print(json.dumps({"status": receipt["status"], "training_returncode": receipt["training_returncode"],
                      "pipeline_receipt": str(path), "pipeline_receipt_sha256": _sha(path),
                      "job_created": False, "job": receipt.get("job"),
                      "evaluation_run": receipt.get("evaluation_run"), "error": receipt.get("error")}), flush=True)
    return 0 if receipt["status"] == "handoff_ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Retain cluster receipts and distinguish interrupted runs from benchmark results.

This collector uses only the standard library and kubectl; it does not import
the simulation runtime. run_receipt.json is always separate from any benchmark
summary, and never invents success counts for an interrupted worker.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess


SUMMARY_KEYS = ("status", "native_successes", "total_tasks", "promoted_versions",
                "library_after", "library_unchanged")
INFRASTRUCTURE_REASONS = {
    "Evicted", "NodeLost", "DeadlineExceeded", "Preempted", "Shutdown",
    "UnexpectedAdmissionError", "NodeShutdown", "TerminationByKubelet",
}


def _read_json(path, errors, operation):
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError("JSON receipt must be an object")
        return value
    except (OSError, ValueError) as error:
        errors.append({"operation": operation, "error": type(error).__name__, "message": str(error)})
        return None


def _request(command, errors, operation):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    except (subprocess.TimeoutExpired, OSError) as error:
        errors.append({"operation": operation, "error": type(error).__name__,
                       "message": "kubectl exceeded its timeout" if isinstance(error, subprocess.TimeoutExpired)
                       else str(error)})
        return None
    if result.returncode:
        errors.append({"operation": operation, "error": "kubectl_failed", "returncode": result.returncode,
                       "stderr": result.stderr[-8000:]})
        return None
    return result.stdout


def _pod_status(pod):
    status = pod.get("status", {})
    containers = []
    for group in ("initContainerStatuses", "containerStatuses", "ephemeralContainerStatuses"):
        for container in status.get(group, []):
            current = container.get("state", {})
            terminated = current.get("terminated", {})
            previous = container.get("lastState", {}).get("terminated", {})
            record = {"name": container.get("name"), "kind": group,
                      "state": next(iter(current), "unknown"), "restart_count": container.get("restartCount", 0),
                      "exit_code": terminated.get("exitCode"), "reason": terminated.get("reason"),
                      "signal": terminated.get("signal")}
            if terminated.get("message"):
                record["message"] = terminated["message"]
            if previous:
                record["previous_termination"] = {key: previous[key]
                    for key in ("exitCode", "reason", "signal", "finishedAt") if key in previous}
            if current.get("waiting"):
                record["waiting"] = current["waiting"]
            containers.append(record)
    return {"name": pod.get("metadata", {}).get("name"), "phase": status.get("phase"),
            "reason": status.get("reason"), "message": status.get("message"),
            "node": pod.get("spec", {}).get("nodeName"), "containers": containers}


def build_receipt(root, job, pods, summaries, errors, sources):
    """Pure receipt assembly, also usable with static cluster JSON fixtures."""
    job_status = (job or {}).get("status", {})
    job_conditions = job_status.get("conditions", [])
    pod_reports = [_pod_status(pod) for pod in (pods or {}).get("items", [])]
    reasons = []
    for condition in job_conditions:
        if condition.get("status") == "True" and condition.get("reason"):
            reasons.append(condition["reason"])
    for pod in pod_reports:
        if pod["reason"]:
            reasons.append(pod["reason"])
        for container in pod["containers"]:
            if container["reason"]:
                reasons.append(container["reason"])
    reasons = sorted(set(reasons))
    infrastructure_reasons = sorted(set(reasons) & INFRASTRUCTURE_REASONS)
    job_failed = any(item.get("type") in {"Failed", "FailureTarget"} and item.get("status") == "True"
                     for item in job_conditions)
    pod_failed = any(pod["phase"] == "Failed" for pod in pod_reports)
    if summaries:
        status = "summary_available"
    elif infrastructure_reasons:
        status = "infrastructure_interrupted"
    elif job_failed or pod_failed:
        status = "run_failed_without_summary"
    elif any(item.get("type") == "Complete" and item.get("status") == "True" for item in job_conditions):
        status = "completed_without_summary"
    elif job_status.get("active") or any(pod["phase"] in {"Pending", "Running"} for pod in pod_reports):
        status = "running_without_summary"
    else:
        status = "summary_unavailable"
    return {"collected_at": datetime.now(timezone.utc).isoformat(), "run": str(root),
            "status": status, "collection_status": "partial" if errors else "complete",
            "job": {"name": (job or {}).get("metadata", {}).get("name"),
                    "status": job_status, "failed": job_failed},
            "pods": pod_reports, "status_reasons": reasons,
            "failure_reasons": [reason for reason in reasons if reason not in {"Completed", "CompletionsReached"}],
            "infrastructure_reasons": infrastructure_reasons,
            "summaries": summaries, "receipt_sources": sources,
            "collection_errors": errors,
            "benchmark_result_available": bool(summaries)}


def collect(root):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError("run must be an existing directory")
    errors, sources = [], {}
    manifest = _read_json(root / "job.json", errors, "read_job_manifest")
    name = (manifest or {}).get("metadata", {}).get("name")
    prefix = ["kubectl", "--kubeconfig", "/root/.kube/p6000-user.conf", "-n", "dreamzero-training",
              "--request-timeout=25s"]
    if name:
        for kind in ("job", "pods"):
            command = prefix + ["get", kind] + ([name] if kind == "job" else ["-l", "job-name=" + name]) + ["-o", "json"]
            raw = _request(command, errors, "get_" + kind)
            if raw is not None:
                try:
                    parsed = json.loads(raw)
                    if not isinstance(parsed, dict):
                        raise ValueError("cluster JSON must be an object")
                except ValueError as error:
                    errors.append({"operation": "get_" + kind, "error": "invalid_json", "message": str(error)})
                else:
                    (root / (kind + "_receipt.json")).write_text(raw)
                    sources[kind] = "collected"
        raw = _request(prefix + ["logs", "job/" + name], errors, "get_logs")
        if raw is not None:
            (root / "pod.log").write_text(raw)
            sources["logs"] = "collected"
    else:
        errors.append({"operation": "read_job_manifest", "error": "missing_job_name",
                       "message": "Cannot query cluster without job.json metadata.name"})
    receipts = {}
    for kind in ("job", "pods"):
        receipts[kind] = _read_json(root / (kind + "_receipt.json"), errors, "read_" + kind + "_receipt")
        sources.setdefault(kind, "previous" if receipts[kind] is not None else "missing")
    sources.setdefault("logs", "previous" if (root / "pod.log").exists() else "missing")
    summaries = {}
    for directory in ("campaign", "evaluation"):
        path = root / directory / "summary.json"
        report = _read_json(path, errors, "read_" + directory + "_summary")
        if report is not None:
            compact = {key: report[key] for key in SUMMARY_KEYS if key in report}
            summaries[directory] = {"path": str(path), "summary": compact}
    receipt = build_receipt(root, receipts["job"] or manifest, receipts["pods"], summaries, errors, sources)
    (root / "run_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": receipt["status"], "collection_status": receipt["collection_status"],
                      "job": receipt["job"]["name"], "failure_reasons": receipt["failure_reasons"],
                      "summaries": {kind: value["summary"] for kind, value in summaries.items()},
                      "collection_errors": errors, "receipt": str(root / "run_receipt.json")},
                     ensure_ascii=False, indent=2))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", help="existing absolute run directory")
    collect(parser.parse_args().run)


if __name__ == "__main__":
    main()

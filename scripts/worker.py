"""Cluster entry; reuse existing auth in memory, never copy credentials to artifacts."""
import json
import os
from pathlib import Path
from evo.cli import run_config, evaluate_frozen

root = Path(os.environ["EVO_RUN"])
config = json.loads((root/"config.json").read_text())
provider = config.get("agent", {})
if provider.get("use_existing_runtime_auth"):
    source = Path(os.environ["EVO_RUNTIME"])/"runtime/kinex_gpt6/agent/auth.json"
    secret = json.loads(source.read_text())["physical-rsi"]["key"]
    os.environ[provider.get("api_key_env", "OPENAI_API_KEY")] = secret
    del secret
if config.get("mode") == "evaluate_policy":
    from evo.policy_evaluation import run_policy_evaluation
    report = run_policy_evaluation(config, root/"campaign")
elif config.get("mode") == "evaluate":
    report = evaluate_frozen(config, root/"evaluation")
else:
    report = run_config(config, root/"campaign")
summary = {key:report[key] for key in ("status", "task_status", "maintenance_status", "maintenance_trigger",
    "native_successes", "program_successes", "total_tasks", "total_episode_attempts",
    "promoted_versions", "accepted_case_count", "library_after", "library_unchanged") if key in report}
if "maintenance_runs" in report:
    summary["maintenance_runs_count"] = len(report["maintenance_runs"])
print(json.dumps(summary, indent=2), flush=True)
raise SystemExit(0 if report.get("status") == "complete" else 1)

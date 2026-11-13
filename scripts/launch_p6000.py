"""Freeze source and launch a CPU RoboCasa worker with an optional wall-time bound."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config")
    p.add_argument("--mode", choices=("run", "evaluate"))
    p.add_argument("--name", default="evo-"+datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))
    p.add_argument("--project", default="/mnt/auwomo-data/yumuzhen/my/evo")
    p.add_argument("--runtime", default="/mnt/auwomo-data/yumuzhen/physical-rsi")
    p.add_argument("--deadline", type=int, default=7200, help="Job seconds; 0 disables the wall-time deadline")
    checks = p.add_mutually_exclusive_group()
    checks.add_argument("--preflight", action="store_true")
    checks.add_argument("--reset-smoke", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if args.deadline < 0:
        p.error("deadline must be nonnegative; use 0 for no wall-time deadline")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,45}", args.name):
        p.error("name must be a Kubernetes name of at most 46 characters")
    root, runtime = Path(args.project).resolve(), Path(args.runtime).resolve()
    run = root / "runs" / args.name
    run.mkdir(parents=True, exist_ok=False)
    snapshot = run / "source"
    shutil.copytree(root / "evo", snapshot / "evo", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(root / "scripts", snapshot / "scripts", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(root / "pyproject.toml", snapshot / "pyproject.toml")
    if args.config:
        config = json.loads(Path(args.config).read_text())
        if args.mode:
            config["mode"] = args.mode
        config["library"] = str(root / "library") if "library" not in config else config["library"]
        if config.get("agent", {}).get("provider") == "scripted":
            destination = snapshot/"fixtures/responses.json"
            destination.parent.mkdir()
            shutil.copy(config["agent"]["responses"], destination)
            config["agent"]["responses"] = str(destination)
        (run / "config.json").write_text(json.dumps(config, indent=2)+"\n")
    elif not (args.preflight or args.reset_smoke):
        p.error("--config is required except with --preflight or --reset-smoke")
    hashes = {str(f.relative_to(snapshot)):hashlib.sha256(f.read_bytes()).hexdigest()
              for f in sorted(snapshot.rglob("*")) if f.is_file()}
    (run / "source_sha256.json").write_text(json.dumps(hashes, indent=2)+"\n")
    job = json.loads((runtime / "deployment/job_template.json").read_text())
    labels = dict(owner="yumuzhen", project="atomic-evo", purpose="two-loop-validation")
    job["metadata"].update(name="yumuzhen-"+args.name, labels=labels)
    job["spec"].update(backoffLimit=0, ttlSecondsAfterFinished=604800)
    if args.deadline:
        job["spec"]["activeDeadlineSeconds"] = args.deadline
    else:
        job["spec"].pop("activeDeadlineSeconds", None)
    pod = job["spec"]["template"]
    pod["metadata"]["labels"] = labels
    for key in ("nodeName", "nodeSelector", "affinity", "priorityClassName"):
        pod["spec"].pop(key, None)
    container = pod["spec"]["containers"][0]
    env = {i["name"]:i["value"] for i in container["env"] if "value" in i and not i["name"].startswith("RSI_")}
    env.update(PYTHONPATH=":".join([str(snapshot), str(runtime/"source/RoboEnv/src"),
        str(runtime/"source/RoboEnv/runtime/src"), str(runtime/"runtime/python")]),
        MUJOCO_GL="osmesa", PYOPENGL_PLATFORM="osmesa", LP_NUM_THREADS="1",
        EVO_RUNTIME=str(runtime), EVO_RUN=str(run))
    container["env"] = [dict(name=k,value=v) for k,v in env.items()]
    entry = "reset_smoke.py" if args.reset_smoke else ("preflight.py" if args.preflight else "worker.py")
    container["command"] = ["/opt/envs/dreamzero/bin/python", "-u", str(snapshot/"scripts"/entry)]
    container["resources"] = {"requests":{"cpu":"4", "memory":"12Gi", "ephemeral-storage":"1Gi"},
                              "limits":{"cpu":"8", "memory":"24Gi", "ephemeral-storage":"8Gi"}}
    for volume in pod["spec"]["volumes"]:
        if volume["name"] == "project":
            volume["hostPath"]["path"] = str(root)
    for mount in container["volumeMounts"]:
        if mount["name"] == "project":
            mount["mountPath"] = str(root)
    manifest = run / "job.json"
    manifest.write_text(json.dumps(job, indent=2)+"\n")
    print(json.dumps(dict(run=str(run), job=job["metadata"]["name"], source_sha256=hashes)), flush=True)
    if not args.dry_run:
        subprocess.run(["kubectl", "--kubeconfig", "/root/.kube/p6000-user.conf", "-n", "dreamzero-training",
                        "--request-timeout=30s", "create", "-f", str(manifest)], check=True, timeout=40)


if __name__ == "__main__":
    main()

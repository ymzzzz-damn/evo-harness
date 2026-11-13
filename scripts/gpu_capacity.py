"""Choose an eight-GPU scene count from measured two-environment peaks."""
import math
import re


def memory_mib(quantity):
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)(Ki|Mi|Gi|Ti|K|M|G|T)?", str(quantity))
    if not match:
        raise ValueError("Unsupported Kubernetes memory quantity: " + str(quantity))
    number, suffix = match.groups()
    units = {None: 1, "Ki": 1024, "Mi": 1024**2, "Gi": 1024**3, "Ti": 1024**4,
             "K": 1000, "M": 1000**2, "G": 1000**3, "T": 1000**4}
    return float(number) * units[suffix] / 1024**2


def choose_capacity(devices, proofs, maximum, policy):
    if len(devices) != 8 or len(proofs) != 8:
        raise ValueError("Capacity planning requires eight real scene probes")
    factor = float(policy.get("memory_safety_factor", 1.5))
    gpu_fraction = float(policy.get("gpu_memory_fraction", .8))
    host_fraction = float(policy.get("host_memory_fraction", .8))
    if factor < 1 or not 0 < gpu_fraction < 1 or not 0 < host_fraction < 1:
        raise ValueError("Invalid capacity headroom policy")
    per_gpu = []
    host_peak = 0
    for device, proof in zip(devices, proofs):
        sample = proof["capacity"]
        if (proof.get("status") != "passed" or not sample.get("dual_environment_verified")
                or sample["gpu_uuid"] != device["uuid"]):
            raise ValueError("Capacity probe does not prove the assigned GPU and dual environment")
        baseline = float(sample["baseline_gpu_mib"])
        peak = max(float(sample["one_scene_gpu_mib"]), float(sample["two_scene_gpu_mib"]))
        total = float(device["memory_mib"])
        rss = float(sample["peak_host_rss_mib"])
        if not all(math.isfinite(x) for x in (baseline, peak, total, rss)) or not 0 <= baseline <= peak <= total or rss <= 0:
            raise ValueError("Invalid measured memory values")
        worker_gpu = max(float(policy.get("min_worker_gpu_mib", 2048)), (peak - baseline) * factor)
        slots = math.floor(max(0, total * gpu_fraction - baseline) / worker_gpu)
        host_peak = max(host_peak, rss)
        per_gpu.append({"uuid": device["uuid"], "total_mib": total, "baseline_mib": baseline,
                        "measured_two_scene_peak_mib": peak, "reserved_mib_per_worker": worker_gpu,
                        "memory_worker_capacity": slots})
    worker_host = max(float(policy.get("min_worker_host_mib", 4096)), host_peak * factor)
    host_budget = min(float(policy["host_memory_budget_mib"]), float(policy["host_memory_limit_mib"]))
    host_slots = math.floor(host_budget * host_fraction / worker_host)
    cpu_slots = math.floor(float(policy["cpu_limit"]))
    per_gpu_slots = min(int(policy.get("max_workers_per_gpu", 8)), maximum // 8,
                        min(item["memory_worker_capacity"] for item in per_gpu), host_slots // 8, cpu_slots // 8)
    if per_gpu_slots < 1:
        raise RuntimeError("Measured capacity cannot safely fit one worker on each of eight GPUs")
    return {"max_parallel": per_gpu_slots * 8, "workers_per_gpu": per_gpu_slots,
            "requested_max_parallel": maximum, "per_gpu": per_gpu,
            "peak_host_rss_mib": host_peak, "reserved_host_mib_per_worker": worker_host,
            "host_memory_budget_mib": host_budget, "host_worker_capacity": host_slots,
            "cpu_worker_capacity": cpu_slots, "policy": policy,
            "measurement_scope": "Eight real GPU probes; exploration plus a simultaneous replay scene per process. Includes rendering and physics, without model requests. Memory headroom also covers differing task assets."}

"""Memory selection accounts for simultaneous maintenance and host memory."""
import pytest

from scripts.gpu_capacity import choose_capacity, memory_mib


def inputs():
    devices = [{"uuid": f"GPU-{i}", "memory_mib": "80000"} for i in range(8)]
    proofs = [{"status": "passed", "capacity": {"gpu_uuid": device["uuid"],
        "baseline_gpu_mib": 0, "one_scene_gpu_mib": 800, "two_scene_gpu_mib": 1600,
        "peak_host_rss_mib": 2000, "dual_environment_verified": True}} for device in devices]
    policy = dict(host_memory_budget_mib=262144, host_memory_limit_mib=393216,
                  cpu_limit=128, max_workers_per_gpu=8)
    return devices, proofs, policy


def test_host_budget_selects_six_workers_per_gpu_with_headroom():
    devices, proofs, policy = inputs()
    result = choose_capacity(devices, proofs, 64, policy)
    assert result["max_parallel"] == 48
    assert result["workers_per_gpu"] == 6
    assert result["reserved_host_mib_per_worker"] == 4096


def test_weakest_gpu_and_two_live_environments_bound_all_assignments():
    devices, proofs, policy = inputs()
    devices[0]["memory_mib"] = "16000"
    proofs[0]["capacity"]["two_scene_gpu_mib"] = 4000
    result = choose_capacity(devices, proofs, 64, policy)
    assert result["max_parallel"] == 16
    assert result["workers_per_gpu"] == 2


def test_requested_ceiling_and_cpu_budget_are_respected():
    devices, proofs, policy = inputs()
    assert choose_capacity(devices, proofs, 32, policy)["max_parallel"] == 32
    policy["cpu_limit"] = 24
    assert choose_capacity(devices, proofs, 64, policy)["max_parallel"] == 24


@pytest.mark.parametrize("change", [dict(gpu_uuid="GPU-wrong"), dict(dual_environment_verified=False),
                                     dict(peak_host_rss_mib=float("nan"))])
def test_missing_or_untrustworthy_measurements_prevent_dispatch(change):
    devices, proofs, policy = inputs()
    proofs[0]["capacity"].update(change)
    with pytest.raises(ValueError):
        choose_capacity(devices, proofs, 64, policy)


def test_cannot_fit_eight_scenes_is_an_error_before_exploration():
    devices, proofs, policy = inputs()
    policy["host_memory_budget_mib"] = 16000
    with pytest.raises(RuntimeError, match="one worker"):
        choose_capacity(devices, proofs, 64, policy)


def test_kubernetes_binary_memory_units():
    assert memory_mib("256Gi") == 262144
    assert memory_mib("4096Mi") == 4096

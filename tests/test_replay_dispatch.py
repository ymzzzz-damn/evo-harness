"""Offline scheduling/resource/lifecycle tests; no real GPU or model is used."""
import fcntl
from contextlib import ExitStack
import json
import multiprocessing
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from evo.io import sha256
from evo.loops import MaintenanceLoop
from evo.replay_dispatch import (_resources, dispatch_replays, replay_slot,
                                 validate_replay_configuration)


def slot_probe(root, queue):
    with replay_slot(Path(root), [{"uuid":"GPU-ONE"}, {"uuid":"GPU-TWO"}], 1) as (_, gpu, _, _):
        queue.put((time.monotonic(), "enter", gpu["uuid"], os.getpid()))
        time.sleep(.15)
        queue.put((time.monotonic(), "exit", gpu["uuid"], os.getpid()))


def slow_factory(config, output):
    Path(output).mkdir(parents=True)
    (Path(output) / "slow_factory_started.json").write_text(json.dumps({"pid":os.getpid()}))
    time.sleep(60)  # Test owner is killed after this explicit child-start marker.
    raise RuntimeError("Parent-death fixture should never finish this sleep")


def parent_death_owner(directory):
    directory = Path(directory)
    from test_replay_parallel_contract import contract_case, GOOD, contract_factory
    from evo.runtime_api import RuntimeAPI
    resources = resource_fixture(directory)
    resources["factory_ref"] = "test_replay_dispatch:slow_factory"
    dispatch_replays([contract_case("death-case")],
        {"atom":"A02", "code":GOOD, "sha256":sha256(GOOD), "dependencies":{}},
        directory / "cases", library=SimpleNamespace(root=directory / "unused-library"),
        adapter_factory=contract_factory, runtime_api=RuntimeAPI,
        supported_predicates=["joint_near"], code_timeout_s=180,
        parallelism=2, resources=resources)


def resource_fixture(directory):
    return {"token_root":str(Path(directory).resolve() / "tokens"), "gpus":[{"uuid":"GPU-ONE"}],
        "slots_per_gpu":1,
        "factory_ref":"test_replay_parallel_contract:contract_factory",
        "runtime_ref":"evo.runtime_api:RuntimeAPI",
        "renderer_mapping_ref":"test_replay_parallel_contract:contract_renderer_mapping",
        "renderer_evidence_ref":"test_replay_parallel_contract:contract_renderer_evidence",
        "fresh_scene_ref":"test_replay_parallel_contract:contract_fresh_scene"}


class ReplayDispatchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.library = SimpleNamespace(root=self.root / "unused-library")

    def tearDown(self):
        self.temporary.cleanup()

    def test_default_keeps_original_serial_path_without_resource_requirement(self):
        loop = MaintenanceLoop(self.library, None, None, None, {}, [], replay_parallelism=1)
        loop.replay = Mock(side_effect=lambda case, candidate, output:case["id"])
        self.assertEqual(loop.replay_many([{"id":"a"}, {"id":"b"}], {}, self.root), ["a", "b"])
        self.assertEqual([call.args[0]["id"] for call in loop.replay.call_args_list], ["a", "b"])
        self.assertFalse((self.root / "tokens").exists())

    def test_bad_parallelism_and_missing_resources_rejected(self):
        for value in (0, 17, True, 2.0):
            with self.subTest(value=value), self.assertRaises(ValueError):
                MaintenanceLoop(self.library, None, None, None, {}, [], replay_parallelism=value)
        with self.assertRaises(ValueError):
            MaintenanceLoop(self.library, None, None, None, {}, [], replay_parallelism=2)

    def test_invalid_schema_refs_or_slots_fail_before_any_factory(self):
        for change in ({"slots_per_gpu":3}, {"unknown":1}, {"factory_ref":"missing_fixture:factory"},
                       {"token_root":"relative-tokens"}, {"gpus":[{"uuid":"GPU-X"}, {"uuid":"GPU-X"}]}):
            resources = resource_fixture(self.root)
            resources.update(change)
            with self.subTest(change=change), self.assertRaises((ValueError, ModuleNotFoundError)):
                validate_replay_configuration(resources, self.library, None, None)
        self.assertFalse((self.root / "cases").exists())

    def test_all_workers_must_agree_on_shared_token_inventory(self):
        resources = resource_fixture(self.root)
        _resources(resources, self.library)
        _resources(resources, self.library)
        with self.assertRaisesRegex(ValueError, "declaration differs"):
            _resources(dict(resources, slots_per_gpu=2), self.library)

    def test_supervisor_inventory_is_eight_cards_in_same_explicit_run(self):
        inventory = self.root / "devices.json"
        data = {"node":"fixture-only", "gpu_requested":8,
                "devices":[{"uuid":"GPU-"+str(i), "name":"OFFLINE NVIDIA FIXTURE"} for i in range(8)]}
        inventory.write_text(json.dumps(data))
        resources = {"token_root":str(self.root / "tokens"), "inventory_path":str(inventory), "slots_per_gpu":2}
        _, gpus, slots = _resources(resources, self.library)
        self.assertEqual((len(gpus), slots), (8, 2))
        data["devices"][0]["name"] = "changed inventory"
        inventory.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "declaration differs"):
            _resources(resources, self.library)
        with self.assertRaisesRegex(ValueError, "same explicit run"):
            _resources(dict(resources, token_root=str(self.root / "other" / "tokens")), self.library)

    def test_tokens_are_shared_across_processes_and_bound_each_gpu(self):
        token_root = self.root / "tokens"
        token_root.mkdir()
        context = multiprocessing.get_context("spawn")
        queue = context.Queue()
        processes = [context.Process(target=slot_probe, args=(str(token_root), queue)) for _ in range(5)]
        for process in processes:
            process.start()
        events = [queue.get(timeout=10) for _ in range(10)]
        for process in processes:
            process.join(timeout=10)
            self.assertEqual(process.exitcode, 0)
        active = {"GPU-ONE":0, "GPU-TWO":0}
        largest = 0
        for _, event, uuid, _ in sorted(events):
            active[uuid] += 1 if event == "enter" else -1
            self.assertTrue(0 <= active[uuid] <= 1)
            largest = max(largest, sum(active.values()))
        self.assertLessEqual(largest, 2)
        self.assertEqual(sum(active.values()), 0)

    def test_child_inherited_fd_holds_slot_after_parent_handle_closes(self):
        token_root = self.root / "tokens"
        token_root.mkdir()
        with replay_slot(token_root, [{"uuid":"GPU-ONE"}], 1) as (handle, _, _, path):
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(.3)"],
                                     pass_fds=(handle.fileno(),))
        try:
            with path.open("a+") as handle:
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            child.wait(timeout=5)
            with path.open("a+") as handle:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()

    def test_first_eight_cases_prefer_one_slot_on_each_gpu(self):
        token_root = self.root / "tokens"
        token_root.mkdir()
        gpus = [{"uuid":"GPU-"+str(i)} for i in range(8)]
        with ExitStack() as stack:
            assignments = [stack.enter_context(replay_slot(token_root, gpus, 2, i))[1] for i in range(8)]
            self.assertEqual({a["uuid"] for a in assignments}, {g["uuid"] for g in gpus})
            self.assertTrue(all(a["replay_slot"] == 0 for a in assignments))
            ninth = stack.enter_context(replay_slot(token_root, gpus, 2, 8))[1]
            self.assertEqual(ninth["replay_slot"], 1)

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux parent-death contract")
    def test_owner_kill_terminates_child_and_releases_inherited_token(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parent.parent)])
        owner = subprocess.Popen([sys.executable, "-c",
            "from test_replay_dispatch import parent_death_owner; parent_death_owner("+repr(str(self.root))+")"],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        marker = self.root / "cases/death-case/simulation/slow_factory_started.json"
        child_pid = None
        try:
            deadline = time.monotonic()+10
            while not marker.exists() and time.monotonic() < deadline:
                if owner.poll() is not None:
                    self.fail("Fixture owner exited before its child-start marker")
                time.sleep(.02)
            self.assertTrue(marker.exists())
            child_pid = json.loads(marker.read_text())["pid"]
            self.assertEqual(os.getpgid(owner.pid), os.getpgid(child_pid))
            owner.kill()
            owner.wait(timeout=5)
            token = self.root / "tokens/GPU-ONE_0.lock"
            released = False
            deadline = time.monotonic()+5
            while time.monotonic() < deadline:
                with token.open("a+") as handle:
                    try:
                        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        released = True
                    except BlockingIOError:
                        pass
                if released:
                    break
                time.sleep(.02)
            self.assertTrue(released, "Dead owner left a live replay/token behind")
            status = Path('/proc') / str(child_pid) / 'stat'
            self.assertTrue(not status.exists() or status.read_text().rsplit(')', 1)[1].split()[0] == 'Z')
            self.assertFalse((self.root / "cases/death-case/report.json").exists())
        finally:
            if owner.poll() is None:
                owner.kill()
                owner.wait()
            if child_pid is not None:
                try:
                    os.kill(child_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


if __name__ == "__main__":
    unittest.main()

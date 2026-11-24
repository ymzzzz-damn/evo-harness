"""Independent offline contracts for subprocess case replay.

The importable one-joint plant is a fixture; no RoboCasa environment/GPU/model
is started by these tests and its reports are never benchmark evidence.
"""
from pathlib import Path
import hashlib
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from evo.runtime_api import RuntimeAPI


class ContractAdapter:
    """Independent host verifier over a measured fixture joint."""
    def __init__(self, config, output):
        self.config = dict(config)
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.position = float(config.get('initial', 0))
        self.goal = float(config.get('goal', 999))
        self.steps = 0
        self.horizon = int(config.get('horizon', 100))
        self.closed = False
        self._write('fixture_process.json', {
            'fixture_only': True, 'pid': os.getpid(), 'ppid': os.getppid(),
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'roboenv_binding': json.loads(os.environ.get('ROBOENV_GPU', '{}')),
            'mujoco_egl_device_id': os.environ.get('MUJOCO_EGL_DEVICE_ID'),
        })

    def _write(self, name, data):
        (self.output / name).write_text(json.dumps(data, sort_keys=True) + '\n')

    def move_joints(self, target, duration=2.0, **kwargs):
        if self.steps >= self.horizon:
            raise RuntimeError('Fixture horizon exhausted')
        self.steps += 1
        self.position = float(target[0])
        if kwargs.get('raise_after_motion'):
            raise ValueError('Fixture controlled motion exception')
        return {'position': self.position}

    def wait(self, duration=.5):
        self.steps += 1
        return {'step': self.steps}

    def observe(self):
        return {'fixture_only': True, 'step': self.steps, 'horizon': self.horizon,
                'joint': self.position, 'images': []}

    def native_success(self):
        return abs(self.position - self.goal) < 1e-9

    def evaluate(self, predicate):
        if predicate['predicate'] == 'native_success':
            success = self.native_success()
        elif predicate['predicate'] == 'joint_near':
            success = abs(self.position - float(predicate['args']['target'][0])) < 1e-9
        else:
            return {'known': False, 'success': False}
        return {'known': True, 'success': success, 'measured': self.position}

    def stats(self):
        return {'task': self.config.get('task', 'Fixture'), 'seed': self.config.get('seed', 0),
                'steps': self.steps, 'horizon': self.horizon,
                'native_success': self.native_success(),
                'terminal': self.steps >= self.horizon or self.native_success()}

    def close(self):
        self.closed = True
        self._write('fixture_close.json', {'fixture_only': True, 'pid': os.getpid(),
                                         'steps': self.steps})


def contract_factory(config, output):
    return ContractAdapter(config, output)


def contract_renderer_mapping():
    """Assert stale parent GPU bindings were removed before fresh bootstrap."""
    if os.environ.get('ROBOENV_GPU') is not None:
        raise RuntimeError('Fixture found stale parent ROBOENV_GPU')
    if os.environ.get('MUJOCO_EGL_DEVICE_ID') is not None:
        raise RuntimeError('Fixture found stale parent EGL selection')
    uuid = os.environ['CUDA_VISIBLE_DEVICES']
    os.environ['ROBOENV_GPU'] = json.dumps({'uuid': uuid, 'egl_index': 0, 'cuda_index': 0})
    os.environ['MUJOCO_EGL_DEVICE_ID'] = '0'


def contract_renderer_evidence(assignment):
    binding = json.loads(os.environ['ROBOENV_GPU'])
    if binding['uuid'] != assignment['uuid']:
        raise RuntimeError('Fixture renderer UUID mismatch')
    return {'fixture_only': True, 'assignment': assignment, 'roboenv_binding': binding,
            'gl_vendor': 'NVIDIA CONTRACT FIXTURE', 'gl_renderer': 'OFFLINE FIXTURE',
            'gl_version': 'OFFLINE FIXTURE', 'gl_errors_before_diagnostic': []}


def contract_fresh_scene(adapter):
    if adapter.native_success() or adapter.stats()['steps'] != 0:
        raise RuntimeError('Fixture fresh scene is not a zero-step native-false reset')
    return {'raw_native_success_at_reset': False, 'native_steps_at_reset': 0}


GOOD = 'def skill(api, params):\n    api.move_joints([params["target"]])\n    return {"motion": "attempted"}\n'
CLAIM_ONLY = 'def skill(api, params):\n    return {"success": True, "target": params["target"]}\n'


def contract_step(target=2, atom='A02'):
    return {'id': 'target', 'atom': atom, 'params': {'target': target},
            'postcondition': {'predicate': 'joint_near', 'args': {'target': [target]}}}


def contract_case(identifier, target=2, **kwargs):
    case = {'id': identifier, 'atom': 'A02',
            'task_config': {'task': 'Fixture', 'seed': 0, 'goal': 999, 'horizon': 100},
            'prefix': [], 'suffix': [], 'step': contract_step(target),
            'require_native_success': False}
    case.update(kwargs)
    return case


class ExactFixtureLibrary:
    """Parent-only immutable published-version source, with no active fallback."""
    def __init__(self, root, records=()):
        self.root = Path(root)
        self.records = {(r['atom'], r['version']): dict(r) for r in records}
        self.lookups = []

    def get_version(self, atom, version, sha256=None):
        self.lookups.append((atom, version))
        record = self.records[(atom, version)]
        if sha256 is not None and record['sha256'] != sha256:
            raise ValueError('Fixture dependency SHA mismatch')
        return dict(record)


def contract_candidate(code=GOOD, dependencies=None):
    return {'atom': 'A02', 'id': 'fixture-candidate', 'code': code,
            'sha256': hashlib.sha256(code.encode()).hexdigest(),
            'dependencies': dict(dependencies or {})}


def contract_version(atom, version, code, dependencies=None):
    return {'atom': atom, 'version': version, 'code': code,
            'sha256': hashlib.sha256(code.encode()).hexdigest(),
            'dependencies': dict(dependencies or {})}


class ReplayParallelContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.sequence = 0

    def tearDown(self):
        self.temporary.cleanup()

    def destination(self, label):
        self.sequence += 1
        return self.root / f'{self.sequence}_{label}'

    def resources(self):
        return {'token_root': str(self.root / 'tokens'), 'slots_per_gpu': 2,
                'gpus': [{'uuid': 'GPU-CONTRACT-A'}, {'uuid': 'GPU-CONTRACT-B'}],
                'factory_ref': 'test_replay_parallel_contract:contract_factory',
                'runtime_ref': 'evo.runtime_api:RuntimeAPI',
                'renderer_mapping_ref': 'test_replay_parallel_contract:contract_renderer_mapping',
                'renderer_evidence_ref': 'test_replay_parallel_contract:contract_renderer_evidence',
                'fresh_scene_ref': 'test_replay_parallel_contract:contract_fresh_scene'}

    def loop(self, *, parallel=True, library=None, resources=None):
        from evo.loops import MaintenanceLoop
        return MaintenanceLoop(library or ExactFixtureLibrary(self.root / 'library'), None,
                               contract_factory, RuntimeAPI, {}, ['joint_near', 'native_success'],
                               max_candidates=2, code_timeout_s=2,
                               replay_parallelism=2 if parallel else 1,
                               replay_resources=self.resources() if resources is None else resources)

    def test_every_case_joined_in_original_order_without_relaxing_native_requirement(self):
        """A rejected first case cannot suppress remaining independent obligations."""
        already = contract_case('first_already_true')
        already['task_config']['initial'] = 2
        native_required = contract_case('second_native_required', require_native_success=True)
        local_only = contract_case('third_local_only')
        cases = [already, native_required, local_only]
        candidate = contract_candidate()
        serial_dir, parallel_dir = self.destination('serial'), self.destination('parallel')
        serial = self.loop(parallel=False).replay_many(cases, candidate, serial_dir)
        parallel = self.loop().replay_many(cases, candidate, parallel_dir)
        self.assertEqual([r['case_id'] for r in parallel], [c['id'] for c in cases])
        self.assertEqual([r['success'] for r in serial], [False, False, True])
        self.assertEqual([r['success'] for r in parallel], [False, False, True])
        self.assertTrue(parallel[1]['atomic_success'])
        self.assertFalse(parallel[1]['native_success'])
        self.assertIn('already true', parallel[0]['error'])
        for case in cases:
            self.assertTrue((parallel_dir / case['id'] / 'simulation/fixture_close.json').is_file())
            self.assertTrue((parallel_dir / case['id'] / 'report.json').is_file())
        batch = json.loads((parallel_dir / 'replay_dispatch_summary.json').read_text())
        self.assertTrue(batch['all_cases_joined'])
        self.assertTrue(batch['results_in_original_order'])

    def test_entry_trace_preserves_failed_prefix_instead_of_substituting_candidate(self):
        bad_preparation = 'def skill(api, params):\n    api.move_joints([params["target"] / 2])\n    return {"motion": "attempted"}\n'
        prior = {'step': contract_step(), 'code': bad_preparation, 'dependencies': {},
                 'sha256': hashlib.sha256(bad_preparation.encode()).hexdigest(),
                 'success': False, 'execution': {'ok': True},
                 'verification': {'known': True, 'success': False}}
        case = contract_case('retained_failed_preparation', entry_trace=[prior])
        candidate = contract_candidate()
        serial = self.loop(parallel=False).replay_many([case], candidate, self.destination('serial'))[0]
        result = self.loop().replay_many([case], candidate, self.destination('parallel'))[0]
        self.assertTrue(serial['success'])
        self.assertTrue(result['success'])
        self.assertFalse(result['prefix'][0]['verification']['success'])
        self.assertEqual(result['prefix'][0]['execution']['sha256'], prior['sha256'])
        self.assertEqual(result['execution']['sha256'], candidate['sha256'])
        self.assertEqual(result['stats']['steps'], 2)
        self.assertFalse(result['native_success'])

    def test_child_fresh_interpreter_rebinds_gpu_and_keeps_proof_per_case(self):
        output = self.destination('parallel')
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': 'GPU-PARENT',
                                     'ROBOENV_GPU': '{"uuid":"GPU-PARENT"}',
                                     'MUJOCO_EGL_DEVICE_ID': '99'}):
            results = self.loop().replay_many([contract_case('case_a'), contract_case('case_b')],
                                              contract_candidate(), output)
        self.assertTrue(all(r['success'] for r in results))
        pids = set()
        for identifier in ('case_a', 'case_b'):
            directory = output / identifier
            evidence = json.loads((directory / 'gpu_binding.json').read_text())
            fixture = json.loads((directory / 'simulation/fixture_process.json').read_text())
            assignment = json.loads((directory / 'gpu_assignment.json').read_text())
            pids.add(fixture['pid'])
            self.assertNotEqual(fixture['pid'], os.getpid())
            self.assertEqual(fixture['pid'], evidence['child_pid'])
            self.assertEqual(evidence['replay_case_id'], identifier)
            self.assertEqual(evidence['simulation'], str(directory / 'simulation'))
            self.assertEqual(fixture['cuda_visible_devices'], assignment['uuid'])
            self.assertEqual(fixture['roboenv_binding']['uuid'], assignment['uuid'])
            self.assertFalse(evidence['raw_native_success_at_reset'])
            self.assertEqual(evidence['native_steps_at_reset'], 0)
        self.assertEqual(len(pids), 2)

    def test_dependency_freeze_uses_old_exact_recursive_pins_not_latest_active(self):
        leaf = contract_version('A05', 1, GOOD)
        leaf_pin = {'version': 1, 'sha256': leaf['sha256']}
        middle_code = 'def skill(api, params):\n    return api.call("A05", params)\n'
        middle = contract_version('A04', 1, middle_code, {'A05': leaf_pin})
        later_middle = contract_version('A04', 2, CLAIM_ONLY)
        library = ExactFixtureLibrary(self.root / 'library', [leaf, middle, later_middle])
        caller = 'def skill(api, params):\n    return api.call("A04", params)\n'
        candidate = contract_candidate(caller, {'A04': {'version': 1, 'sha256': middle['sha256']}})
        output = self.destination('recursive_pin')
        result = self.loop(library=library).replay_many([contract_case('recursive')], candidate, output)[0]
        self.assertTrue(result['success'])
        self.assertEqual(result['stats']['steps'], 1)
        self.assertEqual(set(library.lookups), {('A04', 1), ('A05', 1)})
        payload = json.loads((output / 'recursive/dispatch_input.json').read_text())
        self.assertEqual({(r['atom'], r['version']) for r in payload['dependency_records']},
                         {('A04', 1), ('A05', 1)})
        self.assertFalse((library.root / 'state.json').exists())
        self.assertFalse((library.root / '.lock').exists())

    def test_claimed_success_cannot_replace_independent_host_verification(self):
        output = self.destination('claimed_success')
        result = self.loop().replay_many([contract_case('claim_only')],
                                         contract_candidate(CLAIM_ONLY), output)[0]
        self.assertTrue(result['execution']['ok'])
        self.assertTrue(result['execution']['result']['success'])
        self.assertFalse(result['verification']['success'])
        self.assertFalse(result['success'])
        self.assertEqual(result['stats']['steps'], 0)

    def test_corrupt_recursive_pin_fails_closed_without_changing_other_case_coverage(self):
        from evo.replay_dispatch import FrozenVersionResolver, freeze_dependencies
        leaf = contract_version('A05', 1, GOOD)
        corrupt = dict(leaf, code=CLAIM_ONLY)
        with self.assertRaises(ValueError):
            FrozenVersionResolver([corrupt])
        middle = contract_version('A04', 1, 'def skill(api, params):\n    return api.call("A05", params)\n',
                                  {'A05': {'version': 1, 'sha256': '0' * 64}})
        with self.assertRaises(ValueError):
            FrozenVersionResolver([middle, leaf])
        bad_candidate = contract_candidate('def skill(api, params):\n    return api.call("A05", params)\n',
                                           {'A05': {'version': 1, 'sha256': '0' * 64}})
        library = ExactFixtureLibrary(self.root / 'library', [leaf])
        with self.assertRaises(ValueError):
            freeze_dependencies(contract_case('bad_pin'), bad_candidate, library)

    def test_existing_case_refusal_preserves_all_original_artifacts_and_starts_no_child(self):
        output = self.destination('existing')
        existing = output / 'case_b'
        existing.mkdir(parents=True)
        for name in ('report.json', 'dispatch.json', 'dispatch_result.json'):
            (existing / name).write_bytes(b'original-evidence-' + name.encode())
        before = {p.relative_to(output).as_posix(): p.read_bytes()
                  for p in output.rglob('*') if p.is_file()}
        loop = self.loop()
        with patch('evo.replay_dispatch.subprocess.Popen') as spawn:
            with self.assertRaises(ValueError):
                loop.replay_many([contract_case('case_a'), contract_case('case_b')],
                                 contract_candidate(), output)
        spawn.assert_not_called()
        self.assertFalse((output / 'case_a').exists())
        self.assertEqual(before, {p.relative_to(output).as_posix(): p.read_bytes()
                                  for p in output.rglob('*') if p.is_file()})

    def test_bad_case_dependency_does_not_omit_remaining_good_case(self):
        bad = contract_case('first_missing_pin')
        caller = 'def skill(api, params):\n    return api.call("A04", params)\n'
        bad['prefix'] = [{'step': contract_step(atom='A04'), 'code': caller,
                          'dependencies': {'A04': {'version': 999, 'sha256': '0' * 64}},
                          'success': True}]
        output = self.destination('per_case_failure')
        results = self.loop().replay_many([bad, contract_case('second_good')],
                                          contract_candidate(), output)
        self.assertEqual([r['case_id'] for r in results], ['first_missing_pin', 'second_good'])
        self.assertFalse(results[0]['success'])
        self.assertEqual(results[0]['failure_stage'], 'dispatch')
        self.assertTrue(results[1]['success'])
        self.assertTrue((output / 'first_missing_pin/report.json').is_file())
        self.assertTrue((output / 'second_good/simulation/fixture_close.json').is_file())


if __name__ == '__main__':
    unittest.main()

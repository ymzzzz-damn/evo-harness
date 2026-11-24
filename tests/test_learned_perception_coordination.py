"""Executable P/C lifecycle on a toy plant; no RoboCasa task-success claim."""
import copy

import numpy as np
import pytest
from PIL import Image

from evo.agent import Agent, ScriptedProvider
from evo.codecheck import check_code
from evo.execution import execute_skill
from evo.library import Library
from evo.loops import validate_plan, SkillExecutor
from evo.policy_evaluation import FrozenLibrary, freeze_active_library
from evo.result_verification import evaluate_step
from evo.robocasa_adapter import RoboCasaAdapter
from evo.runtime_api import RuntimeAPI
from test_evolution import Harness, failed_consolidation
from test_loops import FakeAdapter, GOOD, candidate, plan, step


class SensorPlant(FakeAdapter):
    def observe(self):
        return dict(super().observe(), semantic_catalog={'object_ids': ['cup', 'bowl']})

    def deproject(self, camera, u, v, radius=1):
        # Deliberately simple observable geometry for orchestration tests.
        return [self.position + u, v, 1.0]


GEOMETRY = ('def skill(api, params):\n'
            '    point = api.deproject(params["camera"], params["u"], params["v"])\n'
            '    return {"point": point}\n')
ASSIGN = ('def skill(api, params):\n'
          '    remaining = list(params["objects"])\n'
          '    assigned = []\n'
          '    for dest, quota in params["quotas"].items():\n'
          '        for index in range(quota):\n'
          '            assigned.append({"object": remaining.pop(0), "destination": dest})\n'
          '    return {"assignments": assigned, "remaining": remaining}\n')


def example(atom):
    action = step(atom=atom)
    if atom == 'P01':
        code = 'def skill(api, params):\n    return {"ids": api.observe()["semantic_catalog"][params["field"]]}\n'
        action['params'] = {'field': 'object_ids'}
        action['postcondition'] = {'predicate': 'observed_result', 'args': {'checks': [
            {'result_path': ['ids'], 'source': 'observation', 'path': ['semantic_catalog', 'object_ids']}]}}
    elif atom == 'P02':
        code = GEOMETRY
        action['params'] = {'camera': 'camera_left', 'u': 2, 'v': 3}
        action['postcondition'] = {'predicate': 'observed_result', 'args': {'checks': [
            {'result_path': ['point'], 'source': 'deproject', **action['params']}]}}
    elif atom == 'P03':
        code = 'def skill(api, params):\n    return {"effect": abs(api.observe()["joint"] - params["target"]) < 0.01}\n'
        action['postcondition'] = {'predicate': 'effect_report', 'args': {
            'result_path': ['effect'], 'condition': {'predicate': 'joint_near', 'args': {'target': [2]}}}}
    elif atom == 'C01':
        code = ASSIGN
        action['params'] = {'objects': ['cup', 'bowl'], 'quotas': {'shelf': 1}}
        action['postcondition'] = {'predicate': 'assignment_valid', 'args': copy.deepcopy(action['params'])}
    else:
        code = GOOD  # Toy physical effect for C02/C03 execution plumbing only.
    action['goal'] = 'Verify the declared toy output or physical effect; no general recognition claim'
    return action, code


@pytest.mark.parametrize('atom', ['P01', 'P02', 'P03', 'C01', 'C02', 'C03'])
def test_all_six_learn_replay_publish_freeze_and_reuse(tmp_path, atom):
    h = Harness(tmp_path)
    h.factory = SensorPlant
    action, code = example(atom)
    response = candidate(code)
    result = h.explore({'plan': [plan(action)], 'repair': [response]}, max_plan_segments=1)
    assert len(result['pending']) == 1 and not result['failures'], result
    record = result['completed_steps'][0]
    assert record['executed'] and not record.get('skipped')
    if atom in ('P01', 'P02', 'P03', 'C01'):
        assert not result['native_success']  # Partial learning survives task failure.
        assert record['verification']['scope'] == 'declared_output_contract'
        assert record['execution']['after']['steps'] == 0  # Pure code really ran.
    report = h.maintain(result, {'distill': [response]})
    assert report['promoted'] and report['replay_counts']['fresh'] == 1, report
    assert h.library.active(atom)['code'] == code
    reused = h.explore({'plan': [plan(action)]}, max_plan_segments=1)
    assert reused['completed_steps'][0]['reused']
    assert [r['stage'] for r in h.provider.requests] == ['plan']
    frozen_root = tmp_path / 'frozen'
    receipt = freeze_active_library(h.library.root, frozen_root)
    frozen = FrozenLibrary(frozen_root, receipt['state_sha256'])
    scratch = frozen.scratch(tmp_path / 'scratch')
    assert scratch.active(atom)['code'] == code
    # P/C sources can also be called through the same pinned dependency path.
    wrapper = h.library.propose('A25', f'def skill(api, params):\n    return api.call("{atom}", params)\n', dependencies=[atom])
    execution = SkillExecutor(SensorPlant({}, tmp_path), scratch, RuntimeAPI, 2).run(
        wrapper['code'], action['params'], wrapper['dependencies'])
    assert execution['ok'], execution


def test_cross_layer_code_uses_geometry_output_to_execute_published_motion(tmp_path):
    h = Harness(tmp_path)
    h.factory = SensorPlant
    h.learn()
    action, code = example('P02')
    result = h.explore({'plan': [plan(action)], 'repair': [candidate(code)]}, max_plan_segments=1)
    assert h.maintain(result, {'distill': [candidate(code)]})['promoted']
    orchestration = ('def skill(api, params):\n'
                     '    geometry = api.call("P02", params["geometry"])\n'
                     '    return api.call("A02", {"target": geometry["point"][0]})\n')
    action = step(atom='C03', target=4)
    action['params'] = {'geometry': {'camera': 'camera_left', 'u': 4, 'v': 0}}
    response = dict(candidate(orchestration), dependencies=['P02', 'A02'])
    result = h.explore({'plan': [plan(action)], 'repair': [response]}, goal=4)
    assert result['native_success']
    report = h.maintain(result, {'distill': [response]})
    assert report['promoted'] and set(h.library.active('C03')['dependencies']) == {'P02', 'A02'}


def test_perception_result_reaches_next_plan_and_is_not_task_success(tmp_path):
    h = Harness(tmp_path)
    h.factory = SensorPlant
    action, code = example('P02')
    result = h.explore({'plan': [dict(plan(action), plan_complete=False), plan(step(identifier='physical_move'))],
                        'repair': [candidate(code), candidate(GOOD)]})
    assert result['native_success'] and len(result['completed_steps']) == 2
    continuation = [r for r in h.provider.requests if r['stage'] == 'plan'][1]['context']
    assert continuation['completed_steps'][0]['execution']['result']['point'] == [2, 3, 1.0]


@pytest.mark.parametrize('atom', ['P01', 'P02', 'P03', 'C01'])
def test_self_claim_or_wrong_output_cannot_publish(tmp_path, atom):
    h = Harness(tmp_path)
    h.factory = SensorPlant
    action, _ = example(atom)
    code = 'def skill(api, params):\n    return {"status": "success", "claimed": params}\n'
    result = h.explore({'plan': [plan(action)], 'repair': [candidate(code)]}, max_plan_segments=1)
    assert not result['pending'] and result['failures']
    draft = failed_consolidation()
    draft.update(code=code, params=action['params'])
    outcome = h.maintain(result, {'consolidate': [draft]})
    assert outcome['drafts'] and h.library.active(atom) is None
    memory = h.library.working_memory(atom)[0]['payload']
    assert memory['parameterized'] and memory['failure_analysis'] and memory['failure_feedback']


def test_incorrect_distillation_fails_independent_output_replay(tmp_path):
    h = Harness(tmp_path)
    h.factory = SensorPlant
    action, code = example('P02')
    result = h.explore({'plan': [plan(action)], 'repair': [candidate(code)]}, max_plan_segments=1)
    wrong = 'def skill(api, params):\n    return {"point": [params["u"], 999, 1.0]}\n'
    report = h.maintain(result, {'distill': [candidate(wrong)]})
    assert report['drafts'] and not report['promoted'] and h.library.active('P02') is None


def test_data_conditions_cannot_replace_action_effects_or_hidden_success():
    action, _ = example('P02')
    action['atom'] = 'A02'
    with pytest.raises(ValueError, match='valid verifier'):
        validate_plan(plan(action), ['native_success'])
    action, _ = example('P03')
    action['postcondition']['args']['condition']['predicate'] = 'native_success'
    with pytest.raises(ValueError, match='local physical'):
        validate_plan(plan(action), ['native_success'])


@pytest.mark.parametrize('returned', [
    {'assignments': [{'object': 'cup', 'destination': 'shelf'}, {'object': 'cup', 'destination': 'shelf'}], 'remaining': []},
    {'assignments': [{'object': 'alien', 'destination': 'shelf'}], 'remaining': ['cup', 'bowl']},
    {'assignments': [{'object': 'cup', 'destination': 'shelf'}], 'remaining': []},
    {'assignments': [], 'remaining': ['cup', 'bowl']},
])
def test_assignment_rejects_duplicates_foreign_objects_wrong_remaining_and_quota(tmp_path, returned):
    action, _ = example('C01')
    assert not evaluate_step(SensorPlant({}, tmp_path), action, {'ok': True, 'result': returned})['success']


def test_camera_arrays_are_current_detached_and_usable_by_generated_code(tmp_path):
    rgb_path, depth_path = tmp_path / 'frame.png', tmp_path / 'depth.npy'
    Image.fromarray(np.full((3, 4, 3), 60, dtype=np.uint8)).save(rgb_path)
    np.save(depth_path, np.ones((3, 4)), allow_pickle=False)
    metadata = {'image_path': str(rgb_path), 'depth_path': str(depth_path), 'T_base_camera': np.eye(4).tolist()}
    item = RoboCasaAdapter.__new__(RoboCasaAdapter)
    item.observe = lambda: {'step': 7, 'cameras': {'robot0_eye_in_hand': metadata}}
    api = RuntimeAPI(item)
    code = ('def skill(api, params):\n'
            '    frame = api.read_camera(params["camera"])\n'
            '    return {"mean": float(np.mean(frame["rgb"])), "step": frame["step"]}\n')
    assert check_code(code, 'P01')['ok']
    result = execute_skill(code, api, {'camera': 'camera_wrist'}, 2)
    assert result['ok'] and result['result'] == {'mean': 60.0, 'step': 7}
    frame = api.read_camera('camera_wrist')
    frame['rgb'][:] = 0
    frame['depth'][:] = 0
    frame['calibration']['T_base_camera'][0][0] = 999
    assert api.read_camera('camera_wrist')['rgb'].mean() == 60
    assert api.read_camera('camera_wrist')['depth'].mean() == 1
    assert metadata['T_base_camera'][0][0] == 1
    with pytest.raises(ValueError, match='not enabled'):
        api.read_camera(str(rgb_path))  # No arbitrary-file API.

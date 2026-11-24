"""Human300 layers, offline coverage and runtime authority boundaries."""
import json
from collections import Counter
from pathlib import Path

import pytest

from evo.agent import Agent, ResponseError, ScriptedProvider, validate_plan
from evo.catalog import (ACTION_CATALOG, CATALOG, CATALOG_VERSION, PERCEPTION_CATALOG, COORDINATION_CATALOG,
                         get_spec, skill_context, task_catalog)
from evo.cli import main
from evo.codecheck import check_code
from evo.library import Library, LibraryError
from evo.loops import validate_plan as validate_host_plan
from evo.policy_evaluation import FrozenLibrary
from evo.prompts import instructions
from test_evolution import Harness
from test_loops import FakeAdapter, GOOD, candidate, plan, step


EXPECTED_NAMES = {
    'P01': 'ground_targets', 'P02': 'estimate_geometry', 'P03': 'observe_effect',
    'A01': 'navigate_align', 'A02': 'grasp_lift', 'A03': 'release_retreat',
    'A04': 'place_on', 'A05': 'place_in', 'A06': 'place_pose', 'A07': 'insert_dock',
    'A08': 'stack_nest', 'A09': 'carry_preserve', 'A10': 'operate_hinged',
    'A11': 'slide_joint', 'A12': 'detach_seat_lid', 'A13': 'press_button',
    'A14': 'toggle_lever', 'A15': 'turn_control', 'A16': 'sweep_spout',
    'A17': 'tilt_preserve', 'A18': 'dump_contents', 'A19': 'shake_container',
    'A20': 'press_contact', 'A21': 'scrub', 'A22': 'stir', 'A23': 'rinse_hold',
    'A24': 'fill_under_flow', 'A25': 'wait_until', 'A26': 'push_slide',
    'C01': 'assign_objects', 'C02': 'clear_space', 'C03': 'execute_skill_graph',
}


def test_catalog_matches_selected_table_and_training_requirements():
    assert {k: v['name'] for k, v in CATALOG.items()} == EXPECTED_NAMES
    assert Counter(s['layer'] for s in CATALOG.values()) == {'perception': 3, 'action': 26, 'coordination': 3}
    path = Path(__file__).resolve().parents[1] / 'configs/human300_capability_requirements.json'
    audit = json.loads(path.read_text())
    assert audit['catalog_version'] == CATALOG_VERSION
    tasks = audit['tasks']
    assert len(tasks) == len({t['task'] for t in tasks}) == 300
    assert Counter(t['kind'] for t in tasks) == {'atomic': 65, 'composite': 235}
    seen = set()
    for task in tasks:
        ids = set(task['capability_ids'])
        assert ids == {r['capability'] for r in task['requirements']}
        assert ids <= set(CATALOG) and ids & set(ACTION_CATALOG)
        assert {r['mode'] for r in task['requirements']} <= {'core', 'conditional', 'support'}
        seen |= ids
    assert seen == set(CATALOG)
    training = {t['task'] for t in tasks}
    for spec in CATALOG.values():
        assert set(spec['source_training_examples']) <= training
        assert spec['implemented'] is False
        assert spec['contract']['effects'] in spec['description']
        assert spec['contract']['bounded_failure_recovery'] in spec['description']


def test_plan_gets_all_executable_interfaces_but_no_offline_task_solutions(tmp_path):
    library = Library(tmp_path / 'library')
    context = task_catalog(library)
    assert len(context['atomic_catalog']) == 32
    assert set(context['catalog_document']['executable_atoms']) == set(CATALOG)
    assert context['robot_contract']['manipulation_arms'] == 1
    for atom in CATALOG:
        assert context['atomic_catalog'][atom]['execution_mode'] == 'learned_code'
        assert Path(context['atomic_catalog'][atom]['stub_path']).is_file()
        assert skill_context(library, atom)['atomic_skill']['description'] == get_spec(atom)['description']
    assert 'tasks' not in context and 'offline_goal_predicates' not in json.dumps(context)
    assert 'instruction_templates' not in json.dumps(context)
    for stage in ('plan', 'repair', 'distill', 'consolidate'):
        prompt = instructions(stage)
        assert 'ALL 32' in prompt and 'one arm, one gripper' in prompt
        assert 'A17 and A19 include return to target support' in prompt


@pytest.mark.parametrize('atom', ['P01', 'P02', 'P03', 'C01', 'C02', 'C03'])
def test_every_layer_can_be_proposed_planned_and_called(tmp_path, atom):
    library = Library(tmp_path / 'library')
    assert library.propose(atom, GOOD)['atom'] == atom
    assert skill_context(library, atom)['atomic_skill']['entrypoint'] == 'skill(api, params) -> dict'
    assert validate_plan(plan(step(atom=atom)))['steps'][0]['atom'] == atom
    assert validate_host_plan(plan(step(atom=atom)), ['native_success'])['steps'][0]['atom'] == atom
    assert check_code(f'def skill(api, params):\n    return api.call("{atom}", params)\n', 'A25')['ok']


@pytest.mark.parametrize('old_version,old_atom', [('robocasa-r20-v1', 'R02'), ('p40-v1', 'P01')])
def test_legacy_code_state_is_rejected_by_live_and_frozen_readers(tmp_path, old_version, old_atom):
    import hashlib
    library = Library(tmp_path / 'library')
    state = json.loads(library.state_path.read_text())
    state['atomic_catalog_version'] = old_version
    state['lessons'] = {old_atom: []}
    library.state_path.write_text(Library._state_text(state))
    before = library.state_path.read_bytes()
    with pytest.raises(LibraryError, match='Incompatible'):
        Library(library.root)
    with pytest.raises(LibraryError, match='Incompatible'):
        FrozenLibrary(library.root, hashlib.sha256(before).hexdigest())
    assert library.state_path.read_bytes() == before


def test_cli_reports_definitions_separately_from_executable_functions(tmp_path, capsys):
    main(['init', '--library', str(tmp_path / 'library')])
    result = json.loads(capsys.readouterr().out)
    assert result['defined_atoms'] == 32 and result['executable_atoms'] == 32
    assert result['perception_atoms'] == 3 and result['coordination_atoms'] == 3 and result['action_atoms'] == 26 and result['active'] == {}


def test_new_wait_capability_explores_and_passes_first_use_replay(tmp_path):
    class WaitPlant(FakeAdapter):
        def native_success(self):
            return self.steps >= 2
    h = Harness(tmp_path)
    h.factory = WaitPlant
    code = ('def skill(api, params):\n'
            '    for index in range(params["ticks"]):\n'
            '        api.wait(params["duration"])\n'
            '    return {"status": "unknown", "phase": "waiting_finished"}\n')
    action = step(atom='A25')
    action['params'] = {'ticks': 2, 'duration': 0.1}
    response = candidate(code)
    response['description'] = 'Toy wait: ticks bounded calls; duration seconds per call. Host determines elapsed-event success.'
    result = h.explore({'plan': [plan(action)], 'repair': [response]})
    assert result['status'] == 'success'
    request = next(r for r in h.provider.requests if r['stage'] == 'repair')
    assert request['context']['atomic_skill']['name'] == 'wait_until'
    assert set(request['context']['atomic_catalog']) == set(CATALOG)
    report = h.maintain(result, {'distill': [response]})
    assert report['promoted'] and report['replay_counts']['fresh'] == 1
    assert h.library.active('A25')['code'] == code
    # A function's own 'unknown' does not override independent host evidence.
    assert h.library.active('A25')['validation']['validation_kind'] == 'fresh_environment_replay'

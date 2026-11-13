"""Exercise NumPy's real lazy imports within the generated-code contract."""

import pytest

from evo.execution import _numpy_internal_import, execute_skill


def test_ndarray_reductions_and_linalg_in_generated_skill():
    result = execute_skill(
        '''def skill(api, params):
    points = np.array(params["points"], dtype=float)
    matrix = np.array(params["matrix"], dtype=float)
    return {
        "mean": points.mean(axis=0).tolist(),
        "std": points.std(axis=0).tolist(),
        "norm": float(np.linalg.norm(points[0])),
        "solution": np.linalg.solve(matrix, points[0]).tolist(),
        "singular_values": np.linalg.svd(matrix)[1].tolist(),
    }
''',
        None,
        {"points": [[3.0, 4.0], [5.0, 6.0]], "matrix": [[1.0, 0.0], [0.0, 1.0]]},
    )
    assert result["ok"], result
    assert result["result"] == {
        "mean": [4.0, 5.0],
        "std": [1.0, 1.0],
        "norm": 5.0,
        "solution": [3.0, 4.0],
        "singular_values": [1.0, 1.0],
    }


@pytest.mark.parametrize("statement", [
    "import os",
    "import numpy",
    "from numpy import mean",
    "module = __import__('os')",
    "module = __import__('numpy')",
    "module = np.__builtins__['__import__']('os')",
])
def test_generated_code_cannot_call_import_hook(statement):
    result = execute_skill(
        "def skill(api, params):\n    " + statement + "\n    return dict(params)\n", None, {}
    )
    assert not result["ok"]
    assert result["error"]["type"] == "CodeContractError"


@pytest.mark.parametrize("name,level", [
    ("os", 0), ("sys", 0), ("subprocess", 0), ("numpy_other", 0),
    ("numpy", 1), ("core._methods", 1),
])
def test_numpy_import_hook_rejects_foreign_and_relative_modules(name, level):
    with pytest.raises(ImportError, match="absolute NumPy"):
        _numpy_internal_import(name, level=level)


def test_numpy_import_hook_supports_numpy_internal_modules():
    import numpy as np

    assert _numpy_internal_import("numpy") is np
    assert _numpy_internal_import("numpy.linalg", fromlist=("norm",)).norm([3, 4]) == 5

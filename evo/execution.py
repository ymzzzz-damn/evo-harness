"""Bounded trusted Python execution. This is an API contract, NOT a security sandbox."""
import builtins
import contextlib
import copy
import io
import json
import math
import signal
import time
import traceback

import numpy as np

from .codecheck import check_code
from .io import sha256, jsonable


class SkillDeadlineExceeded(BaseException):
    """Not swallowed by a generated policy's ordinary except Exception recovery."""


def _numpy_internal_import(name, globals=None, locals=None, fromlist=(), level=0):
    """Support NumPy's lazy C-level imports using the caller's builtins.

    ndarray.mean(), for example, imports numpy.core._methods this way.
    Generated code still cannot use import statements or __import__: the AST
    contract rejects both. This hook is limited to absolute NumPy modules and
    does not turn the execution contract into a security sandbox.
    """
    if level != 0 or not isinstance(name, str) or not (name == "numpy" or name.startswith("numpy.")):
        raise ImportError("Only absolute NumPy internal imports are supported")
    return builtins.__import__(name, globals, locals, fromlist, level)


@contextlib.contextmanager
def deadline(seconds):
    if not hasattr(signal, "setitimer"):
        raise RuntimeError("Execution deadlines require POSIX setitimer")
    def expired(_signum, _frame):
        raise SkillDeadlineExceeded("Atomic code execution wall-clock limit reached")
    previous = signal.getsignal(signal.SIGALRM)
    old_timer = signal.getitimer(signal.ITIMER_REAL)
    entered = time.monotonic()
    signal.signal(signal.SIGALRM, expired)
    effective = min(float(seconds), old_timer[0]) if old_timer[0] else float(seconds)
    signal.setitimer(signal.ITIMER_REAL, max(0.001, effective))
    try:
        yield
    finally:
        remaining = max(0.000001, old_timer[0]-(time.monotonic()-entered)) if old_timer[0] else 0
        signal.signal(signal.SIGALRM, previous)
        signal.setitimer(signal.ITIMER_REAL, remaining, old_timer[1])


def execute_skill(code, api, params, timeout_s=90):
    started = time.monotonic()
    check = check_code(code)
    if not check["ok"]:
        return dict(ok=False, error={"type": "CodeContractError", "message": str(check["errors"])},
                    sha256=sha256(code), elapsed_s=0, stdout="")
    stream = io.StringIO()
    allowed = {name: getattr(builtins, name) for name in (
        "abs", "all", "any", "bool", "dict", "enumerate", "float", "int", "isinstance",
        "len", "list", "max", "min", "print", "range", "reversed", "round", "set", "sorted",
        "str", "sum", "tuple", "zip", "Exception", "RuntimeError", "ValueError", "TypeError")}
    allowed["__import__"] = _numpy_internal_import
    namespace = {"__builtins__": allowed, "np": np, "math": math}
    try:
        with deadline(timeout_s), contextlib.redirect_stdout(stream):
            exec(compile(code, "<atomic-skill>", "exec"), namespace)
            result = namespace["skill"](api, copy.deepcopy(params))
            if not isinstance(result, dict):
                raise TypeError("Atomic skill must return a diagnostic dictionary")
            result = jsonable(result)
            json.dumps(result, allow_nan=False)
        return dict(ok=True, result=jsonable(result), sha256=sha256(code),
                    elapsed_s=time.monotonic()-started, stdout=stream.getvalue()[-16000:])
    except (Exception, SkillDeadlineExceeded) as error:
        return dict(ok=False, error={"type": "TimeoutError" if isinstance(error, SkillDeadlineExceeded) else type(error).__name__, "message": str(error),
                                    "traceback": traceback.format_exc(limit=8)},
                    sha256=sha256(code), elapsed_s=time.monotonic()-started,
                    stdout=stream.getvalue()[-16000:])

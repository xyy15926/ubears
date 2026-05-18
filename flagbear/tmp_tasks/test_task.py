#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_task.py
#   Author: xyy15926
#   Created: 2026-04-27 18:28:18
#   Updated: 2026-04-28 14:34:32
#   Description:
# ---------------------------------------------------------

# %%
from typing import List, Dict, Any
import pytest
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, force=True)
    from importlib import reload
    from flagbear.tree import dag
    from flagbear.slp import storage, checkpoint, task 
    reload(dag)
    reload(storage)
    reload(checkpoint)
    reload(task)

import time
import shutil

from flagbear.slp.finer import get_tmp_path
from flagbear.slp.storage import LocalFileStorage
from flagbear.slp.checkpoint import CheckpointManager
from flagbear.slp.task import(
    Task,
    TaskFlow,
)

PYTEST_DIR = "tmp/pytest_tmpdir"
TMP_DIR = get_tmp_path(PYTEST_DIR)


# %%
@pytest.fixture(scope="module", autouse=False)
def tmpfile_fixture(request):
    yield

    # Remove the tmp file during the pytest.
    # Clear only once with `scope=module`.
    pytest_tmp = TMP_DIR
    shutil.rmtree(pytest_tmp, ignore_errors=True)
    if not any(get_tmp_path().iterdir()):
        get_tmp_path().rmdir()


# %%
def test_Task():
    def add1(a: int, b: int = 0, c: int = 0):
        return a + b + c + 1

    # `Task.__call__`
    task_add1 = Task(add1)
    assert task_add1(3) == 4

    # `Task.execute`
    context = {"a": 1, "b": 2}
    upstream_results = {"a": 2, "c": 3}
    assert task_add1.execute(context, upstream_results) == 8

    # `Task.name == Task.id_`
    assert task_add1.name == task_add1.id_ == add1.__qualname__
    task_add1.name = "name"
    assert task_add1.name == task_add1.id_ == "name"


# %%
def test_Task_retry():
    run_times = 2
    def add1(a: int, b: int = 0, c: int = 0):
        nonlocal run_times
        if run_times > 0:
            run_times -= 1
            raise RuntimeError(f"run_times {run_times} > 0.")
        return a + b + c + 1

    # Retry 3 times at most.
    task_add1 = Task(add1, retries = 3, retry_delay = 1.0)
    with pytest.raises(RuntimeError):
        assert task_add1(2, 2, 3) == 8
    context = {"a": 1, "b": 2}
    upstream_results = {"a": 2, "c": 3}
    # Retry.
    assert task_add1.execute(context, upstream_results) == 8


# %%
def test_Flow_parallel_and_sequential():
    flow = TaskFlow()

    @flow.task(name = "a")
    def a():
        time.sleep(.1)
        return 2

    @flow.task(name = "b")
    def b():
        time.sleep(.1)
        return 3

    @flow.task(name = "c")
    def c():
        time.sleep(.1)
        return 3

    @flow.task(name = "sum3")
    def sum3(a: int, b: int, c: int = 9):
        return a + b + c

    @flow.task(name = "square")
    def square(sum3: int):
        return sum3 ** 2

    @flow.task(name = "output")
    def output(square: float, context: Dict[str, Any]):
        return f"{context['prefix']}: {square}"

    [a, b, c] >> sum3 >> square >> output

    # Parallel execution.
    ret = flow.process({"prefix": "flow_prefix"}, parallel = True)
    assert ret["output"].value == "flow_prefix: 64"
    assert (max(ret["a"].end_time, ret["b"].end_time, ret["c"].end_time)
            - min(ret["a"].start_time, ret["b"].start_time, ret["c"].start_time)) < .2

    # Sequential execution.
    ret = flow.process({"prefix": "flow_prefix"}, parallel = False)
    assert ret["output"].value == "flow_prefix: 64"
    assert (max(ret["a"].end_time, ret["b"].end_time, ret["c"].end_time)
            - min(ret["a"].start_time, ret["b"].start_time, ret["c"].start_time)) > .3


# %%
def test_Flow_fail():
    flow = TaskFlow()

    @flow.task(name = "a")
    def a():
        return 2

    @flow.task(name = "b")
    def b():
        return 3

    @flow.task(name = "c")
    def c():
        return 3

    @flow.task(name = "sum3")
    def sum3(a: int, b: int, c: int = 9):
        return a + b + c

    @flow.task(name = "square")
    def square(sum3: int):
        return sum3 ** 2

    @flow.task(name = "output")
    def output(square: float, context: Dict[str, Any]):
        return f"{context['prefix']}: {square}"

    [a, b, c] >> sum3 >> square >> output

    # Task fails.
    ret = flow.process({"no_prefix": "flow_prefix"}, fail_fast = False)
    assert isinstance(ret["output"].error, KeyError)

    # Fail and raise error immediately.
    with pytest.raises(RuntimeError):
        ret = flow.process(
            {"no_prefix": "flow_prefix"},
            fail_fast = True,
            parallel = True,
        )
    with pytest.raises(RuntimeError):
        ret = flow.process(
            {"no_prefix": "flow_prefix"},
            fail_fast = True,
            parallel = False,
        )


# %%
def test_Flow_with_cache(tmpfile_fixture):
    lstorage = LocalFileStorage(TMP_DIR, 1024)
    chkp = CheckpointManager(lstorage)
    flow = TaskFlow()

    @flow.task(name = "a")
    @chkp.checkpoint
    def a():
        time.sleep(.1)
        return 2

    @flow.task(name = "b")
    @chkp.checkpoint
    def b():
        time.sleep(.1)
        return 3

    @flow.task(name = "c")
    @chkp.checkpoint
    def c():
        time.sleep(.1)
        return 3

    @flow.task(name = "sum3")
    def sum3(a: int, b: int, c: int = 9):
        return a + b + c

    @flow.task(name = "square")
    def square(sum3: int):
        return sum3 ** 2

    @flow.task(name = "output")
    def output(square: float, context: Dict[str, Any]):
        return f"{context['prefix']}: {square}"

    [a, b, c] >> sum3 >> square >> output

    # Parallel execution first time without cache.
    ret = flow.process({"prefix": "flow_prefix"}, parallel = True)
    assert ret["output"].value == "flow_prefix: 64"
    assert (max(ret["a"].end_time, ret["b"].end_time, ret["c"].end_time)
            - min(ret["a"].start_time, ret["b"].start_time, ret["c"].start_time)) > .1

    # Parallel execution with cache.
    ret = flow.process({"prefix": "flow_prefix"}, parallel = True)
    assert ret["output"].value == "flow_prefix: 64"
    assert (max(ret["a"].end_time, ret["b"].end_time, ret["c"].end_time)
            - min(ret["a"].start_time, ret["b"].start_time, ret["c"].start_time)) < .1

    # Sequential execution with cache.
    ret = flow.process({"prefix": "flow_prefix"}, parallel = False)
    assert ret["output"].value == "flow_prefix: 64"
    assert (max(ret["a"].end_time, ret["b"].end_time, ret["c"].end_time)
            - min(ret["a"].start_time, ret["b"].start_time, ret["c"].start_time)) < .1

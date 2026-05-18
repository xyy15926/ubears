#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_task.py
#   Author: xyy15926
#   Created: 2026-05-06 22:33:12
#   Updated: 2026-05-17 21:19:38
#   Description:
# ---------------------------------------------------------

# %%
import pytest

if __name__ == "__main__":
    from importlib import reload
    from flagbear.slp import cache
    reload(cache)
    from flagbear.sched import task
    reload(task)

from flagbear.slp.cache import MemoryCache
from flagbear.sched.task import(
    TaskProxy,
    task,
    TaskOnce,
)


# %%
def test_Task():
    def add(a, b, c):
        return a + b + c

    add_task = TaskProxy(add, "ka")
    assert add_task(1, 2, 3) == add(1, 2, 3) == 6
    assert isinstance(add_task, TaskProxy)
    with pytest.raises(RuntimeError):
        add_task.submit(1, 2, 3)

    @task
    def add2(a, b, c):
        return a + b + c

    assert add2(1, 2, 3) == add(1, 2, 3) == 6
    assert isinstance(add2, TaskProxy)


# %%
def test_TaskOnce():
    @task
    def add(a, b, c, d):
        return a + b + c + d

    @task
    def mul(a, b, c, d):
        return a * b * c * d

    with pytest.raises(RuntimeError):
        _sum = add.submit(1, 2, 3, 4)

    add_fut = TaskOnce(add, (1, 2, 3, 4), {})
    mul_fut = TaskOnce(mul, (1, add_fut), {"c": add_fut, "d": 4})

    cache = MemoryCache()
    pready, kready, unready, failed = add_fut.resolve_args(cache)
    assert pready == [1, 2, 3, 4]
    assert len(kready) == 0
    assert len(unready) == 0
    assert len(failed) == 0

    pready, kready, unready, failed = mul_fut.resolve_args(cache)
    assert pready == [1, ]
    assert kready == {"d": 4}
    assert len(unready) == 1
    assert unready[0] == add_fut
    assert len(failed) == 0

    deps = mul_fut.resolve_dependencies(cache)
    assert len(deps) == 1
    assert deps[0] == add_fut

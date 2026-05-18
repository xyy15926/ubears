#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_scheduler.py
#   Author: xyy15926
#   Created: 2026-05-16 20:34:00
#   Updated: 2026-05-18 13:46:47
#   Description:
# ---------------------------------------------------------

# %%
import pytest
import time
import asyncio
from datetime import timedelta
import threading
import os

if __name__ == "__main__":
    from importlib import reload
    from flagbear.tree import dag
    reload(dag)
    from flagbear.slp import cache
    reload(cache)
    from flagbear.sched import protocols, task, executor, scheduler
    reload(protocols)
    reload(task)
    reload(executor)
    reload(scheduler)

from flagbear.slp.cache import MemoryCache
from flagbear.sched.protocols import(
    RetryPolicy,
    ExecutionPolicy,
)
from flagbear.sched.task import(
    task,
    TaskOnce,
)
from flagbear.sched.executor import LocalExecutor
from flagbear.sched.scheduler import DAGScheduler


# %%
def test_DAGScheduler_single_task():
    local_exec = LocalExecutor()
    mem_cache = MemoryCache()
    dag_sched = DAGScheduler(mem_cache, local_exec)

    @task
    def add(a, b, c, d):
        time.sleep(0.1)
        return a + b + c + d

    @task
    async def async_add(a, b, c, d):
        await asyncio.sleep(0.1)
        return a + b + c + d

    # Sync task.
    add_once = TaskOnce(add, (1, 2, 3, 4), {})
    dag_sched.add(add_once)
    dag_sched.wait(add_once)
    add_ret = mem_cache.get(add_once.id_)
    assert add_ret.value == 10

    # Async task.
    async_add_once = TaskOnce(async_add, (1, 2, 3, 4), {})
    dag_sched.add(async_add_once)
    dag_sched.wait(async_add_once)
    async_add_ret = mem_cache.get(async_add_once.id_)
    assert async_add_ret.value == 10

    # TaskOnce can't be submited again.
    with pytest.raises(ValueError):
        dag_sched.add(add_once)

    dag_sched.shutdown()


# %%
def test_DAGScheduler_add_unordered():
    local_exec = LocalExecutor()
    mem_cache = MemoryCache()
    dag_sched = DAGScheduler(mem_cache, local_exec)

    @task
    def add(a, b, c, d):
        time.sleep(0.1)
        return a + b + c + d

    @task
    async def async_add(a, b, c, d):
        await asyncio.sleep(0.1)
        return a + b + c + d

    add_once = TaskOnce(add, (1, 2, 3, 4), {})
    add_once2 = TaskOnce(add, (1, 2, 3, 4), {})
    add_once3 = TaskOnce(add, (1, 2, 3, 4), {})
    async_add_once = TaskOnce(
        async_add,
        (1, add_once, add_once2, add_once3),
        {},
    )

    start = time.time()
    # Though `add_once2 >> async_add_once`, `.add(async_add_once, add_once)`
    # and `.wait(async_add_once, add_once)` is fine.
    dag_sched.add(add_once, async_add_once, add_once2, add_once3)
    dag_sched.wait(async_add_once, add_once)
    end = time.time()

    add_ret = mem_cache.get(add_once.id_)
    async_add_ret = mem_cache.get(async_add_once.id_)
    assert add_ret.value == 10
    assert async_add_ret.value == 31
    assert end - start < 0.3
    assert len(mem_cache.list_keys()) == 4

    dag_sched.shutdown()

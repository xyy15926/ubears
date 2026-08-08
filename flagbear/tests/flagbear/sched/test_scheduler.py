#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_scheduler.py
#   Author: xyy15926
#   Created: 2026-05-16 20:34:00
#   Updated: 2026-05-20 15:23:38
#   Description:
# ---------------------------------------------------------

# %%
import asyncio
import shutil
import time

import pytest

if __name__ == "__main__":
    from importlib import reload

    from flagbear.tree import dag

    reload(dag)
    from flagbear.slp import cache, finer

    reload(finer)
    reload(cache)
    from flagbear.sched import executor, protocols, scheduler, task

    reload(protocols)
    reload(task)
    reload(executor)
    reload(scheduler)

from flagbear.sched.executor import LocalExecutor
from flagbear.sched.scheduler import DAGScheduler
from flagbear.sched.task import (
    TaskOnce,
    task,
)
from flagbear.slp.cache import MemoryCache, PersistentCache
from flagbear.slp.finer import get_tmp_path
from flagbear.slp.storage import LocalFileStorage

PYTEST_DIR = "tmp/pytest_tmpdir"
TMP_DIR = get_tmp_path(PYTEST_DIR)


# %%
@pytest.fixture(scope="function", autouse=False)
def tmpfile_fixture(request):
    yield

    # Remove the tmp file during the pytest.
    # Clear only once with `scope=module`.
    pytest_tmp = TMP_DIR
    shutil.rmtree(pytest_tmp, ignore_errors=True)
    if not any(get_tmp_path().iterdir()):
        get_tmp_path().rmdir()


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
def test_DAGScheduler_add_unordered_and_reuse_cache():
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
    assert len(mem_cache.list_keys()) == 4 + 2

    dag_sched.shutdown()

    # Reuse the cache.
    local_exec = LocalExecutor()
    dag_sched = DAGScheduler(mem_cache, local_exec)

    add_once = TaskOnce(add, (1, 2, 3, 4), {})
    add_once2 = TaskOnce(add, (1, 2, 3, 4), {})
    add_once3 = TaskOnce(add, (1, 2, 3, 4), {})
    async_add_once = TaskOnce(
        async_add,
        (1, add_once, add_once2, add_once3),
        {},
    )

    start = time.time()
    dag_sched.add(add_once, async_add_once, add_once2, add_once3)
    dag_sched.wait(async_add_once, add_once)
    end = time.time()

    # Task has been re-executed, namely the `TaskResult`s in cache has been
    # restored, so `add_ret.value` is not None.
    add_ret = mem_cache.get(add_once.id_)
    assert add_ret.value == 10
    add_ret_value = mem_cache.get(add_ret.cache_key)
    assert add_ret_value == 10

    async_add_ret = mem_cache.get(async_add_once.id_)
    assert async_add_ret.value == 31
    async_add_ret_value = mem_cache.get(async_add_ret.cache_key)
    assert async_add_ret_value == 31
    assert end - start < 0.1

    dag_sched.shutdown()


# %%
def test_DAGScheduler_with_persistent_cache(tmpfile_fixture):
    lstorage = LocalFileStorage(TMP_DIR)
    pcache = PersistentCache(lstorage, max_mem_size=1024)
    local_exec = LocalExecutor()
    dag_sched = DAGScheduler(pcache, local_exec)

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
    dag_sched.add(add_once, async_add_once, add_once2, add_once3)
    dag_sched.wait(async_add_once, add_once)
    end = time.time()

    add_ret = pcache.get(add_once.id_)
    async_add_ret = pcache.get(async_add_once.id_)
    assert add_ret.value == 10
    assert async_add_ret.value == 31
    assert end - start < 0.3

    dag_sched.shutdown()

    # Reinit cache, executor and scheduler.
    lstorage = LocalFileStorage(TMP_DIR)
    pcache = PersistentCache(lstorage, max_mem_size=1024)
    local_exec = LocalExecutor()
    dag_sched = DAGScheduler(pcache, local_exec)

    add_once = TaskOnce(add, (1, 2, 3, 4), {})
    add_once2 = TaskOnce(add, (1, 2, 3, 4), {})
    add_once3 = TaskOnce(add, (1, 2, 3, 4), {})
    async_add_once = TaskOnce(
        async_add,
        (1, add_once, add_once2, add_once3),
        {},
    )

    start = time.time()
    dag_sched.add(add_once, async_add_once, add_once2, add_once3)
    dag_sched.wait(async_add_once, add_once)
    end = time.time()

    # Task has been re-executed, namely the `TaskResult`s in cache has been
    # restored, so `add_ret.value` is not None.
    add_ret = pcache.get(add_once.id_)
    assert add_ret.value == 10
    add_ret_value = pcache.get(add_ret.cache_key)
    assert add_ret_value == 10

    async_add_ret = pcache.get(async_add_once.id_)
    assert async_add_ret.value == 31
    async_add_ret_value = pcache.get(async_add_ret.cache_key)
    assert async_add_ret_value == 31
    assert end - start < 0.1

    dag_sched.shutdown()

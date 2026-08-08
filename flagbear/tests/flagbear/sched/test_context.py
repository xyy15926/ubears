#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_flow.py
#   Author: xyy15926
#   Created: 2026-05-07 10:03:37
#   Updated: 2026-05-20 18:29:28
#   Description:
# ---------------------------------------------------------

# %%
import asyncio
import shutil
import time

import pytest

# from IPython.core.debugger import set_trace

if __name__ == "__main__":
    from importlib import reload

    from flagbear.slp import cache, finer, storage

    reload(finer)
    reload(storage)
    reload(cache)
    from flagbear.tree import dag

    reload(dag)
    from flagbear.sched import context, executor, protocols, scheduler, task

    reload(protocols)
    reload(task)
    reload(executor)
    reload(scheduler)
    reload(context)

from flagbear.sched.context import SimpleContext
from flagbear.sched.protocols import _current_context
from flagbear.sched.task import TaskOnce, task
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
def test_SimpleContext_submit_run():
    @task
    def add(a, b, c, d):
        time.sleep(0.1)
        return a + b + c + d

    @task
    async def async_add(a, b, c, d):
        await asyncio.sleep(0.1)
        return a + b + c + d

    with SimpleContext() as ctx:
        assert isinstance(ctx, SimpleContext)

        # Submit and gather.
        add_once = TaskOnce(add, (1, 2, 3, 4), {})
        async_add_once = TaskOnce(async_add, (1, 2, 3, 4), {})
        ctx.submit(add_once, async_add_once)
        start = time.time()
        add_ret = ctx.get_result(add_once)
        async_add_ret = ctx.get_result(async_add_once)
        end = time.time()
        assert add_ret.value == 10
        assert async_add_ret.value == 10
        assert end - start < 0.2

        # Run.
        add_once = TaskOnce(add, (1, 2, 3, 4), {})
        async_add_once = TaskOnce(async_add, (1, 2, 3, 4), {})
        start = time.time()
        add_ret, async_add_ret = ctx.run(add_once, async_add_once)
        end = time.time()
        assert add_ret.value == 10
        assert async_add_ret.value == 10
        # Cache will be used.
        assert end - start < 0.1


# %%
def test_SimpleContext_submit_unordered_twice():
    @task
    def add(a, b, c, d):
        time.sleep(0.1)
        return a + b + c + d

    @task
    async def async_add(a, b, c, d):
        await asyncio.sleep(0.1)
        return a + b + c + d

    with SimpleContext() as ctx:
        add_once = TaskOnce(add, (1, 2, 3, 4), {})
        add_once2 = TaskOnce(add, (1, 2, 3, 4), {})
        add_once3 = TaskOnce(add, (1, 2, 3, 4), {})
        async_add_once = TaskOnce(
            async_add,
            (1, add_once, add_once2, add_once3),
            {},
        )

        # Though `add_once2 >> async_add_once`
        # But, `ctx.submit(async_add_once, add_once)` is fine.
        start = time.time()
        ctx.submit(add_once, async_add_once, add_once3, add_once2)
        add_ret = ctx.get_result(add_once)
        async_add_ret = ctx.get_result(async_add_once)
        end = time.time()
        assert add_ret.value == 10
        assert async_add_ret.value == 31
        assert end - start < 0.3
        assert len(ctx.task_results.list_keys()) == 6

        # TaskOnce can't be submited again.
        with pytest.raises(ValueError):
            ctx.submit(add_once)


# %%
def test_SimpleContext_context_and_TaskProxy_submit():
    @task
    def add(a, b, c, d):
        time.sleep(0.1)
        return a + b + c + d, _current_context.get()

    @task
    async def async_add(a, b, c, d):
        await asyncio.sleep(0.1)
        return a + b + c + d, _current_context.get()

    with SimpleContext() as ctx:
        # Simple task.
        start = time.time()
        add_task = add.submit(1, 2, 3, 4)
        async_add_task = async_add.submit(1, 2, 3, 4)
        assert isinstance(add_task, TaskOnce)
        assert isinstance(async_add_task, TaskOnce)
        add_ret, sync_ctx = add_task.result()
        async_add_ret, async_ctx = async_add_task.result()
        end = time.time()
        assert add_ret == 10
        assert async_add_ret == 10
        assert end - start < 0.2
        assert sync_ctx is None
        assert async_ctx is None

        # TaskOnce can resulted twice but can't be gathere twice.
        add_ret, sync_ctx = add_task.result()
        with pytest.raises(ValueError):
            add_ret, sync_ctx = ctx.run(add_task)


# %%
def test_SimpleContext_TaskOnce_call_and_submit_then_result():
    @task
    def add(a, b, c, d):
        time.sleep(0.1)
        return a + b + c + d, _current_context.get()

    @task
    async def async_add(a, b, c, d):
        await asyncio.sleep(0.1)
        return a + b + c + d, _current_context.get()

    with SimpleContext():
        # `ctx.gather` to execute task simutaneously.
        start = time.time()
        add_task = add.submit(1, 2, 3, 4)
        async_add_task = async_add.submit(1, 2, 3, 4)
        add_ret, sync_ctx = add_task.result()
        async_add_ret, async_ctx = async_add_task.result()
        end = time.time()
        assert add_ret == 10
        assert async_add_ret == 10
        assert end - start < 0.2
        assert sync_ctx is None
        assert async_ctx is None

    with SimpleContext():
        # Simple task but do calculation immediately.
        start = time.time()
        add_ret, sync_ctx = add(1, 2, 3, 4)
        async_add_ret, async_ctx = async_add(1, 2, 3, 4)
        end = time.time()
        assert add_ret == 10
        assert async_add_ret == 10
        assert end - start > 0.2
        assert sync_ctx is None
        assert async_ctx is None


# %%
def test_SimpleContext_TaskOnce_with_dag_topo():
    @task
    def add(a, b, c, d):
        time.sleep(0.1)
        return a + b + c + d

    @task
    async def async_add(a, b, c, d):
        await asyncio.sleep(0.1)
        return a + b + c + d

    with SimpleContext():
        start = time.time()
        add_task1 = add.submit(1, 2, 3, 4)
        add_task2 = add.submit(1, 2, 3, 4)
        async_add_task1 = async_add.submit(add_task1, 2, 3, 4)
        async_add_task2 = async_add.submit(add_task1, 2, 3, 4)
        total = add(add_task1, add_task2, async_add_task1, async_add_task2)
        end = time.time()
        assert total == 58
        assert end - start < 0.4


# %%
def test_SimpleContext_TaskOnce_error():
    @task
    def add(a, b, c, d):
        raise RuntimeError("RuntimeError")
        return a + b + c + d

    @task
    async def async_add(a, b, c, d):
        return a + b + c + d

    mem_cache = MemoryCache()
    with SimpleContext(mem_cache):
        add_task1 = add.submit(1, 2, 3, 4)
        add_task2 = add.submit(1, 2, 3, 4)
        async_add_task1 = async_add.submit(add_task1, 2, 3, 4)
        async_add_task2 = async_add.submit(add_task1, 2, 3, 4)
        with pytest.raises(RuntimeError):
            _total = add(
                add_task1, add_task2, async_add_task1, async_add_task2
            )

    for key in mem_cache.list_keys():
        add_ret = mem_cache.get(key)
        # As no Task is successful, no cache of function result is set.
        assert add_ret.is_failed()
        assert add_ret.value is None

    # Reuse the cache.
    with SimpleContext(mem_cache):
        add_task1 = add.submit(1, 2, 3, 4)
        add_task2 = add.submit(1, 2, 3, 4)
        async_add_task1 = async_add.submit(add_task1, 2, 3, 4)
        async_add_task2 = async_add.submit(add_task1, 2, 3, 4)
        with pytest.raises(RuntimeError):
            _total = add(
                add_task1, add_task2, async_add_task1, async_add_task2
            )

    for key in mem_cache.list_keys():
        add_ret = mem_cache.get(key)
        # As no Task is successful, no cache of function result is set.
        assert add_ret.is_failed()
        assert add_ret.value is None


# %%
def test_SimpleContext_TaskOnce_error_with_persistent_cache(tmpfile_fixture):
    @task
    def add(a, b, c, d):
        raise RuntimeError("RuntimeError")
        return a + b + c + d

    @task
    async def async_add(a, b, c, d):
        return a + b + c + d

    lstorage = LocalFileStorage(TMP_DIR)
    pcache = PersistentCache(lstorage, max_mem_size=1024)
    with SimpleContext(pcache):
        add_task1 = add.submit(1, 2, 3, 4)
        add_task2 = add.submit(1, 2, 3, 4)
        async_add_task1 = async_add.submit(add_task1, 2, 3, 4)
        async_add_task2 = async_add.submit(add_task1, 2, 3, 4)
        with pytest.raises(RuntimeError):
            _total = add(
                add_task1, add_task2, async_add_task1, async_add_task2
            )

    for key in pcache.list_keys():
        add_ret = pcache.get(key)
        # As no Task is successful, no cache of function result is set.
        assert add_ret.is_failed()
        assert add_ret.value is None

    # Reuse the cache.
    lstorage = LocalFileStorage(TMP_DIR)
    pcache = PersistentCache(lstorage, max_mem_size=1024)
    with SimpleContext(pcache):
        add_task1 = add.submit(1, 2, 3, 4)
        add_task2 = add.submit(1, 2, 3, 4)
        async_add_task1 = async_add.submit(add_task1, 2, 3, 4)
        async_add_task2 = async_add.submit(add_task1, 2, 3, 4)
        with pytest.raises(RuntimeError):
            _total = add(
                add_task1, add_task2, async_add_task1, async_add_task2
            )

    for key in pcache.list_keys():
        add_ret = pcache.get(key)
        # As no Task is successful, no cache of function result is set.
        assert add_ret.is_failed()
        assert add_ret.value is None
        assert isinstance(add_ret.error, Exception)


# %%
def test_SimpleContext_nested_sync_task():
    @task
    def add(a, b, c, d):
        time.sleep(0.1)
        return a + b + c + d

    @task
    def mul(a, b, c, d):
        time.sleep(0.1)
        return a * b * c * d

    @task
    def mix_error(a, b):
        add_task = add.submit(a, b, 3, 4)
        mul_task = mul.submit(1, add_task, c=3, d=add_task)
        add_ret = add_task.result()
        mul_ret = mul_task.result()
        return add_ret + mul_ret

    @task
    def mix(a, b):
        add_ret = add(a, b, 3, 4)
        mul_ret = mul(1, add_ret, c=3, d=add_ret)
        return add_ret + mul_ret

    with SimpleContext() as ctx:
        # Task can't be submited as a task in another task by default.
        with pytest.raises(RuntimeError):
            mix_error(1, 2)

        # Nested task.
        start = time.time()
        mix_task = mix.submit(1, 2)
        mix_ret = mix_task.result()
        end = time.time()
        assert mix_ret == 310
        assert end - start < 0.3

    # The task nested in another task will be treated as a normal function.
    # So, only `mix_error` and `mix` will be set in `ctx.task_results`.
    assert len(ctx.task_results.list_keys()) == 3


# %%
def test_SimpleContext_nested_async_func_and_coro():
    async def async_add(a, b, c, d):
        await asyncio.sleep(0.1)
        return a + b + c + d

    @task
    async def async_mul(a, b, c, d):
        await asyncio.sleep(0.1)
        return a * b * c * d

    @task
    async def mix_loop(a, b):
        add_coro = async_add(a, b, 3, 4)
        mul_coro = async_mul(a, b, 3, 4)
        add_ret = await add_coro
        mul_ret = await mul_coro
        return add_ret + mul_ret

    @task
    async def mix_gather(a, b):
        add_coro = async_add(a, b, 3, 4)
        mul_coro = asyncio.create_task(async_mul(a, b, 3, 4))
        add_ret, mul_ret = await asyncio.gather(add_coro, mul_coro)
        return add_ret + mul_ret

    with SimpleContext() as ctx:
        start = time.time()
        mix_ret = mix_loop(1, 2)
        end = time.time()
        assert mix_ret == 34
        assert end - start > 0.2

        start = time.time()
        mix_ret = mix_gather(1, 2)
        end = time.time()
        assert mix_ret == 34
        assert end - start < 0.2

    # Nested task will be treated as normal function or async function.
    # So, only `mix_loop` and `mix_gather` will be set in `ctx.task_results`.
    assert len(ctx.task_results.list_keys()) == 4


# %%
def test_SimpleContext_nested_context_result_immediately():
    @task
    def add(a, b, c, d):
        time.sleep(0.1)
        return a + b + c + d, _current_context.get()

    @task
    async def async_add(a, b, c, d):
        await asyncio.sleep(0.1)
        return a + b + c + d, _current_context.get()

    with SimpleContext() as outer_ctx:
        add_ret, sync_ctx = add(1, 2, 3, 4)
        async_add_ret, async_ctx = async_add(1, 2, 3, 4)
        assert add_ret == 10
        assert sync_ctx is None
        assert async_add_ret == 10
        assert async_ctx is None
        with SimpleContext(parent_context=outer_ctx) as inner_ctx:
            add_ret, sync_ctx = add(1, 2, 3, 4)
            async_add_ret, async_ctx = async_add(1, 2, 3, 4)
            assert add_ret == 10
            assert sync_ctx is None
            assert async_add_ret == 10
            assert async_ctx is None
            assert inner_ctx.scheduler is outer_ctx.scheduler
            assert inner_ctx._scheduler is None
            assert inner_ctx.task_results is outer_ctx.task_results
            assert inner_ctx._task_results is None

    assert len(outer_ctx.task_results.list_keys()) == 6


# %%
def test_SimpleContext_nested_context_submit_and_waiting_shutdown():
    @task
    def add(a, b, c, d):
        time.sleep(0.1)
        return a + b + c + d, _current_context.get()

    @task
    async def async_add(a, b, c, d):
        await asyncio.sleep(0.1)
        return a + b + c + d, _current_context.get()

    with SimpleContext() as outer_ctx:
        add_task = add.submit(1, 2, 3, 4)
        _async_task = async_add.submit(1, 2, 3, 4)
        with SimpleContext(parent_context=outer_ctx) as inner_ctx:
            add_task = add.submit(1, 2, 3, 4)
            _async_task = async_add.submit(1, 2, 3, 4)
            assert inner_ctx.scheduler is outer_ctx.scheduler
            assert inner_ctx._scheduler is None
            assert inner_ctx.task_results is outer_ctx.task_results
            assert inner_ctx._task_results is None

    task_results = outer_ctx.task_results
    add_ret = task_results.get(add_task.id_)
    assert add_ret.is_successful()
    assert add_ret.value == (10, None)


# %%
def test_SimpleContext_nested_context_submit_and_force_shutdown():
    @task
    def add(a, b, c, d):
        time.sleep(0.1)
        return a + b + c + d, _current_context.get()

    @task
    async def async_add(a, b, c, d):
        await asyncio.sleep(0.1)
        return a + b + c + d, _current_context.get()

    with SimpleContext() as outer_ctx:
        _add_task = add.submit(1, 2, 3, 4)
        _async_task = async_add.submit(1, 2, 3, 4)
        with SimpleContext(parent_context=outer_ctx) as inner_ctx:
            _add_task = add.submit(1, 2, 3, 4)
            _async_task = async_add.submit(1, 2, 3, 4)
            assert inner_ctx.scheduler is outer_ctx.scheduler
            assert inner_ctx._scheduler is None
            assert inner_ctx.task_results is outer_ctx.task_results
            assert inner_ctx._task_results is None
        # Shutdown the executor immediately without waiting for tasks.
        outer_ctx.shutdown(True)

    task_results = outer_ctx.task_results
    for key in task_results.list_keys():
        add_ret = task_results.get(key)
        assert add_ret.is_failed()
        assert add_ret.value is None
    # TODO

#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_executor.py
#   Author: xyy15926
#   Created: 2026-05-06 23:13:48
#   Updated: 2026-05-20 14:25:57
#   Description:
# ---------------------------------------------------------

# %%
import asyncio
import concurrent
import os
import shutil
import threading
import time
from datetime import timedelta

import pytest

if __name__ == "__main__":
    from importlib import reload

    from flagbear.slp import cache

    reload(cache)
    from flagbear.sched import executor, protocols, task

    reload(protocols)
    reload(task)
    reload(executor)

from flagbear.sched.executor import LocalExecutor
from flagbear.sched.protocols import (
    ExecutionPolicy,
    RetryPolicy,
    TaskState,
)
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
def test_LocalExecutor_run_coro_shutdown_execute_func_once():
    mpid = os.getpid()
    mtid = threading.get_ident()
    engine = LocalExecutor()
    assert engine._event_loop is None
    assert engine._thread_pool is None
    assert engine._process_pool is None

    # Async function.
    async def add(a, b, c, d):
        return a + b + c + d, threading.get_ident(), os.getpid()

    coro = engine.execute_func_once(add, [1, 2, 3, 4], {})
    result, loop_tid, loop_pid = engine.run_coro(coro).result()
    assert result == 10
    assert loop_tid != mtid
    assert loop_pid == mpid
    assert engine._event_loop is not None
    assert engine._thread_pool is None
    assert engine._process_pool is None

    # Sync function.
    def add(a, b, c, d):
        return a + b + c + d, threading.get_ident(), os.getpid()

    coro = engine.execute_func_once(add, [1, 2, 3, 4], {})
    result, tp_tid, tp_pid = engine.run_coro(coro).result()
    assert result == 10
    assert tp_tid != mtid
    assert tp_tid != loop_tid
    assert tp_pid == mpid
    assert engine._event_loop is not None
    assert engine._thread_pool is not None
    assert engine._process_pool is None

    # Execution policy with timeout.
    execution_policy = ExecutionPolicy(0.1, False)

    async def add(a, b, c, d):
        await asyncio.sleep(0.2)
        return a + b + c + d

    coro = engine.execute_func_once(add, [1, 2, 3, 4], {}, execution_policy)
    with pytest.raises(TimeoutError):
        result = engine.run_coro(coro).result()

    def add(a, b, c, d):
        time.sleep(0.2)
        return a + b + c + d

    coro = engine.execute_func_once(add, [1, 2, 3, 4], {}, execution_policy)
    with pytest.raises(TimeoutError):
        result = engine.run_coro(coro).result()

    # Shutdown.
    engine.shutdown()
    assert not engine._event_loop.is_running()
    assert engine._thread_pool is not None
    assert engine._process_pool is None


# %%
def test_LocalExecutor_shutdown():
    engine = LocalExecutor()

    def add(a, b, c, d):
        time.sleep(0.1)
        return a + b + c + d

    coro = engine.execute_func_once(add, [1, 2, 3, 4], {})
    future = engine.run_coro(coro)
    engine.shutdown()

    # `LocalExecutor.shutdown()` will cancel the `asyncio.Task`s and raises
    # `asyncio.CancelledError` and then causes
    # `concurrent.futures.CancelledError`
    with pytest.raises(concurrent.futures.CancelledError):
        _ret = future.result()
    with pytest.raises(concurrent.futures.CancelledError):
        _ret = future.exception()


# %%
def _main_add(a, b, c, d):
    return a + b + c + d, threading.get_ident(), os.getpid()


def test_LocalExecutor_run_process():
    mpid = os.getpid()
    mtid = threading.get_ident()
    engine = LocalExecutor(4, 2)
    assert engine._event_loop is None
    assert engine._thread_pool is None
    assert engine._process_pool is None

    # Execution policy with process used.
    execution_policy = ExecutionPolicy(None, True)

    # `_main_add` should not be nested in other function to be
    # executed in sub-process.
    coro = engine.execute_func_once(
        _main_add, [1, 2, 3, 4], {}, execution_policy
    )
    result, pp_tid, pp_pid = engine.run_coro(coro).result()
    assert result == 10
    assert pp_tid != mtid
    assert pp_pid != mpid
    assert engine._event_loop is not None
    assert engine._thread_pool is None
    assert engine._process_pool is not None

    engine.shutdown()
    assert not engine._event_loop.is_running()
    assert engine._thread_pool is None
    assert engine._process_pool is not None


# %%
def test_LocalExecutor_execute_resolved():
    engine = LocalExecutor()

    # Succeed normally.
    @task
    def error_add(a, b, c, d):
        time.sleep(0.3)
        return a + b + c + d

    error_add_fut = TaskOnce(error_add, (1, 2, 3, 4), {})
    coro = engine.execute_resolved(error_add_fut, [1, 2, 3, 4], {})
    result = engine.run_coro(coro).result()
    assert result.is_successful()
    assert result.attempt == 1
    assert result.error is None

    # Retry and failed.
    execution_policy = ExecutionPolicy(0.1, False)
    retry_policy = RetryPolicy(
        4,
        0.1,
        2,
        60,
        [
            TimeoutError,
        ],
    )

    @task(
        execution_policy=execution_policy,
        retry_policy=retry_policy,
    )
    def error_add(a, b, c, d):
        time.sleep(0.3)
        return a + b + c + d

    error_add_fut = TaskOnce(error_add, (1, 2, 3, 4), {})
    coro = engine.execute_resolved(error_add_fut, [1, 2, 3, 4], {})
    result = engine.run_coro(coro).result()
    assert result.is_failed()
    assert result.attempt == retry_policy.max_retries
    assert isinstance(result.error, TimeoutError)

    # Retry and succeed.
    execution_policy = ExecutionPolicy(0.1, False)
    retry_policy = RetryPolicy(
        4,
        0.1,
        2,
        60,
        [
            TimeoutError,
            RuntimeError,
        ],
    )

    counter_flag = 3

    @task(
        execution_policy=execution_policy,
        retry_policy=retry_policy,
    )
    def error_add(a, b, c, d):
        # global counter_flag
        nonlocal counter_flag
        counter_flag -= 1
        if counter_flag > 0:
            raise RuntimeError

        return a + b + c + d

    error_add_fut = TaskOnce(error_add, (1, 2, 3, 4), {})
    coro = engine.execute_resolved(error_add_fut, [1, 2, 3, 4], {})
    result = engine.run_coro(coro).result()
    assert result.is_successful()
    assert result.attempt < retry_policy.max_retries
    assert isinstance(result.error, RuntimeError)

    engine.shutdown()


# %%
def test_LocalExecutor_excecute_future():
    engine = LocalExecutor()
    cache = MemoryCache()

    # Succeed.
    @task
    def add(a, b, c, d):
        time.sleep(0.1)
        return a + b + c + d

    @task
    def mul(a, b, c, d):
        return a * b * c * d

    add_fut = TaskOnce(add, (1, 2, 3, 4), {})
    mul_fut = TaskOnce(mul, (1, add_fut), {"c": 3, "d": add_fut})

    # Precedent tasks must be executed first.
    mul_coro = engine.execute_with_cache(mul_fut, cache)
    with pytest.raises(RuntimeError):
        mul_result = engine.run_coro(mul_coro).result()

    # Execute `add` first.
    add_coro = engine.execute_with_cache(add_fut, cache)
    add_result = engine.run_coro(add_coro).result()

    assert add_result.is_successful()
    assert add_result.value == 10
    assert add_result.end_time - add_result.start_time > timedelta(seconds=0.1)
    add_result_gotten = cache.get(add_fut.id_)
    assert add_result_gotten is add_result

    # Execute `mul` then.
    mul_fut = TaskOnce(mul, (1, add_fut), {"c": 3, "d": add_fut})
    mul_coro = engine.execute_with_cache(mul_fut, cache)
    mul_result = engine.run_coro(mul_coro).result()
    assert mul_result.is_successful()
    assert mul_result.value == 300
    assert mul_result.end_time - mul_result.start_time < timedelta(seconds=0.1)

    # Skip for precedent error.
    @task
    def error_add(a, b, c, d):
        raise RuntimeError("Test Error")
        return a + b + c + d

    error_add_fut = TaskOnce(error_add, (1, 2, 3, 4), {})
    error_add_coro = engine.execute_with_cache(error_add_fut, cache)
    _error_add_result = engine.run_coro(error_add_coro).result()

    error_mul_fut = TaskOnce(mul, (1, error_add_fut), {"c": 3, "d": add_fut})
    error_mul_coro = engine.execute_with_cache(error_mul_fut, cache)
    error_mul_result = engine.run_coro(error_mul_coro).result()
    assert error_mul_result.is_failed()
    assert error_mul_result.value is None

    engine.shutdown()


# %%
def test_LocalExecutor_excecute_future_with_persistent_cache(tmpfile_fixture):
    lstorage = LocalFileStorage(TMP_DIR)
    pcache = PersistentCache(lstorage, max_mem_size=1024)
    engine = LocalExecutor()

    # Succeed.
    @task
    def add(a, b, c, d):
        time.sleep(0.1)
        return a + b + c + d

    @task
    def mul(a, b, c, d):
        return a * b * c * d

    # Execute `add` first.
    add_fut = TaskOnce(add, (1, 2, 3, 4), {})
    add_coro = engine.execute_with_cache(add_fut, pcache)
    add_result = engine.run_coro(add_coro).result()
    assert add_result.value == 10

    # Execute `mul` then.
    mul_fut = TaskOnce(mul, (1, add_fut), {"c": 3, "d": add_fut})
    mul_coro = engine.execute_with_cache(mul_fut, pcache)
    mul_result = engine.run_coro(mul_coro).result()
    assert mul_result.value == 300

    engine.shutdown()

    # Restart engine and persistent cache.
    lstorage = LocalFileStorage(TMP_DIR)
    pcache = PersistentCache(lstorage, max_mem_size=1024)
    engine = LocalExecutor()

    add_fut = TaskOnce(add, (1, 2, 3, 4), {})
    mul_fut = TaskOnce(mul, (1, add_fut), {"c": 3, "d": add_fut})

    # Precedent tasks must be executed first, though cached.
    mul_coro = engine.execute_with_cache(mul_fut, pcache)
    with pytest.raises(RuntimeError):
        mul_result = engine.run_coro(mul_coro).result()

    add_fut = TaskOnce(add, (1, 2, 3, 4), {})
    add_coro = engine.execute_with_cache(add_fut, pcache)
    start_time = time.time()
    cached_add_result = engine.run_coro(add_coro).result()
    end_time = time.time()
    assert cached_add_result.state == TaskState.CACHED
    assert cached_add_result.value == 10
    # Cache will be used.
    assert end_time - start_time < 0.1

    mul_fut = TaskOnce(mul, (1, add_fut), {"c": 3, "d": add_fut})
    mul_coro = engine.execute_with_cache(mul_fut, pcache)
    cached_mul_result = engine.run_coro(mul_coro).result()
    assert cached_mul_result.state == TaskState.CACHED
    assert cached_mul_result.value == 300


# %%
def test_LocalExecutor_submit():
    engine = LocalExecutor()
    cache = MemoryCache()

    @task
    async def async_add(a, b, c, d):
        await asyncio.sleep(0.1)
        return a + b + c + d

    @task
    def add(a, b, c, d):
        time.sleep(0.1)
        return a + b + c + d

    add_fut = TaskOnce(add, (1, 2, 3, 4), {})
    add_fut2 = TaskOnce(add, (1, 2, 3, 4), {})
    async_add_fut = TaskOnce(async_add, (1, 2, 3, 4), {})

    # Submit tasks.
    futures = engine.submit(cache, add_fut, add_fut2, async_add_fut)
    time.sleep(0.1)

    # Gather results.
    start = time.time()
    result, result2, result_async = [fut.result() for fut in futures]
    end = time.time()
    # Task has been submited and run background.
    assert end - start < 0.1
    assert result.is_successful()
    assert result.value == 10
    assert result2.is_successful()
    assert result2.value == 10
    assert result_async.is_successful()
    assert result_async.value == 10

    # Shut down engine.
    engine.shutdown()

#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_flow.py
#   Author: xyy15926
#   Created: 2026-05-07 10:03:37
#   Updated: 2026-05-12 12:09:37
#   Description:
# ---------------------------------------------------------

# %%
import pytest
import time
import threading
import os
import asyncio
from IPython.core.debugger import set_trace

if __name__ == "__main__":
    from importlib import reload
    from flagbear.tree import dag
    from flagbear.sched import protocols, task, executor, context
    reload(dag)
    reload(protocols)
    reload(task)
    reload(executor)
    reload(context)

from flagbear.sched.protocols import _current_context
from flagbear.sched.task import task, TaskOnce
from flagbear.sched.executor import LocalExecutor 
from flagbear.sched.context import LazyContext


# %%
def test_LazyContext_submit_gather_run():
    @task
    def add(a, b, c, d):
        time.sleep(0.1)
        return a + b + c + d

    @task
    async def async_add(a, b, c, d):
        await asyncio.sleep(0.1)
        return a + b + c + d

    with LazyContext() as ctx:
        assert isinstance(ctx, LazyContext)

        # Submit and gather.
        add_once = TaskOnce(add, (1, 2, 3, 4), {})
        async_add_once = TaskOnce(async_add, (1, 2, 3, 4), {})
        ctx.submit(add_once, async_add_once)
        start = time.time()
        add_ret, async_add_ret = ctx.gather(add_once, async_add_once)
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
        assert end - start < 0.2


# %%
def test_LazyContext_submit_unordered_twice():
    @task
    def add(a, b, c, d):
        time.sleep(0.1)
        return a + b + c + d

    @task
    async def async_add(a, b, c, d):
        await asyncio.sleep(0.1)
        return a + b + c + d

    with LazyContext() as ctx:
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
        ctx.submit(add_once, async_add_once, add_once3, add_once2)
        start = time.time()
        add_ret, async_add_ret = ctx.gather(add_once, async_add_once)
        end = time.time()
        assert add_ret.value == 10
        assert async_add_ret.value == 31
        assert end - start < 0.3
        assert len(ctx.result_cache.list_keys()) == 4

        # TaskOnce can't be submited again.
        with pytest.raises(ValueError):
            ctx.submit(add_once)


# %%
def test_LazyContext_context_and_TaskOnce_submit():
    @task
    def add(a, b, c, d):
        time.sleep(0.1)
        return a + b + c + d, _current_context.get()

    @task
    async def async_add(a, b, c, d):
        await asyncio.sleep(0.1)
        return a + b + c + d, _current_context.get()

    with LazyContext() as ctx:
        # Simple task.
        add_fut = add.submit(1, 2, 3, 4)
        async_add_fut = async_add.submit(1, 2, 3, 4)
        assert isinstance(add_fut, TaskOnce)
        assert isinstance(async_add_fut, TaskOnce)
        start = time.time()
        add_ret, sync_ctx = add_fut.result()
        async_add_ret, async_ctx = async_add_fut.result()
        end = time.time()
        assert add_ret == 10
        assert async_add_ret == 10
        assert end - start > 0.2
        assert sync_ctx is None
        assert async_ctx is None

        # TaskOnce can resulted twice but can't be gathere twice.
        add_ret, sync_ctx = add_fut.result()
        with pytest.raises(ValueError):
            add_ret, sync_ctx = ctx.gather(add_fut)

# %%
def test_LazyContext_TaskOnce_call_and_submit_then_gather():
    @task
    def add(a, b, c, d):
        time.sleep(0.1)
        return a + b + c + d, _current_context.get()

    @task
    async def async_add(a, b, c, d):
        await asyncio.sleep(0.1)
        return a + b + c + d, _current_context.get()

    with LazyContext() as ctx:
        # `ctx.gather` to execute task simutaneously.
        add_fut = add.submit(1, 2, 3, 4)
        async_add_fut = async_add.submit(1, 2, 3, 4)
        start = time.time()
        ctx.gather(add_fut, async_add_fut)
        add_ret, sync_ctx = add_fut.result()
        async_add_ret, async_ctx = async_add_fut.result()
        end = time.time()
        assert add_ret == 10
        assert async_add_ret == 10
        assert end - start < 0.2

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
def test_LazyContext_TaskOnce_with_dag_topo():
    @task
    def add(a, b, c, d):
        time.sleep(0.1)
        return a + b + c + d

    @task
    async def async_add(a, b, c, d):
        await asyncio.sleep(0.1)
        return a + b + c + d

    with LazyContext() as ctx:
        add_fut1 = add.submit(1, 2, 3, 4)
        add_fut2 = add.submit(1, 2, 3, 4)
        async_add_fut1 = async_add.submit(add_fut1, 2, 3, 4)
        async_add_fut2 = async_add.submit(add_fut1, 2, 3, 4)
        start = time.time()
        total = add(add_fut1, add_fut2, async_add_fut1, async_add_fut2)
        end = time.time()
        assert total  == 58
        assert end - start < 0.4


# %%
def test_LazyContext_nested_sync_task():
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
        add_fut = add.submit(a, b, 3, 4)
        mul_fut = mul.submit(1, add_fut, c = 3, d = add_fut)
        add_ret = add_fut.result()
        mul_ret = mul_fut.result()
        return add_ret + mul_ret

    @task
    def mix(a, b):
        add_ret = add(a, b, 3, 4)
        mul_ret = mul(1, add_ret, c = 3, d = add_ret)
        return add_ret + mul_ret

    with LazyContext() as ctx:
        # Task can't be submited as a task in another task by default.
        with pytest.raises(RuntimeError):
            mix_error(1, 2)

        # Nested task.
        mix_fut = mix.submit(1, 2)
        ctx.gather(mix_fut)
        time.sleep(0.2)
        start = time.time()
        mix_ret = mix_fut.result()
        end = time.time()
        assert mix_ret == 310
        assert end - start < 0.1

    # The task nested in another task will be treated as a normal function.
    # So, only `mix_error` and `mix` will be set in `ctx.result_cache`.
    assert len(ctx.result_cache.list_keys()) == 2


# %%
def test_LazyContext_nested_async_func_and_coro():
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

    with LazyContext() as ctx:
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
    # So, only `mix_loop` and `mix_gather` will be set in `ctx.result_cache`.
    assert len(ctx.result_cache.list_keys()) == 2


# %%
def test_LazyContext_nested_context():
    @task
    def add(a, b, c, d):
        time.sleep(0.1)
        return a + b + c + d, _current_context.get()

    @task
    async def async_add(a, b, c, d):
        await asyncio.sleep(0.1)
        return a + b + c + d, _current_context.get()

    with LazyContext() as outer_ctx:
        add_ret, sync_ctx = add(1, 2, 3, 4)
        async_add_ret, async_ctx = async_add(1, 2, 3, 4)
        assert add_ret == 10
        assert sync_ctx is None
        assert async_add_ret == 10
        assert async_ctx is None
        with LazyContext(parent_context = outer_ctx) as inner_ctx:
            add_ret, sync_ctx = add(1, 2, 3, 4)
            async_add_ret, async_ctx = async_add(1, 2, 3, 4)
            assert add_ret == 10
            assert sync_ctx is None
            assert async_add_ret == 10
            assert async_ctx is None
            assert inner_ctx.executor is outer_ctx.executor
            assert inner_ctx._executor is None

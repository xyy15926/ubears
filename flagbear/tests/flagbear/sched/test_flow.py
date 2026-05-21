#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_flow.py
#   Author: xyy15926
#   Created: 2026-05-10 22:24:51
#   Updated: 2026-05-21 14:21:34
#   Description:
# ---------------------------------------------------------

# %%
import pytest
import time
import asyncio
from datetime import timedelta

if __name__ == "__main__":
    from importlib import reload
    from flagbear.slp import cache
    reload(cache)
    from flagbear.sched import protocols, task, executor, scheduler, context, flow
    reload(protocols)
    reload(task)
    reload(executor)
    reload(scheduler)
    reload(context)
    reload(flow)

from flagbear.slp.cache import CachePolicy, MemoryCache
from flagbear.sched.protocols import _current_context
from flagbear.sched.task import task
from flagbear.sched.flow import Flow, flow
# from IPython.core.debugger import set_trace


# %%
def test_Flow_nested_with():
    @task
    def add(a, b, c, d):
        time.sleep(0.1)
        return a + b + c + d, _current_context.get()

    @task
    async def async_add(a, b, c, d):
        await asyncio.sleep(0.1)
        return a + b + c + d, _current_context.get()

    with Flow("outer_flow") as outer_ctx:
        add_ret, sync_ctx = add(1, 2, 3, 4)
        async_add_ret, async_ctx = async_add(1, 2, 3, 4)
        assert add_ret == 10
        assert sync_ctx is None
        assert async_add_ret == 10
        assert async_ctx is None
        with Flow("inner_flow") as inner_ctx:
            add_ret, sync_ctx = add(1, 2, 3, 4)
            async_add_ret, async_ctx = async_add(1, 2, 3, 4)
            assert add_ret == 10
            assert sync_ctx is None
            assert async_add_ret == 10
            assert async_ctx is None
            assert inner_ctx.scheduler is outer_ctx.scheduler
            assert inner_ctx._scheduler is None
            assert inner_ctx.parent_context is outer_ctx


# %%
def test_flow():
    @task
    def add(a, b, c, d):
        time.sleep(0.1)
        return a + b + c + d

    @task
    async def async_add(a, b, c, d):
        await asyncio.sleep(0.1)
        return a + b + c + d

    @flow
    def inner(a, b):
        once = add.submit(1, 1, a, b)
        return async_add(once, 1, 1, 1)

    @flow
    def outer(a, b):
        once = add.submit(1, 1, a, b)
        flow_once = inner.submit(1, 1)
        return async_add(once, flow_once, 1, 1)

    start = time.time()
    result = outer(1, 2)
    end = time.time()
    assert result == 14
    assert end - start < 0.4

    # `outer` will be skipped becasue `once` failed.
    with pytest.raises(RuntimeError):
        _result = outer(1, "a")


# %%
def test_flow_inner_task_with_policy_name_cache_policy():
    @task
    def add(a, b, c, d):
        time.sleep(0.1)
        return a + b + c + d

    cache_policy = CachePolicy(ttl = timedelta(1))
    mem_cache = MemoryCache()
    @flow(task_results = mem_cache)
    def outer_with_cache(a, b):
        once = add.with_policy(
            name = "ka",
            cache_policy = cache_policy,
        ).run(1, 1, a, b)
        once2 = add.with_policy(name = "ka").run(1, 1, a, b)
        return once + once2

    start = time.time()
    result = outer_with_cache(1, 2)
    end = time.time()
    assert result == 10
    assert len(mem_cache.list_keys("ka")) == 3
    assert end - start < 0.2
    assert end - start > 0.1

    # Different task name will lead cache to miss.
    cache_policy = CachePolicy(ttl = timedelta(1))
    mem_cache = MemoryCache()
    @flow(task_results = mem_cache)
    def outer_with_cache(a, b):
        once = add.with_policy(
            name = "ka",
            cache_policy = cache_policy,
        ).run(1, 1, a, b)
        once2 = add.run(1, 1, a, b)
        return once + once2

    start = time.time()
    result = outer_with_cache(1, 2)
    end = time.time()
    assert result == 10
    assert len(mem_cache.list_keys("ka")) == 2
    assert end - start < 0.3
    assert end - start > 0.2

    # `cache_policy(ttl = 0)` will lead cache to expire.
    cache_policy = CachePolicy(ttl = timedelta(0))
    mem_cache = MemoryCache()
    @flow(task_results = mem_cache)
    def outer_without_cache(a, b):
        once = add.with_policy(
            name = "ka",
            cache_policy = cache_policy,
        ).run(1, 1, a, b)
        once2 = add.with_policy(name = "ka").run(1, 1, a, b)
        return once + once2

    start = time.time()
    result = outer_without_cache(1, 2)
    end = time.time()
    assert result == 10
    assert end - start < 0.3
    assert end - start > 0.2
    assert len(mem_cache.list_keys("ka")) == 3


# %%
def test_flow_inner_flow_with_policy_name_cache_policy():
    @task
    def add(a, b, c, d):
        time.sleep(0.1)
        return a + b + c + d

    @task
    async def async_add(a, b, c, d):
        await asyncio.sleep(0.1)
        return a + b + c + d

    @flow
    def inner(a, b):
        once = add.submit(1, 1, a, b)
        return async_add(once, 1, 1, 1)

    # Cache will miss for `once` and `once2` for different name.
    @flow
    def outer_immediate(a, b):
        once = add.with_policy(name = "ka").run(1, 1, a, b)
        once2 = add.with_policy(name = "ya").run(1, 1, a, b)
        flow_once = inner.run(1, 1)
        flow_once2 = inner.run(1, 1)
        return async_add(once, flow_once, once2, flow_once2)

    start = time.time()
    result = outer_immediate(1, 2)
    end = time.time()
    assert result == 24
    assert end - start < 0.6
    assert end - start > 0.5

    # But `cache_policy(ttl = 0)` will lead cache to expire.
    cache_policy = CachePolicy(ttl = timedelta(0))
    mem_cache = MemoryCache()
    @flow(task_results = mem_cache)
    def outer_immediate_with_expired_cache(a, b):
        once = add.with_policy(
            name = "ka",
            cache_policy = cache_policy,
        ).run(1, 1, a, b)
        once2 = add.with_policy(name = "ka").run(1, 1, a, b)
        flow_once = inner.with_policy(
            name = "inner",
            cache_policy = cache_policy,
        ).run(1, 1)
        flow_once2 = inner.with_policy(name = "inner").run(1, 1)
        return async_add(once, flow_once, once2, flow_once2)

    start = time.time()
    result = outer_immediate_with_expired_cache(1, 2)
    end = time.time()
    assert result == 24
    assert end - start < 0.6
    assert end - start > 0.5

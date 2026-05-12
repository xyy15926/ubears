#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_flow.py
#   Author: xyy15926
#   Created: 2026-05-10 22:24:51
#   Updated: 2026-05-12 22:32:25
#   Description:
# ---------------------------------------------------------

# %%
import pytest
import time
import asyncio

if __name__ == "__main__":
    from importlib import reload
    from flagbear.sched import protocols, task, executor, context, flow
    reload(protocols)
    reload(task)
    reload(executor)
    reload(context)
    reload(flow)

from flagbear.sched.protocols import _current_context
from flagbear.sched.task import task
from flagbear.sched.flow import Flow, flow

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
            assert inner_ctx.executor is outer_ctx.executor
            assert inner_ctx._executor is None
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

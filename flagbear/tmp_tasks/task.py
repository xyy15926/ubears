#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: task.py
#   Author: xyy15926
#   Created: 2026-04-23 17:09:42
#   Updated: 2026-05-06 15:00:08
#   Description:
# ---------------------------------------------------------

# %%
import logging
from typing import Dict, List, Callable, Any, Optional, Self, Tuple, Set, Protocol, Type
import functools
import inspect
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass, field, asdict
from enum import Enum
import threading
import uuid
from IPython.core.debugger import set_trace
import asyncio
import contextvars

if __name__ == "__main__":
    from importlib import reload
    from flagbear.tree import dag
    from flagbear.slp import finer, storage, serializer, cache, checkpoint
    reload(dag)
    reload(finer)
    reload(storage)
    reload(serializer)
    reload(cache)
    reload(checkpoint)
from flagbear.tree.dag import Node, DirectedGraph, NID, topological_sort
from flagbear.slp.storage import LocalFileStorage
from flagbear.slp.cache import CachePolicy, CacheMeta, PersistentCache, Cache, MemoryCache
from flagbear.slp.checkpoint import CheckpointPolicy, CheckpointManager

logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"),
)
logger = logging.getLogger()
logger.info("Logging Start.")


# %%
# %%
_flow_ctx_stack: contextvars.ContextVar[List[FlowContext]] = contextvars.ContextVar(
    "flow_context_stack",
    default = None,
)


# %%
class FlowContext:
    """Task Flow context.

    `FlowContext` is the core to determine the behavior of `Task.__call__()`
    as its class attribute `_stack` maintains the current flow context
    attaching to some specific `Flow`. Task will call the
    `FlowContext.current()` to check the context when called directly.
    1. If a Task is called within no `FlowContext`, run wrapping function
      directly.
    2. Else, submit the Task itself for further scheduling.
    """
    # Instance attributes.
    def __init__(
        self,
        flow: Flow,
        cache: Optional[Cache] = None,
    ):
        self.flow = flow
        self.parent_context = self.get_parent_context()
        self.task_futures = {}
        self.task_results = cache or MemoryCache()
        self.artifacts = {}
        self.flow_result: Optional[TaskResult] = None
        self._lock = threading.Lock()
        self._engine: Optional[FlowEngine] = None

    @classmethod
    def current(cls):
        global _flow_ctx_stack
        ctx_stack = _flow_ctx_stack.get()
        if ctx_stack is not None and len(ctx_stack) > 0:
            return ctx_stack[-1]
        return None

    @classmethod
    def get_parent_context(cls) -> Optional[FlowContext]:
        global _flow_ctx_stack
        ctx_stack = _flow_ctx_stack.get()
        if ctx_stack is not None and len(ctx_stack) > 0:
            return ctx_stack[-1]
        return None

    def __enter__(self) -> Self:
        ctx_stack = _flow_ctx_stack.get()
        if ctx_stack is None:
            _token = _flow_ctx_stack.set([self, ])
        else:
            ctx_stack.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        ctx_stack = _flow_ctx_stack.get()
        ctx_stack.pop()
        if self._engine is not None:
            self._engine.shutdown()

    @property
    def engine(self):
        if self._engine is not None:
            return self._engine
        if self.parent_context is not None:
            return self.parent_context.engine
        self._engine = FlowEngine()
        return self._engine


    def get_result(self, future: str | TaskFuture) -> Optional[TaskResult]:
        if isinstance(future, TaskFuture):
            key = future.id_
        else:
            key = future
        with self._lock:
            return self.task_results.get(key)

    def set_result(self, future: str | TaskFuture, result: TaskResult) -> None:
        if isinstance(future, TaskFuture):
            cache_policy = future.cache_policy
            key = future.id_
        else:
            cache_policy = None
            key = future.id_
        with self._lock:
            self.task_results.set(key, result, cache_policy)

    def add_future(self, future: TaskFuture) -> "Future":
        key = future.id_
        if key in self.task_futures:
            raise ValueError(f"future {future.id_} already exists.")
        async_fut = self.engine.submit(future, self)
        with self._lock:
            self.task_futures[key] = async_fut
        return async_fut

    def execute_task(self, future: TaskFuture) -> TaskResult:
        key = future.id_
        set_trace()
        if key in self.task_futures:
            async_fut = self.task_futures[key]
        else:
            async_fut = self.add_future(future)
        return self.engine.wait_for(async_fut)

    def set_artifact(self, key: str, value: Any) -> None:
        with self._lock:
            self.artifacts[key] = value

    def get_artifact(self, key: str, default: Any = None) -> Any:
        with self._lock:
            if key in self.artifacts:
                return self.artifacts[key]
            if self.parent_context:
                return self.parent_context.get_artifact(key, default)
            return default

    def process(self):
        levels = topological_sort(self.task_runs)


# %%
class Flow:
    """Working flow constructed by Tasks.

    Attrs:
    -----------------------------
    name: Flow's name.
    """
    def __init__(
        self,
        name: Optional[str] = None,
        func: Optional[Callable] = None,
    ):
        self.func = func
        self.name = name or f"{func.__module__}.{func.__qualname__}"
        functools.update_wrapper(self, func)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        with FlowContext(self) as ctx:
            res = self.func(*args, **kwargs)
            return res

    def __enter__(self) -> FlowContext:
        ctx = FlowContext(self)
        self._ctx = ctx
        return ctx.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._ctx.__exit__()


def flow(
    func: Callable = None,
    *,
    name: Optional[str] = None,
) -> Flow:
    @functools.wraps(func)
    def decorator(ffunc):
        nonlocal name
        return Flow(name, ffunc)
    if func is None:
        return decorator
    return decorator(func)

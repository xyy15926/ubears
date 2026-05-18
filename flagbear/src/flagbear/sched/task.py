#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: task.py
#   Author: xyy15926
#   Created: 2026-05-06 15:15:38
#   Updated: 2026-05-18 18:15:53
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
import logging
from typing import Dict, List, Callable, Any, Optional, Self, Tuple
import functools
import uuid
import asyncio
import inspect
# from IPython.core.debugger import set_trace

if __name__ == "__main__":
    from importlib import reload
    from flagbear.slp import cache
    from flagbear.sched import task, flow, protocols
    reload(cache)
    reload(task)
    reload(flow)
    reload(protocols)

from flagbear.slp.cache import CachePolicy, Cache
from flagbear.sched.protocols import(
    TaskState,
    TaskResult,
    RetryPolicy,
    ExecutionPolicy,
    Context,
    _current_context,
)

logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"),
)
logger = logging.getLogger()
logger.info("Logging Start.")


# %%
class TaskProxy:
    """Proxy of from a callable to a task in a working flow.

    Attrs:
    -------------------------------
    func: Callable to execute to reach a goal.
    name: Task name.
    retry_policy: Retry policy.
    cache_policy: Cache policy.
    execution_policy: Execution policy.
    is_async: If inner function is async.
    """
    def __init__(
        self,
        func: Callable,
        name: Optional[str] = None,
        retry_policy: Optional[RetryPolicy] = None,
        cache_policy: Optional[CachePolicy] = None,
        execution_policy: Optional[ExecutionPolicy] = None,
    ):
        """Init task.

        Params:
        -------------------------------
        func: Callable to execute to reach a goal.
        name: Name of the task that should be unique among a Flow, namely
          the `Node.id_`.
        #TODO
        """
        self.func = func
        self.name = name or f"{func.__module__}.{func.__qualname__}"
        self.retry_policy = retry_policy
        self.cache_policy = cache_policy
        self.execution_policy = execution_policy
        self.is_async = inspect.iscoroutinefunction(func)
        functools.update_wrapper(self, func)

    def __call__(self, *args, **kwargs):
        """Call `self.func`."""
        ctx = _current_context.get()
        if ctx is None:
            return self.func(*args, **kwargs)
        task_once = self.submit(*args, **kwargs)
        return task_once.result()

    def submit(self, *args, **kwargs) -> TaskOnce:
        """Call `self.func`."""
        ctx = _current_context.get()
        if ctx is None:
            raise RuntimeError("Cannot sumbit task within no Flow.")
        task_once = TaskOnce(self, args, kwargs)
        ctx.submit(task_once)
        return task_once

    def __repr__(self):
        return f"Task({self.name})"


# %%
def task(
    func: Callable = None,
    *,
    name: Optional[str] = None,
    retry_policy: Optional[RetryPolicy] = None,
    cache_policy: Optional[CachePolicy] = None,
    execution_policy: Optional[ExecutionPolicy] = None,
) -> TaskProxy:
    def decorator(ffunc):
        nonlocal name, retry_policy, cache_policy, execution_policy
        return TaskProxy(ffunc, name, retry_policy, cache_policy, execution_policy)
    if func is None:
        return decorator
    return decorator(func)


# %%
class TaskOnce:
    def __init__(
        self,
        task: TaskProxy,
        args: List[Any],
        kwargs: Dict[str, Any],
    ):
        self.name = task.name
        self.func = task.func
        self.retry_policy = task.retry_policy
        self.cache_policy = task.cache_policy
        self.execution_policy = task.execution_policy
        self.is_async = task.is_async
        self.id_ = f"{task.name}_{uuid.uuid4()}"
        self.raw_args = args
        self.raw_kwargs = kwargs

    def __hash__(self):
        return hash(self.id_)

    def __eq__(self, rhs: Self):
        return self.id_ == rhs.id_

    @classmethod
    def from_func(
        cls,
        func: Callable,
        args: List[Any],
        kwargs: Dict[str, Any],
        *,
        name: Optional[str] = None,
        retry_policy: Optional[RetryPolicy] = None,
        cache_policy: Optional[CachePolicy] = None,
        execution_policy: Optional[ExecutionPolicy] = None,
    ) -> Self:
        """Create a TaskOnce with function and arguments directly."""
        task = TaskProxy(func, name, retry_policy, cache_policy, execution_policy)
        task_once = cls(task, args, kwargs)
        return task_once

    @classmethod
    def _resolve_with_context(
        cls,
        unresolved: List | Dict,
        cache: Cache,
    ) -> Tuple[Dict, Dict, Dict]:
        tasks = (enumerate(unresolved)
                 if isinstance(unresolved, (tuple, list))
                 else unresolved.items())
        ready = {}
        failed = {}
        unready = []
        for key, task in tasks:
            if not isinstance(task, cls):
                ready[key] = task
            else:
                if cache is not None:
                    result = cache.get(task.id_)
                else:
                    result = None

                # Check the state of the result of task.
                if result is not None and result.is_successful():
                    ready[key] = result.value
                elif result is not None and result.is_failed():
                    failed[task] = result
                else:
                    unready.append(task)

        return ready, unready, failed

    def resolve_args(
        self,
        cache: Optional[Cache] = None,
    ):
        """Resolve the arguments of the task function from the context.

        Note:
        Task as part of the argument, namely an element in a list, tuple, dic
        or etc., won't be resolved.
        """
        if cache is None:
            ctx = _current_context.get()
            if ctx is not None:
                cache = ctx.result_cache
        raw_args = self.raw_args
        raw_kwargs = self.raw_kwargs
        pready, punready, pfailed = self._resolve_with_context(raw_args, cache)
        kready, kunready, kfailed = self._resolve_with_context(raw_kwargs, cache)
        unready = list(set(punready + kunready))
        pfailed.update(kfailed)
        return list(pready.values()), kready, unready, pfailed

    def resolve_dependencies(
        self,
        cache: Optional[Cache] = None,
    ) -> List[TaskOnce]:
        """Resolve the TaskOnce current TaskOnce depending on."""
        if cache is None:
            ctx = _current_context.get()
            if ctx is not None:
                cache = ctx.result_cache
        pready, kready, unready, pfailed = self.resolve_args(cache)
        return unready

    def result(
        self,
        ctx: Optional[Context] = None,
    ) -> Any:
        """Fetch the result."""
        ctx = ctx or _current_context.get()
        if ctx is None:
            raise RuntimeError(
                f"Can't fetch the result of task {self.name} without a flow."
            )
        result = ctx.get_result(self.id_)
        if result is None:
            raise RuntimeError(
                f"Can't fetch the result of task {self.name}."
            )

        if result.is_successful():
            return result.value
        if result.is_failed():
            raise RuntimeError(f"Task {self.name} failed.") from result.error
        raise RuntimeError(
            f"Task {self.name} can't be ready."
        ) from result.error

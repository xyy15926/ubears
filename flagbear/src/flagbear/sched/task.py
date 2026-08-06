#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: task.py
#   Author: xyy15926
#   Created: 2026-05-06 15:15:38
#   Updated: 2026-05-21 14:37:43
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:
    from collections.abc import Callable

# from IPython.core.debugger import set_trace

if __name__ == "__main__":
    from importlib import reload

    from flagbear.slp import cache, checkpoint

    reload(cache)
    reload(checkpoint)
    from flagbear.sched import flow, protocols, task

    reload(task)
    reload(flow)
    reload(protocols)

from flagbear.sched.protocols import (
    Context,
    ExecutionPolicy,
    RetryPolicy,
    TaskProxyBase,
    _current_context,
)
from flagbear.slp.checkpoint import CheckpointPolicy

if TYPE_CHECKING:
    from flagbear.slp.cache import Cache, CachePolicy

logger = logging.getLogger(__name__)


# %%
class TaskProxy(TaskProxyBase):
    """Proxy of from a callable to a task in a working flow.

    Attrs:
    -------------------------------
    func: Callable to execute to reach a goal.
    _default_policies: Default policies.
      name: Name of the task that will be used to construct the `Task.id_`
        and `Task.cache_key`.
      cache_policy: Cache policy.
      checkpoint_policy: Checkpoint policy
      retry_policy: Retry policy.
      execution_policy: Execution policy.
    _current_policies: Policies updated by `with_policy` should should be
      reset to be the same as `_default_policies` once a Task has been
      inited.
      Policies in `_current_context` could be accessed just like normal
      attributes.
    is_async: If inner function is async.
    """

    def __call__(self, *args, **kwargs) -> Any:
        """Call `self.func`."""
        return self.run(*args, **kwargs)

    def run(self, *args, **kwargs) -> Any:
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

        # Reset setting such as `self.tmp_name`.
        self.reset_policy()
        ctx.submit(task_once)
        return task_once

    def __repr__(self):
        """Return repr."""
        return f"Task({self.name})"


# %%
def task(
    func: Callable | None = None,
    *,
    name: str | None = None,
    cache_policy: CachePolicy | None = None,
    checkpoint_policy: CheckpointPolicy | None = None,
    retry_policy: RetryPolicy | None = None,
    execution_policy: ExecutionPolicy | None = None,
) -> TaskProxy:
    """Create a TaskProxy from a callable."""

    def decorator(ffunc):
        nonlocal name, cache_policy, checkpoint_policy
        nonlocal retry_policy, execution_policy
        return TaskProxy(
            ffunc,
            name,
            cache_policy,
            checkpoint_policy,
            retry_policy,
            execution_policy,
        )

    if func is None:
        return decorator
    return decorator(func)


# %%
class TaskOnce:
    """Task that should be executed only once.

    Attrs:
    --------------------------
    name: Task name.
    func: Callable to execute to reach a goal.
    name: Task name.
    cache_policy: Cache policy.
    checkpoint_policy: Checkpoint policy
    retry_policy: Retry policy.
    execution_policy: Execution policy.
    is_async: If inner function is async.
    cache_key: Unique key generated with `checkpoint_policy` for cache.
      Cache key can only be determined after all raw arguments has been
      resolved.
      So an another `id_` is used to identify the `TaskOnce`.
      Also `cache_key` and `id_` could serperate the `TaskResult` with
      the result of the inner function.
    id_: Unique identity of the `TaskOnce`.
    raw_args: Raw arguments that may contain other `TaskOnce`s to be resolved.
    raw_kwargs: Raw arguments that may contain other `TaskOnce`s to be
      resolved.
    """

    def __init__(
        self,
        task: TaskProxy,
        args: list[Any],
        kwargs: dict[str, Any],
    ):
        self.func = task.func
        self.name = task.name
        self.cache_policy = task.cache_policy
        self.checkpoint_policy = task.checkpoint_policy
        self.retry_policy = task.retry_policy
        self.execution_policy = task.execution_policy
        self.is_async = task.is_async
        self._cache_key = None
        self.id_ = f"{self.name}_{uuid.uuid4()}"
        self.raw_args = args
        self.raw_kwargs = kwargs

    def __hash__(self):
        """Return hash."""
        return hash(self.id_)

    def __eq__(self, rhs: Self):
        """Return equality."""
        return self.id_ == rhs.id_

    @property
    def cache_key(self):
        """Check if cache key has been set before get."""
        if self._cache_key is None:
            self.resolve_args()
        if self._cache_key is None:
            raise RuntimeError(
                f"Cache key of task {self.name} hasn't been set."
            )
        return self._cache_key

    @classmethod
    def from_func(
        cls,
        func: Callable,
        args: list[Any],
        kwargs: dict[str, Any],
        *,
        name: str | None = None,
        cache_policy: CachePolicy | None = None,
        checkpoint_policy: CheckpointPolicy | None = None,
        retry_policy: RetryPolicy | None = None,
        execution_policy: ExecutionPolicy | None = None,
    ) -> Self:
        """Create a TaskOnce with function and arguments directly."""
        task = TaskProxy(
            func,
            name,
            cache_policy,
            checkpoint_policy,
            retry_policy,
            execution_policy,
        )
        task_once = cls(task, args, kwargs)
        return task_once

    @classmethod
    def _resolve_with_context(
        cls,
        unresolved: list | dict,
        cache: Cache,
    ) -> tuple[dict, dict, dict]:
        tasks = (
            enumerate(unresolved)
            if isinstance(unresolved, (tuple, list))
            else unresolved.items()
        )
        ready = {}
        failed = {}
        unready = []
        for key, task in tasks:
            if not isinstance(task, cls):
                ready[key] = task
            else:
                result = cache.get(task.id_) if cache is not None else None

                # Check the state of the result of task.
                if result is not None and result.is_successful():
                    ready[key] = cache.get(task.cache_key)
                elif result is not None and result.is_failed():
                    failed[task] = result
                else:
                    unready.append(task)

        return ready, unready, failed

    def resolve_args(
        self,
        cache: Cache | None = None,
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
        kready, kunready, kfailed = self._resolve_with_context(
            raw_kwargs, cache
        )
        unready = list(set(punready + kunready))
        pfailed.update(kfailed)
        pready = list(pready.values())

        # Set `cache_key` if all raw arguments has been resolved.
        if len(unready) == 0 and len(pfailed) == 0 and self._cache_key is None:
            checkpoint_policy = self.checkpoint_policy or CheckpointPolicy()
            self._cache_key = checkpoint_policy.gen_key(
                self.name, pready, kready
            )

        return pready, kready, unready, pfailed

    def resolve_dependencies(
        self,
        cache: Cache | None = None,
    ) -> list[TaskOnce]:
        """Resolve the TaskOnce current TaskOnce depending on.

        Sometimes, it's meant to get all dependencies without cache taken
        into consideration may be needed, so don't use
        `_current_context.get().task_results` here.
        """
        raw_args = self.raw_args
        raw_kwargs = self.raw_kwargs
        _pready, punready, _pfailed = self._resolve_with_context(
            raw_args, cache
        )
        _kready, kunready, _kfailed = self._resolve_with_context(
            raw_kwargs, cache
        )
        unready = list(set(punready + kunready))
        return unready

    def result(
        self,
        ctx: Context | None = None,
    ) -> Any:
        """Fetch the result."""
        ctx = ctx or _current_context.get()
        if ctx is None:
            raise RuntimeError(
                f"Can't fetch the result of task {self.name} without a flow."
            )
        result = ctx.get_result(self)
        if result is None:
            raise RuntimeError(f"Can't fetch the result of task {self.name}.")

        if result.is_successful():
            return result.value
        if result.is_failed():
            raise RuntimeError(f"Task {self.name} failed.") from result.error
        raise RuntimeError(
            f"Task {self.name} can't be ready."
        ) from result.error

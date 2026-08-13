#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: flow.py
#   Author: xyy15926
#   Created: 2026-05-06 16:02:03
#   Updated: 2026-05-22 22:16:55
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

# from IPython.core.debugger import set_trace

if __name__ == "__main__":
    from importlib import reload

    from flagbear.slp import cache, checkpoint

    reload(cache)
    reload(checkpoint)
    from flagbear.sched import context, executor, protocols, scheduler

    reload(protocols)
    reload(executor)
    reload(scheduler)
    reload(context)
from flagbear.sched.context import SimpleContext
from flagbear.sched.protocols import (
    Context,
    ExecutionPolicy,
    RetryPolicy,
    Scheduler,
    TaskProxyBase,
    _current_context,
)
from flagbear.sched.task import TaskOnce

if TYPE_CHECKING:
    from flagbear.slp.cache import Cache, CachePolicy
    from flagbear.slp.checkpoint import CheckpointPolicy

logger = logging.getLogger(__name__)


# %%
class Flow(TaskProxyBase):
    """Working flow of tasks.

    Attrs:
    -----------------------------
    name: Flow name.
    func: The function body of the flow.
    cache_policy: Cache policy of result.
    retry_policy: Retry policy of the flow as a task.
    execution_policy: Execution policy of the flow as a task.
    task_results: Cache to store the result of inner tasks.
    scheduler: scheduler to execute the task.
    tmp_name: Temperary name of the task that will be used to construct
      the `Task.id_` and `Task.cache_key`.
      Temperary name should be reset once a Task has been constructed.
    tmp_cache_policy: Temperary cache policy.
    tmp_checkpoint_policy: Temperary checkpoint policy
    tmp_retry_policy: Temperary retry policy.
    tmp_execution_policy: Temperary execution policy.
    """

    def __init__(
        self,
        name: str | None = None,
        func: Callable | None = None,
        cache_policy: CachePolicy | None = None,
        checkpoint_policy: CheckpointPolicy | None = None,
        retry_policy: RetryPolicy | None = None,
        execution_policy: ExecutionPolicy | None = None,
        task_results: Cache | None = None,
        scheduler: Scheduler | None = None,
    ):
        """Init the flow.

        Params:
        -----------------------------
        name: Flow name.
        func: The function body of the flow.
        cache_policy: Cache policy of the value of TaskResult.
        checkpoint_policy: Checkpoint policy of the value of TaskResult.
        retry_policy: Retry policy of the flow as a task.
        execution_policy: Execution policy of the flow as a task.
        task_results: Cache to store the result of inner tasks.
        scheduler: scheduler to execute the task.
        """
        if func is None and name is None:
            raise ValueError("One of flow name or function must be provided.")
        super().__init__(
            func,
            name or f"Flow.{func.__module__}.{func.__qualname__}",
            cache_policy,
            checkpoint_policy,
            retry_policy,
            execution_policy,
        )
        self.task_results = task_results
        self.scheduler = scheduler
        functools.update_wrapper(self, func)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the flow directly."""
        return self.run(*args, **kwargs)

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Call the flow directly.

        A context will be constructed execute the flow.
        1. If no context is set currently, current flow should be the top
          flow.And then execute the flow under the context newly-inited
          directly.
        2. Else wrap and submit current flow as a task to current context.
        """
        parent_ctx = _current_context.get()
        if parent_ctx is None:
            ctx = SimpleContext(
                self.task_results,
                self.scheduler,
                parent_ctx,
                name=self.name,
            )
            with ctx:
                return self.func(*args, **kwargs)
        flow_once = self.submit(*args, **kwargs)
        return flow_once.result()

    def submit(self, *args: Any, **kwargs: Any) -> TaskOnce:
        """Wrap and submit flow as a task to current context."""
        parent_ctx = _current_context.get()
        if parent_ctx is None:
            raise RuntimeError("Can's sumbit Flow as sub-flow within no Flow.")
        ctx = SimpleContext(
            self.task_results,
            self.scheduler,
            parent_ctx,
            name=self.name,
        )
        flow_once = self.as_task(args, kwargs, ctx)
        # Reset policy after `TaskOnce` has been constructed.
        self.reset_policy()
        parent_ctx.submit(flow_once)
        return flow_once

    def as_task(
        self,
        args: list[Any],
        kwargs: dict[str, Any],
        ctx: SimpleContext,
    ) -> TaskOnce:
        """Wrap flow as a task."""
        func = self.func

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal ctx, func
            # Flow should be run under newly-inited context.
            with ctx:
                ret = func(*args, **kwargs)
                # Just in case.
                if isinstance(ret, TaskOnce):
                    logger.warning(
                        "No TaskOnce should be returned by a function."
                    )
                    return ret.result()
                return ret

        flow_once = TaskOnce.from_func(
            wrapper,
            args,
            kwargs,
            name=self.name,
            cache_policy=self.cache_policy,
            checkpoint_policy=self.checkpoint_policy,
            retry_policy=self.retry_policy,
            execution_policy=self.execution_policy,
        )
        flow_once.id_ = ctx.id_

        return flow_once

    def __enter__(self) -> Context:
        """Enter the newly-inited context."""
        if self.func is not None:
            raise RuntimeError(
                "Only empty Flow could be used as a task container."
            )
        ctx = SimpleContext(
            self.task_results,
            self.scheduler,
            _current_context.get(),
            name=self.name,
        )
        self._ctx = ctx
        return ctx.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the newly-inited context."""
        self._ctx.__exit__(exc_type, exc_val, exc_tb)


# %%
def flow(
    func: Callable | None = None,
    *,
    name: str | None = None,
    cache_policy: CachePolicy | None = None,
    checkpoint_policy: CheckpointPolicy | None = None,
    retry_policy: RetryPolicy | None = None,
    execution_policy: ExecutionPolicy | None = None,
    task_results: Cache | None = None,
    scheduler: Scheduler | None = None,
) -> Flow:
    """Decorator to wrap a function as a Flow.

    Params:
    ---------------------------
    func: The function body of the flow.
    name: Flow name.
    cache_policy: Cache policy of the value of TaskResult.
    checkpoint_policy: Checkpoint policy of the value of TaskResult.
    retry_policy: Retry policy of the flow as a task.
    execution_policy: Execution policy of the flow as a task.
    task_results: Cache to store the result of inner tasks.
    scheduler: scheduler to execute the task.

    Return:
    ---------------------------
    Flow
    """

    def decorator(ffunc):
        # nonlocal name, retry_policy, cache_policy, execution_policy
        # nonlocal task_results, scheduler
        return Flow(
            name,
            ffunc,
            cache_policy,
            checkpoint_policy,
            retry_policy,
            execution_policy,
            task_results,
            scheduler,
        )

    if func is None:
        return decorator
    return decorator(func)

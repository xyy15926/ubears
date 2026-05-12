#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: flow.py
#   Author: xyy15926
#   Created: 2026-05-06 16:02:03
#   Updated: 2026-05-12 22:31:03
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
import logging
from typing import List, Callable, Any, Optional, Dict
import functools
# from IPython.core.debugger import set_trace

if __name__ == "__main__":
    from importlib import reload
    from flagbear.slp import cache
    from flagbear.sched import protocols, executor, context
    reload(cache)
    reload(protocols)
    reload(executor)
    reload(context)
from flagbear.sched.protocols import(
    Context,
    RetryPolicy,
    ExecutionPolicy,
    _current_context,
)
from flagbear.slp.cache import Cache, CachePolicy
from flagbear.sched.task import TaskOnce, Task
from flagbear.sched.context import LazyContext

logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"),
)
logger = logging.getLogger()
logger.info("Logging Start.")

# %%
class Flow:
    """Working flow of tasks.

    Attrs:
    -----------------------------
    name: Flow name.
    func: The function body of the flow.
    retry_policy: Retry policy of the flow as a task.
    cache_policy: Cache policy of result.
    execution_policy: Execution policy of the flow as a task.
    result_cache: Cache to store the result of inner tasks.
    executor: Executor to execute the task.
    """
    def __init__(
        self,
        name: Optional[str] = None,
        func: Optional[Callable] = None,
        retry_policy: Optional[RetryPolicy] = None,
        cache_policy: Optional[CachePolicy] = None,
        execution_policy: Optional[ExecutionPolicy] = None,
        result_cache: Optional[Cache] = None,
        executor: Optional[Cache] = None,
    ):
        """Init the flow.

        Params:
        -----------------------------
        name: Flow name.
        func: The function body of the flow.
        retry_policy: Retry policy of the flow as a task.
        cache_policy: Cache policy of result.
        execution_policy: Execution policy of the flow as a task.
        result_cache: Cache to store the result of inner tasks.
        executor: Executor to execute the task.
        """
        if func is None and name is None:
            raise ValueError("One of flow name or function must be provided.")
        self.func = func
        self.name = name or f"Flow.{func.__module__}.{func.__qualname__}"
        self.retry_policy = retry_policy
        self.cache_policy = cache_policy
        self.execution_policy = execution_policy
        self.result_cache = result_cache
        self.executor = executor
        functools.update_wrapper(self, func)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the flow directly.

        A context will be constructed execute the flow.
        1. If no context is set currently, current flow should be the top
          flow.And then execute the flow under the context newly-inited
          directly.
        2. Else wrap and submit current flow as a task to current context.
        """
        parent_ctx = _current_context.get()
        if parent_ctx is None:
            ctx = LazyContext(self.result_cache, self.executor, parent_ctx)
            with ctx:
                return self.func(*args, **kwargs)
        flow_once = self.submit(*args, **kwargs)
        return flow_once.result()

    def submit(self, *args: Any, **kwargs: Any) -> TaskOnce:
        """Wrap and submit flow as a task to current context."""
        parent_ctx = _current_context.get()
        if parent_ctx is None:
            raise RuntimeError(
                "Can's sumbit Flow as sub-flow within no Flow."
            )
        ctx = LazyContext(self.result_cache, self.executor, parent_ctx)
        flow_once = self.as_task(args, kwargs, ctx)
        parent_ctx.submit(flow_once)
        return flow_once

    def as_task(
        self,
        args: List[Any],
        kwargs: Dict[str, Any],
        ctx: LazyContext,
    ) -> TaskOnce:
        """Wrap flow as a task."""
        func = self.func

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal ctx, func
            # Flow should be run under newly-inited context.
            with ctx:
                return func(*args, **kwargs)

        wrapped_flow = Task(
            wrapper,
            self.name,
            self.retry_policy,
            self.cache_policy,
            self.execution_policy,
        )
        flow_once = TaskOnce(wrapped_flow, args, kwargs)

        return flow_once

    def __enter__(self) -> Context:
        """Enter the newly-inited context."""
        assert self.func is None, (
            "Only empty Flow could be used as a task container."
        )
        ctx = LazyContext(
            self.result_cache,
            self.executor,
            _current_context.get(),
        )
        self._ctx = ctx
        return ctx.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the newly-inited context."""
        self._ctx.__exit__(exc_type, exc_val, exc_tb)


# %%
def flow(
    func: Callable = None,
    *,
    name: Optional[str] = None,
    retry_policy: Optional[RetryPolicy] = None,
    cache_policy: Optional[CachePolicy] = None,
    execution_policy: Optional[ExecutionPolicy] = None,
    result_cache: Optional[Cache] = None,
    executor: Optional[Cache] = None,
) -> Flow:
    """Decorator to wrap a function as a Flow.

    Params:
    ---------------------------
    func: The function body of the flow.
    name: Flow name.
    retry_policy: Retry policy of the flow as a task.
    cache_policy: Cache policy of result.
    execution_policy: Execution policy of the flow as a task.
    result_cache: Cache to store the result of inner tasks.
    executor: Executor to execute the task.

    Return:
    ---------------------------
    Flow
    """
    def decorator(ffunc):
        # nonlocal name, retry_policy, cache_policy, execution_policy
        # nonlocal result_cache, executor
        return Flow(
            name,
            ffunc,
            retry_policy,
            cache_policy,
            execution_policy,
            result_cache,
            executor,
        )
    if func is None:
        return decorator
    return decorator(func)

#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: context.py
#   Author: xyy15926
#   Created: 2026-05-10 14:06:35
#   Updated: 2026-05-22 22:07:32
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations

import logging
import uuid
from typing import Self

# from IPython.core.debugger import set_trace

if __name__ == "__main__":
    from importlib import reload

    from flagbear.tree import dag

    reload(dag)
    from flagbear.slp import cache, finer, serializer, storage

    reload(finer)
    reload(storage)
    reload(serializer)
    reload(cache)
    from flagbear.sched import executor, protocols, scheduler

    reload(protocols)
    reload(executor)
    reload(scheduler)

from flagbear.sched.executor import LocalExecutor
from flagbear.sched.protocols import (
    Context,
    Scheduler,
    Task,
    TaskResult,
    _current_context,
)
from flagbear.sched.scheduler import DAGScheduler
from flagbear.slp.cache import Cache, MemoryCache

logger = logging.getLogger(__name__)


# %%
class SimpleContext:
    """Simple context.

    Simple context manage a cache to store results of tasks, a scheduler to
    execute task.

    Attrs:
    ----------------------------
    task_results: Cache to store the result of the task.
    scheduler: Scheduler to schedule and run tasks.
    parent_context: Parent Contex.
    name: Name to be distinct from other contexts.
    id_: ID to be distinct from other contexts.
    """

    def __init__(
        self,
        task_results: Cache | None = None,
        scheduler: Scheduler | None = None,
        parent_context: Context | None = None,
        *,
        name: str = "Root",
    ):
        self._task_results = task_results
        self._scheduler = scheduler
        self.parent_context = parent_context
        self.name = name
        self.id_ = f"{self.name}_{uuid.uuid4()}"

    def __enter__(self) -> Self:
        """Set current context."""
        self._context_token = _current_context.set(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Revert current context."""
        _current_context.reset(self._context_token)
        self.shutdown()

    def shutdown(self, force: bool = False):
        """Shutdown the context."""
        if self._scheduler is not None:
            self._scheduler.shutdown(force)

    @property
    def task_results(self):
        """Share the same cache of results of tasks or init a new cache."""
        if self._task_results is not None:
            return self._task_results
        if self.parent_context is not None:
            return self.parent_context.task_results
        self._task_results = MemoryCache()
        return self._task_results

    @property
    def scheduler(self):
        """Get an scheduler from parent context or init a new scheduler."""
        if self._scheduler is not None:
            return self._scheduler
        if self.parent_context is not None:
            return self.parent_context.scheduler
        local_exec = LocalExecutor(4, 0)
        self._scheduler = DAGScheduler(self.task_results, local_exec)
        return self._scheduler

    def get_result(
        self,
        task: Task | str,
    ) -> TaskResult | None:
        """Get result of specified task."""
        self.scheduler.wait(task)
        tid = getattr(task, "id_", task)
        task_result = self.task_results.get(tid)
        return task_result

    def set_result(
        self,
        task: Task,
        result: TaskResult,
    ):
        """Set the result of specified task in `task_results`."""
        self.task_results.set(
            task.cache_key,
            result,
            task.cache_policy,
        )

    def submit(self, *tasks: Task):
        """Submit tasks to scheduler."""
        self.scheduler.add(*tasks, ctx_name=self.id_)

    def run(self, *tasks: Task):
        """Run tasks with scheduler."""
        self.submit(*tasks)
        results = [self.get_result(task) for task in tasks]
        return results

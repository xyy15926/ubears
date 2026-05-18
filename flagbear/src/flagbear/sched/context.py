#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: context.py
#   Author: xyy15926
#   Created: 2026-05-10 14:06:35
#   Updated: 2026-05-18 22:14:19
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
import logging
from typing import Optional, Self
import threading
# from IPython.core.debugger import set_trace

if __name__ == "__main__":
    from importlib import reload
    from flagbear.tree import dag
    reload(dag)
    from flagbear.slp import finer, storage, serializer, cache
    reload(finer)
    reload(storage)
    reload(serializer)
    reload(cache)
    from flagbear.sched import protocols, executor, scheduler
    reload(protocols)
    reload(executor)
    reload(scheduler)

from flagbear.slp.cache import Cache, MemoryCache
from flagbear.sched.protocols import(
    TaskResult,
    Task,
    Context,
    _current_context,
    Scheduler,
)
from flagbear.sched.scheduler import DAGScheduler

logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"),
)
logger = logging.getLogger()
logger.info("Logging Start.")


# %%
class SimpleContext:
    """Simple context

    Task submited in the context will be deferred until `ctx.gathered` is
    called explicitly.

    Attrs:
    ----------------------------
    task_results: Cache to store the result of the task.
    scheduler: Scheduler to schedule and run tasks.
    parent_context: Parent Contex.
    """
    def __init__(
        self,
        task_results: Optional[Cache] = None,
        scheduler: Optional[Scheduler] = None,
        parent_context: Optional[Context] = None,
    ):
        self._task_results = task_results
        self._scheduler = scheduler
        self.parent_context = parent_context

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
        self._scheduler = DAGScheduler(self.task_results)
        return self._scheduler

    def get_result(
        self,
        task: str | Task,
    ) -> Optional[TaskResult]:
        """Get result of specified task."""
        tid = getattr(task, "id_", task)
        self.scheduler.wait(tid)
        return self.task_results.get(tid)

    def set_result(
        self,
        task: str | Task,
        result: TaskResult,
    ):
        """Set the result of specified task in `task_results`."""
        key = getattr(task, "id_", task)
        cache_policy = getattr(task, "cache_policy", None)
        self.task_results.set(key, result, cache_policy)

    def submit(self, *tasks: Task):
        self.scheduler.add(*tasks)

    def run(self, *tasks: Task):
        self.submit(*tasks)
        results = [self.get_result(task) for task in tasks]
        return results

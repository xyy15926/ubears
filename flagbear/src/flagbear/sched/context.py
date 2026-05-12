#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: context.py
#   Author: xyy15926
#   Created: 2026-05-10 14:06:35
#   Updated: 2026-05-12 14:07:00
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
import logging
from typing import Any, Optional, Self, Dict
import threading
# from IPython.core.debugger import set_trace

if __name__ == "__main__":
    from importlib import reload
    from flagbear.tree import dag
    from flagbear.slp import finer, storage, serializer, cache
    from flagbear.sched import protocols, executor
    reload(dag)
    reload(finer)
    reload(storage)
    reload(serializer)
    reload(cache)
    reload(protocols)
    reload(executor)
from flagbear.tree.dag import (
    DirectedGraph,
    Node,
    topological_sort_from_entry,
)
from flagbear.slp.cache import Cache, MemoryCache
from flagbear.sched.protocols import(
    TaskResult,
    TaskFuture,
    Executor,
    Context,
    _current_context,
)
from flagbear.sched.executor import LocalExecutor

logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"),
)
logger = logging.getLogger()
logger.info("Logging Start.")


# %%
class LazyContext:
    """Lazy context

    Task submited in the context will be deferred until `ctx.gathered` is
    called explicitly.

    Attrs:
    ----------------------------
    result_cache: Cache to store the result of the task.
    parent_context: Parent Contex.
    dag: DAG of ungathered tasks.
    executor: Executor to run the task.
    _global_lock: Lock to protect task submission from multi-threads.
    artifacts:
    flow_results: 
    """
    def __init__(
        self,
        result_cache: Optional[Cache] = None,
        executor: Optional[Executor] = None,
        parent_context: Optional[Context] = None,
    ):
        self.result_cache = result_cache or MemoryCache()
        self.parent_context = parent_context
        self.dag = DirectedGraph()
        self._executor = executor
        self._global_lock = threading.Lock()
        self.artifacts = {}
        self.flow_result: Optional[TaskResult] = None
    
    def __enter__(self) -> Self:
        """Set current context."""
        self._context_token = _current_context.set(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Revert current context."""
        _current_context.reset(self._context_token)
        if self.dag.node_count > 0:
            nodes = self.dag._nodes.values()
            task_names = ",".join([node.task.name for node in nodes])
            logger.warning(f"Result of task {task_names} are not gathered yet.")
        self.shutdown()

    def shutdown(self):
        """Shutdown the context."""
        if self._executor is not None:
            self._executor.shutdown()

    @property
    def executor(self):
        """Get an executor from parent context or init a new executor."""
        if self._executor is not None:
            return self._executor
        if self.parent_context is not None:
            return self.parent_context.executor
        self._executor = LocalExecutor()
        return self._executor

    def get_result(
        self,
        task_future: str | TaskFuture,
    ) -> Optional[TaskResult]:
        """Get result of specified task."""
        key = getattr(task_future, "id_", task_future)
        return self.result_cache.get(key)

    def set_result(
        self,
        task_future: str | TaskFuture,
        result: TaskResult,
    ):
        """Set the result of specified task in `result_cache`."""
        key = getattr(task_future, "id_", task_future)
        cache_policy = getattr(task_future, "cache_policy", None)
        self.result_cache.set(key, result, cache_policy)

    def submit(
        self,
        *task_futures: TaskFuture,
    ):
        """Submit task.

        1. Untraced tasks will be added to DAG but won't be executed until
          gathering results is required.
        2. ValueError will be raised if a task has been submitted before.
        """
        with self._global_lock:
            # Add tasks to DAG first.
            for tfut in task_futures:
                key = getattr(tfut, "id_", tfut)
                if (self.result_cache.exists(key)
                    or self.dag.has_node(key)):
                    raise ValueError(
                        f"Task {key} has been submited before."
                    )
                cur_node = Node(key)
                cur_node.task = tfut
                self.dag.add_node(cur_node)
            # Then add the edges to the DAG, so to avoid the edges are added
            # before the nodes. As the tasks in `task_features` may not be
            # topological sorted.
            for tfut in task_futures:
                key = getattr(tfut, "id_", tfut)
                deps = tfut.resolve_dependencies(self)
                for dep in deps:
                    dep_key = dep.id_
                    if dep_key in self.dag:
                        self.dag.add_edge(dep_key, key)
                    elif not self.result_cache.exists(dep_key):
                        raise ValueError(
                            f"Task {dep_key} has not been submited before."
                        )

    def gather(
        self,
        *task_futures: TaskFuture,
    ) -> list[Any]:
        """Gather the results of tasks."""
        task_nodes = []
        # Result of the tasks can't be gather twice.
        for tfut in task_futures:
            node = self.dag.get_node(tfut.id_)
            if node is None:
                raise ValueError(
                    f"Result of task {tfut.name} result has been "
                    f"gathered beofore."
                )
            task_nodes.append(node)

        # Sort tasks topologically first so to determine the execution order
        # of tasks, which will also detect the unreasonable, cyclic tasks
        # dependency relations.
        # TODO: Level by level topo -> node by node topo.
        node_levels = topological_sort_from_entry(*task_nodes)
        if node_levels is None:
            cycle = self.dag.find_cycle()
            task_names = ",".join(cycle)
            raise ValueError(f"Tasks {task_names} form a cycle.")

        # Execute tasks.
        # The last `results` in the `for` loop is the last level of the
        # task-DAG.
        for level in node_levels:
            # set_trace()
            tfuts = [node.task for node in level]
            self.executor.submit(self, *tfuts)
            results = self.executor.gather(*tfuts)
            for tfut, res in zip(level, results, strict=True):
                self.set_result(tfut, res)
                with self._global_lock:
                    self.dag.remove_node(tfut.id_)

        results = [self.get_result(tfut) for tfut in task_futures]
        return results

    def run(
        self, 
        *task_futures: TaskFuture,
    ) -> list[Any]:
        """Submit tasks and gather the results."""
        self.submit(*task_futures)
        return self.gather(*task_futures)

    def set_artifact(self, key: str, value: Any) -> None:
        """Set the artifact."""
        with self._global_lock:
            self.artifacts[key] = value

    def get_artifact(self, key: str, default: Any = None) -> Any:
        """Get the artifact."""
        if key in self.artifacts:
            return self.artifacts[key]
        if self.parent_context:
            return self.parent_context.get_artifact(key, default)
        return default

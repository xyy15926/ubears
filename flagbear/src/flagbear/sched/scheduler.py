#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: scheduler.py
#   Author: xyy15926
#   Created: 2026-05-14 15:10:46
#   Updated: 2026-05-21 14:54:35
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
import logging
import threading
import functools
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
    from flagbear.sched import protocols, executor
    reload(protocols)
    reload(executor)
from flagbear.tree.dag import (
    DirectedGraph,
)
from flagbear.slp.cache import Cache, MemoryCache
from flagbear.sched.protocols import(
    Task,
    Future,
    Executor,
)
from flagbear.sched.executor import LocalExecutor

logger = logging.getLogger(__name__)


# %%
class DAGScheduler:
    """Task scheduler with DAG to resolve dependencies relations.

    Attrs:
    --------------------------
    task_results: Cache to store the results of task.
    task_dag: DAG to organize the tasks waiting for results, namely in
      execution or waiting to be executed.
    task_submited: Set to the store the tasks that has been submitted to
      executor.
    executor: Executor to execute the tasks.
    ctx_dags: Dict of DAGs to records all tasks that has been added to current
      scheduler from different contexts.
    """
    def __init__(
        self,
        task_results: Cache | None = None,
        executor: Executor | None = None,
    ):
        self.task_results = task_results or MemoryCache()
        self.task_dag = DirectedGraph()
        self.task_submited = set()
        self._task_lock = threading.Lock()
        self.executor = executor or LocalExecutor()
        self.ctx_dags: dict[str, DirectedGraph] = {}

    def is_added(
        self,
        task: Task,
    ) -> bool:
        """If the task has been submitted before."""
        tid = getattr(task, "id_", task)
        if self.task_dag.has_node(tid) or self.task_results.exists(tid):
            return False
        return True

    def shutdown(self, force = False):
        """Shutdown.

        Params:
        -------------------------
        force: If to shutdown executor without waiting for tasks.
        """
        task_dag = self.task_dag
        if len(task_dag.nodes) == 0:
            self.executor.shutdown()
        else:
            task_idstr = ",".join(task_dag.nodes.keys())
            logger.warning(f"Task {task_idstr} hasn't been done yet.")
            if force:
                self.executor.shutdown()
            else:
                tasks = [node.task for node in task_dag.nodes.values()]
                self.wait(*tasks)

        # Log the DAG of tasks.
        for ctx_name, dag in self.ctx_dags.items():
            logger.info("\n" + ctx_name + dag.visualize())

    def add(
        self,
        *tasks: Task,
        ctx_name: str = "",
    ):
        """Add tasks."""
        task_dag = self.task_dag
        log_dag = self.ctx_dags.setdefault(ctx_name, DirectedGraph())
        task_results = self.task_results
        with self._task_lock:
            # Add tasks to DAG first.
            for task in tasks:
                if not self.is_added(task):
                    raise ValueError(
                        f"Task {task.name} has been submited before."
                    )
                tid = getattr(task, "id_", task)
                task_dag.add_node(tid)
                log_dag.add_node(tid)

                # Bind task future to the node in the DAG.
                task_node = task_dag.get_node(tid)
                task_node.task = task
                task_node.done = threading.Event()

            # Then add the edges to the DAG, so to avoid the edges are added
            # before the nodes. As the tasks in `tasks` may not be
            # topological sorted.
            for task in tasks:
                tid = getattr(task, "id_", task)
                deps = task.resolve_dependencies()
                for dep in deps:
                    did = dep.id_
                    log_dag.add_edge(did, tid)
                    if did in self.task_dag:
                        task_dag.add_edge(did, tid)
                    elif not task_results.exists(did):
                        raise ValueError(
                            f"Task {dep.name} has not been submited before."
                        )

        self.submit_ready()

    def submit_ready(self):
        """Submit ready tasks.

        Ready tasks are the leaf nodes in the DAG as they don't need to
        wait for any other tasks.
        """
        # Check if the tasks forms a cycle, that no any other tasks could
        # be submited to executor.
        task_dag = self.task_dag
        if len(task_dag.leaf_nodes) == 0 and task_dag.node_count > 0:
            cycle = task_dag.find_cycle()
            task_names = ",".join(cycle)
            raise RuntimeError(f"Cycle {task_names} found in tasks.")

        # Lock has to be locked outside of the `for` loop, as other threads
        # of the executor will try to remove nodes within `mark_done`
        # callback, which may lead to the `RuntimeError` because the
        # `leaf_nodes` changed during the iteration.
        with self._task_lock:
            # Submit ready tasks, namely tasks with no upstream, to executor.
            for tid in task_dag.leaf_nodes:
                if tid in self.task_submited:
                    continue
                task_node = task_dag.get_node(tid)
                task = task_node.task
                on_done = functools.partial(self.mark_done, tid)
                self.executor.submit(self.task_results, (task, on_done))
                # future, = self.executor.submit(self.task_results, task)
                # future.add_done_callback(on_done)
                self.task_submited.add(tid)
                logger.info(f"Task {task.name} has been submited.")

    def mark_done(
        self,
        tid: str,
        future: Future | None = None,
    ):
        """Mark task done.

        1. The should be used as the callback of the `future` in most cases
          so that the task will mark itself done when it finished.
        2. But it may also be called to mark some task done manually and the
          `future` could be ignored.

        Params:
        -------------------------
        task: Task to be marked as done.
        future: Future of the task to fetch the result of the task.
        """
        # Try to fetch the result from the Future to raise the implicit
        # exceptions.
        if future is not None:
            try:
                _ret = future.result()
            except Exception as e:
                raise RuntimeError(
                    f"Fail to gather the result of {tid}."
                ) from e

        task_dag = self.task_dag
        assert tid in task_dag.leaf_nodes, (
            f"Tasks depended by {tid} should be done."
        )

        # Remove the task from the dag and then submit the ready tasks.
        task_node = task_dag.get_node(tid)
        with self._task_lock:
            task_node.done.set()
            task_dag.remove_node(tid)
        self.submit_ready()

    def wait(
        self,
        *tasks: Task | str,
    ):
        """Wait for tasks to finish."""
        task_dag = self.task_dag
        task_results = self.task_results
        for task in tasks:
            tid = getattr(task, "id_", task)
            task_name = getattr(task, "name", task)
            if task_results.exists(tid):
                continue
            task_node = task_dag.get_node(tid)
            if task_node is not None:
                logger.info(f"Waiting for task {task_name} to be done.")
                task_node.done.wait()
                continue
            raise RuntimeError(
                f"Task {task_name} should be submitted before being waited."
            )

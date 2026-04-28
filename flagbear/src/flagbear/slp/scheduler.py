#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: scheduler.py
#   Author: xyy15926
#   Created: 2026-04-23 17:09:42
#   Updated: 2026-04-27 22:19:16
#   Description:
# ---------------------------------------------------------

# %%
import logging
from typing import Dict, List, Callable, Any, Optional
import functools
import inspect
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent import futures
import dataclasses
from enum import Enum
import uuid
# from IPython.core.debugger import set_trace

if __name__ == "__main__":
    from importlib import reload
    from flagbear.tree import dag
    from flagbear.slp import finer, storage, serializer
    reload(dag)
    reload(finer)
    reload(storage)
    reload(serializer)
from flagbear.tree.dag import Node, DirectedGraph, NID

logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"),
)
logger = logging.getLogger()
logger.info("Logging Start.")


# %%
#TODO
class TaskState(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


@dataclasses.dataclass
class TaskResult:
    state: TaskState = dataclasses.field(default_factory = lambda : TaskState.PENDING)
    value: Any = None
    error: Optional[Exception] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    @property
    def duration(self) -> float:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0


# %%
class Task(Node):
    """Task acting as the a node in a working flow.

    Task is derived from Node in DirectedGraph and that:
    1. The task's downstream depends on current task.
    2. The task's upstream is depended by current task.

    Attrs:
    -------------------------------
    func: Callable to execute to reach a goal.
    retries: The maximum times to retry.
    result: TaskResult wrapping the `func` return with some metadata.
    """
    def __init__(
        self,
        func: Callable,
        name: Optional[str] = None,
        retries: int = 0,
        retry_delay: float = 1.0
    ):
        """Init task.

        Attrs:
        -------------------------------
        func: Callable to execute to reach a goal.
        name: Name of the task that should be unique among a Flow, namely
          the `Node.id_`.
        retries: The maximum times to retry.
        retry_delay: The unit of interval between retries.
        """
        super().__init__(name or func.__qualname__)
        self.func = func
        self.retries = retries
        self.retry_delay = retry_delay
        self.result: Optional[TaskResult] = None
        self._func_signature = inspect.signature(func)
        functools.update_wrapper(self, func)

    # Alias `name` as `id_` for `Task` derived from `Node`.
    @property
    def name(self):
        """`Node.id_` alias."""
        return self.id_

    @name.setter
    def name(self, name: NID):
        """`Node.id_` alias."""
        self.id_ = name
        return self.id_

    def __call__(self, *args, **kwargs):
        """Call `self.func`."""
        return self.func(*args, **kwargs)

    def execute(
        self,
        context: Dict[str, Any],
        upstream_results: Dict[str, Any],
    ) -> Any:
        """Execute task with parameters set from context and upstream result.

        Parameters of the task, or the wrapped function, will be set with
        results of upstream tasks or context by the their names. So the
        parameters' name should be set carefully.

        Params:
        ---------------------------
        context: Execution context.
        upstream_result: Dict of results of upstream tasks.
        """
        bound_args = {}

        # Set arguments automatically from context and upstream-results.
        for param_name, param in self._func_signature.parameters.items():
            if param_name == "upstream_results":
                bound_args["upstream_results"] = upstream_results
            elif param_name == "context":
                bound_args["context"] = context
            # Try to set parameters with upstream-results, context and the
            # default value in order.
            elif param_name in upstream_results:
                bound_args[param_name] = upstream_results[param_name]
            elif param_name in context:
                bound_args[param_name] = context[param_name]
            elif param.default != inspect.Parameter.empty:
                bound_args[param_name] = param.default

        # Retry.
        last_error = None
        for attempt in range(self.retries + 1):
            try:
                return self.func(**bound_args)
            except Exception as e:
                last_error = e
                if attempt < self.retries:
                    # Exponential backoff.
                    time.sleep(self.retry_delay * (2 ** attempt))

        raise last_error

    def __repr__(self):
        return f"Task({self.name})"


# %%
class Flow(DirectedGraph):
    """Working flow constructed by Tasks.

    Attrs:
    -----------------------------
    name: Flow's name.
    max_workers: Maximum number of workers to execute tasks at the same time.
    """
    def __init__(
        self,
        name: Optional[str] = None,
        max_workers: int = 4,
    ):
        super().__init__()
        self.name = name or f"flow_{uuid.uuid4().hex[:8]}"
        self.max_workers = max_workers
        self._executor = None

    def __repr__(self):
        return f"Task flow {self.name} with {len(self.tasks)} tasks"

    # Alias `tasks` as `_nodes` for `Flow` derived from `DirectedGraph`.
    @property
    def tasks(self):
        """`DirectedGraph._nodes` alias."""
        return self._nodes

    def task(
        self,
        func=None,
        *,
        name: Optional[str] = None,
        retries: int = 0,
        retry_delay: float = 1.0,
    ) -> Task:
        """Decorator to wrap and regist a function as a task.

        Attention:
        Since `func.__qualname__` is not easy to determined and will be used
        as the parameter by the task that depends on the task, it's maybe
        better to set name explicitly.

        Params:
        ------------------------------
        func: Callable to execute to reach a goal.
        name: Name of the task that should be unique among a Flow, namely
          the `Node.id_`.
        retries: The maximum times to retry.
        retry_delay: The unit of interval between retries.
        """
        def decorator(ffunc):
            nonlocal name, retries, retry_delay
            task = Task(
                ffunc,
                name = name or ffunc.__qualname__,
                retries=retries,
                retry_delay=retry_delay,
            )
            if task.name in self.tasks:
                raise ValueError(f"Replicated task name: {task.name}.")
            self._nodes[task.name] = task
            return task

        # `@task`：decorate the `func` directly with the default setting
        # `@task()`: return the inner decorator with customed setting
        if func is None:
            return decorator
        return decorator(func)

    @property
    def executor(self):
        """Init ThreadPoolExecutor lazily."""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(self.max_workers)
        return self._executor

    @staticmethod
    def _execute_task(
        task: Task,
        context: Dict[str, Any],
        upstream_results: Dict[str, Any]
    ):
        """Execute single task.

        Params:
        ----------------------------
        task: Single Task.
        upstream_results: Upstream tasks' results.
        """
        task.result = TaskResult(
            state = TaskState.RUNNING,
            start_time = time.time(),
        )
        try:
            value = task.execute(context, upstream_results)
            task.result.__dict__.update(dict(
                state = TaskState.SUCCESS,
                value = value,
                end_time = time.time(),
            ))
            logger.info(f"Task {task.name} succeeded.")
        except Exception as e:
            task.result.__dict__.update(dict(
                state = TaskState.FAILED,
                error = e,
                end_time = time.time()
            ))
            logger.warning(f"Task {task.name} failed: {e}.")

    def _execute_sequential(
        self,
        tasks: List[Task],
        upstream_results: Dict[str, Any] = None,
        fail_fast: bool = True,
    ) -> Dict[str, TaskResult]:
        """Execute task sequentially.

        Params:
        ----------------------------
        tasks: List of tasks to be executed.
        upstream_results: Dict of the results of upstream tasks, namely the
          results current `tasks` depends on, that will be passed as arguments
          to the tasks.
        fail_fast: If to stop executing when any task fails.

        Return:
        ----------------------------
        Dict[task.name, task.result] of tasks passed in.
        """
        task_results = {}
        for task in tasks:
            self._execute_task(
                task,
                self.context,
                upstream_results
            )
            if task.result.state == TaskState.FAILED and fail_fast:
                raise  RuntimeError(
                    f"Task {task.name} failed: {task.result.error}"
                ) from task.result.error
            task_results[task.name] = task.result
        return task_results

    def _execute_parallel(
        self,
        tasks: List[Task],
        upstream_results: Dict[str, Any] = None,
        fail_fast: bool = True,
    ) -> Dict[str, TaskResult]:
        """Execute task parallelly with inner executor.

        Params:
        ----------------------------
        tasks: List of tasks to be executed.
        upstream_results: Dict of the results of upstream tasks, namely the
          results current `tasks` depends on, that will be passed as arguments
          to the tasks.
        fail_fast: If to stop executing when any task fails.

        Return:
        ----------------------------
        Dict[task.name, task.result] of tasks passed in.
        """
        # Submit tasks and collect futures.
        task_futures = {}
        for task in tasks:
            future = self.executor.submit(
                self._execute_task,
                task,
                self.context,
                upstream_results,
            )
            task_futures[future] = task

        # Wait for the futures.
        task_results = {}
        for future in futures.as_completed(task_futures):
            task = task_futures[future]
            # `future.result()` may raise errors though
            # `try...catch...` is in `_execute_task`.
            try:
                future.result()
                task_results[task.name] = task.result
            except Exception as e:
                task.result.__dict__.update(dict(
                    state = TaskState.FAILED,
                    error = e,
                    end_time = time.time(),
                ))
                task_results[task.name] = task.result

            # Raise error immediataly if failing fast.
            if fail_fast and task.result.state == TaskState.FAILED:
                e = task.result.error
                raise RuntimeError(f"Task {task.name} failed: {e}") from e

        return task_results

    def process(
        self,
        context: Optional[Dict] = None,
        fail_fast: bool = True,
        parallel: bool = True,
    ) -> Dict[str, TaskResult]:
        """Execute tasks level by level sequentially.

        Params:
        ----------------------------
        context: Context providing additional arguments for tasks.
        fail_fast: If to stop executing when any task fails.
        parallel: If to execute tasks parallel.

        Return:
        ----------------------------
        Dict[task.name, TaskResult] of all tasks.
        """
        self.context = context or {}
        levels = self.topological_sort()
        if levels is None:
            logger.error("Task flow is not a DAG.")
            raise RuntimeError("Task flow is not a DAG.")

        logger.info(f"Start Flow {self.name}: {len(self.tasks)} tasks "
                    f"within {len(levels)} levels.")

        # Exceute tasks level by level sequentially.
        task_results: Dict[NID, TaskResult] = {}
        upstream_results: Dict[NID, Any] = {}
        for level in levels:
            tasks = [self.tasks[name] for name in level]
            level_results = self._execute_parallel(
                tasks,
                upstream_results,
                fail_fast,
            ) if parallel else self._execute_sequential(
                tasks,
                upstream_results,
                fail_fast,
            )
            task_results.update(level_results)
            upstream_results = {nid: ret.value
                                for nid, ret in task_results.items()}

        return task_results

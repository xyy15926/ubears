#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: protocols.py
#   Author: xyy15926
#   Created: 2026-05-06 15:36:50
#   Updated: 2026-05-21 11:21:31
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations

import contextvars
import copy
import functools
import inspect
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, Self

if TYPE_CHECKING:
    from collections.abc import Callable

    from flagbear.slp.cache import Cache
    from flagbear.slp.checkpoint import CheckpointPolicy

from flagbear.slp.cache import CachePolicy
from flagbear.slp.ser_exception import (
    ExceptionRecord,
    exception_to_records,
    restore_exception,
)
from flagbear.slp.serializer import (
    checker,
    deserialize,
    deserializer,
    serialize,
    serializer,
)

logger = logging.getLogger(__name__)


# %%
class TaskState(str, Enum):
    """Possible states of a task."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    RETRYING = "RETRYING"
    CANCELLED = "CANCELLED"
    CACHED = "CACHED"


@dataclass
class TaskResult:
    """Result of a task execution."""

    state: TaskState = TaskState.PENDING
    cache_key: str | None = None
    value: Any | None = None
    error: Exception | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    attempt: int = 1
    _error_recs: ExceptionRecord | None = field(default=None, repr=False)

    @property
    def duration(self) -> float:
        """Calculate duration between start and end times."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0

    @property
    def error_recs(self) -> list[dict[str, str]]:
        """ExceptionRecords."""
        if self._error_recs is not None:
            return self._error_recs
        if self.error is not None:
            self._error_recs = exception_to_records(self.error)
            return self._error_recs
        return None

    def is_successful(self) -> bool:
        """Check if the task completed successfully."""
        return self.state in (TaskState.SUCCESS, TaskState.CACHED)

    def is_failed(self) -> bool:
        """Check if the task failed."""
        return self.state in (
            TaskState.FAILED,
            TaskState.SKIPPED,
            TaskState.CANCELLED,
        )

    def to_json(self) -> bytes:
        """Serialize TaskResult into json bytes."""
        error_recs = None
        if self.error_recs is not None:
            error_recs = [asdict(rec) for rec in self.error_recs]
        metadict = {
            "state": self.state,
            "cache_key": self.cache_key,
            "error_recs": error_recs,
            "start_time": (
                self.start_time.isoformat()
                if self.start_time is not None
                else None
            ),
            "end_time": (
                self.end_time.isoformat()
                if self.end_time is not None
                else None
            ),
            "attempt": self.attempt,
        }
        return json.dumps(metadict, ensure_ascii=False).encode("utf8")

    @classmethod
    def from_json(cls, bytes_: bytes) -> Self:
        """Deserialize json bytes into TaskResult."""
        metadict = json.loads(bytes_.decode("utf8"))
        error_recs = None
        error = None
        if metadict["error_recs"] is not None:
            error_recs = [
                ExceptionRecord(**rec) for rec in metadict["error_recs"]
            ]
            error = restore_exception(metadict["error_recs"])
        meta = cls(
            state=TaskState(metadict["state"]),
            cache_key=metadict["cache_key"],
            error=error,
            start_time=(
                datetime.fromisoformat(metadict["start_time"])
                if metadict["start_time"] is not None
                else None
            ),
            end_time=(
                datetime.fromisoformat(metadict["end_time"])
                if metadict["end_time"] is not None
                else None
            ),
            attempt=metadict["attempt"],
            _error_recs=error_recs,
        )
        return meta

    def cache_policy(
        self,
        value_cache_policy: CachePolicy | None = None,
    ) -> CachePolicy:
        """Construct cache policy with this `TaskResult`.

        Construct cache policy with this `TaskResult` and additional
        CachePolicy for `TaskResult.value`.
        """
        if value_cache_policy is None:
            return CachePolicy(type_="TaskResult")
        if self.value is None:
            cache_policy = CachePolicy(
                None,
                "TaskResult:json",
                value_cache_policy.inline,
            )
        cache_policy = CachePolicy(
            None,
            f"TaskResult:{value_cache_policy.type_}",
            value_cache_policy.inline,
        )
        return cache_policy


# %%
@checker("TaskResult", priority=90)
def is_task_result(obj: Any):
    """If could serialize and deserialize with this."""
    return isinstance(obj, TaskResult)


@serializer("TaskResult")
def task_result_serialize(
    data: TaskResult,
    addon: str | list[str] | None = None,
) -> bytes:
    """Serialize TaskResult into bytes."""
    meta_bytes = data.to_json()
    meta_size = len(meta_bytes).to_bytes(4, "big")
    bytes_, val_type = serialize(data.value, addon)
    type_bytes = val_type.encode("utf8")
    type_size = len(type_bytes).to_bytes(4, "big")
    bytes_ = meta_size + meta_bytes + type_size + type_bytes + bytes_
    return bytes_


@deserializer("TaskResult")
def task_result_deserialize(
    bytes_: bytes,
    addon: str | list[str] | None = None,
) -> TaskResult:
    """Deserialize bytes into TaskResult."""
    meta_size = int.from_bytes(bytes_[:4], "big")
    data = TaskResult.from_json(bytes_[4 : 4 + meta_size])
    type_size = int.from_bytes(bytes_[4 + meta_size : 8 + meta_size])
    val_type = bytes_[8 + meta_size : 8 + meta_size + type_size].decode("utf8")
    if len(bytes_) > 8 + meta_size + type_size:
        value = deserialize(
            bytes_[8 + meta_size + type_size :],
            val_type,
        )
        data.value = value
    return data


# %%
@dataclass
class RetryPolicy:
    """Policy for retrying failed tasks."""

    max_retries: int = 1
    delay_seconds: float = 1.0
    backoff_factor: float = 2.0
    max_delay: float = 60.0
    retry_on: tuple[type[Exception]] | None = None

    def should_retry(
        self,
        error: Exception | None,
        attempt: int,
    ) -> bool:
        """Determine whether to retry based on error and attempt count."""
        if self.retry_on is not None and not isinstance(self.retry_on, tuple):
            self.retry_on = tuple(self.retry_on)
        if attempt >= self.max_retries:
            return False
        return (
            self.retry_on is None
            or error is None
            or isinstance(error, self.retry_on)
        )

    def get_delay(self, attempt: int) -> float:
        """Get the delay duration for a given attempt."""
        delay = self.delay_seconds * (self.backoff_factor**attempt)
        return min(delay, self.max_delay)


# %%
@dataclass
class ExecutionPolicy:
    """Policy for task execution settings."""

    timeout: float | None = None
    use_process: bool = False
    with_context: bool = False


# %%
class TaskProxyBase:
    """Proxy to convert a callable into Task.

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

    def __init__(
        self,
        func: Callable,
        name: str | None = None,
        cache_policy: CachePolicy | None = None,
        checkpoint_policy: CheckpointPolicy | None = None,
        retry_policy: RetryPolicy | None = None,
        execution_policy: ExecutionPolicy | None = None,
    ):
        """Init task.

        Params:
        -------------------------------
        func: Callable to execute to reach a goal.
        name: Name of the task that will be used to construct the `Task.id_`
          and `Task.cache_key`.
        cache_policy: Cache policy.
        checkpoint_policy: Checkpoint policy
        retry_policy: Retry policy.
        execution_policy: Execution policy.
        """
        self.func = func
        self._default_policies = {
            "name": name or f"{func.__module__}.{func.__qualname__}",
            "cache_policy": cache_policy,
            "checkpoint_policy": checkpoint_policy,
            "retry_policy": retry_policy,
            "execution_policy": execution_policy,
        }
        self._current_policies = copy.deepcopy(self._default_policies)
        self.is_async = inspect.iscoroutinefunction(func)
        functools.update_wrapper(self, func)

    def __getattr__(self, name: str) -> Any:
        """Get policy attributes from current policies."""
        if name in [
            "name",
            "cache_policy",
            "checkpoint_policy",
            "retry_policy",
            "execution_policy",
        ]:
            return self._current_policies[name]
        # Get attributes start from `super()`.
        return getattr(super(), name)

    def __setattr__(self, name: str, value: Any):
        """Set policy attributes in current policies."""
        if name in [
            "name",
            "cache_policy",
            "checkpoint_policy",
            "retry_policy",
            "execution_policy",
        ]:
            self._current_policies[name] = value
        else:
            # Use `super().__setattr__` to set attributes on self.
            super().__setattr__(name, value)

    def with_policy(
        self,
        *,
        name: str | None = None,
        cache_policy: CachePolicy | None = None,
        checkpoint_policy: CheckpointPolicy | None = None,
        retry_policy: RetryPolicy | None = None,
        execution_policy: ExecutionPolicy | None = None,
    ):
        """Set the temporary policies for constructing a Task.

        Temporary name should be reset once a Task has been constructed.
        """
        if name is not None:
            self.name = name
        if cache_policy is not None:
            self.cache_policy = cache_policy
        if checkpoint_policy is not None:
            self.checkpoint_policy = checkpoint_policy
        if retry_policy is not None:
            self.retry_policy = retry_policy
        if execution_policy is not None:
            self.execution_policy = execution_policy
        return self

    def reset_policy(self):
        """Reset policies for constructing a Task."""
        self._current_policies = copy.deepcopy(self._default_policies)

    def __call__(self, *args, **kwargs) -> Any:
        """Execute the wrapped callable."""
        pass

    def run(self, *args, **kwargs) -> Any:
        """Run the task synchronously."""
        pass

    def submit(self, *args, **kwargs) -> Task:
        """Submit the task for asynchronous execution."""
        pass


# %%
class Task(Protocol):
    """Protocol defining a schedulable task."""

    id_: str
    name: str

    def resolve_args(
        self,
        ctx: Context | None,
    ) -> tuple[list[Self], dict[str, Self], list[Self], dict[Self, Exception]]:
        """Resolve task arguments from context."""
        ...

    def resolve_dependencies(self, ctx: Context | None) -> list[Self]:
        """Resolve task dependencies from context."""
        ...

    def result(self, ctx: Context | None) -> Any:
        """Get the result from context."""
        ...


# %%
_current_context: contextvars.ContextVar[Context | None] = (
    contextvars.ContextVar(
        "_context",
        default=None,
    )
)


class Future(Protocol):
    """Protocol for asynchronous task results."""

    def result(self) -> Any:
        """Get the result of the future."""
        ...

    def add_done_callback(self, callback: callable):
        """Register a callback for when the future completes."""
        ...


class Context(Protocol):
    """Protocol for task execution context."""

    scheduler: Scheduler
    task_results: Cache

    def __enter__(self) -> Self:
        """Enter the context."""
        ...

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context."""
        ...

    def shutdown(self):
        """Shutdown the context."""
        ...

    def get_result(self, task: str | Task) -> TaskResult | None:
        """Get the result for a given task."""
        ...

    def set_result(self, task: str | Task, result: TaskResult):
        """Set the result for a given task."""
        ...

    def submit(self, task: Task | list[Task]):
        """Submit a task or list of tasks."""
        ...

    def run(self, task: Task | list[Task]) -> Any | list[Any]:
        """Run a task or list of tasks synchronously."""
        ...


class Executor(Protocol):
    """Protocol for task execution."""

    def submit(
        self, task_results: Cache, *task: Task
    ) -> Future | list[Future]:
        """Submit tasks for execution."""
        ...

    def shutdown(self):
        """Shutdown the executor."""
        ...


class Scheduler(Protocol):
    """Protocol for task scheduling."""

    def add(self, *tasks: Task) -> list[TaskResult]:
        """Add tasks to the scheduler."""
        ...

    def wait(self, *tasks: Task) -> Future | list[Future]:
        """Wait for tasks to complete."""
        ...

    def shutdown(self):
        """Shutdown the scheduler."""
        ...

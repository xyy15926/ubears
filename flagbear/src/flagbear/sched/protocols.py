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
import logging
from typing import Any, Optional, Type, Protocol, Self, TYPE_CHECKING
if TYPE_CHECKING:
    from flagbear.slp.cache import Cache
from collections.abc import Callable
from dataclasses import dataclass, asdict, field
from enum import Enum
import contextvars
import json
from datetime import datetime
import inspect
import functools
import copy

from flagbear.slp.ser_exception import(
    exception_to_records,
    restore_exception,
    ExceptionRecord,
)
from flagbear.slp.serializer import(
    checker, serializer, deserializer,
    serialize, deserialize,
)
from flagbear.slp.cache import CachePolicy
from flagbear.slp.checkpoint import CheckpointPolicy

logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"),
)
logger = logging.getLogger()
logger.info("Logging Start.")


# %%
class TaskState(str, Enum):
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
    state: TaskState = TaskState.PENDING
    cache_key: Optional[str] = None
    value: Optional[Any] = None
    error: Optional[Exception] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    attempt: int = 1
    _error_recs: Optional[ExceptionRecord] = field(
        default = None, repr = False
    )

    @property
    def duration(self) -> float:
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
        return self.state in (TaskState.SUCCESS, TaskState.CACHED)

    def is_failed(self) -> bool:
        return self.state in (TaskState.FAILED, TaskState.SKIPPED, TaskState.CANCELLED)

    def to_json(self) -> bytes:
        """Serialize TaskResult into json bytes."""
        error_recs = None
        if self.error_recs is not None:
            error_recs = [asdict(rec) for rec in self.error_recs]
        metadict = {
            "state": self.state,
            "cache_key": self.cache_key,
            "error_recs": error_recs,
            "start_time": (self.start_time.isoformat()
                           if self.start_time is not None else None),
            "end_time": (self.end_time.isoformat()
                         if self.end_time is not None else None),
            "attempt": self.attempt,
        }
        return json.dumps(metadict, ensure_ascii = False).encode("utf8")

    @classmethod
    def from_json(cls, bytes_: bytes) -> Self:
        """Deserialize json bytes into TaskResult."""
        metadict = json.loads(bytes_.decode("utf8"))
        error_recs = None
        error = None
        if metadict["error_recs"] is not None:
            error_recs = [ExceptionRecord(**rec) for rec in metadict["error_recs"]]
            error = restore_exception(metadict["error_recs"])
        meta = cls(
            state = TaskState(metadict["state"]),
            cache_key = metadict["cache_key"],
            error = error,
            start_time = (datetime.fromisoformat(metadict["start_time"])
                          if metadict["start_time"] is not None
                          else None),
            end_time = (datetime.fromisoformat(metadict["end_time"])
                          if metadict["end_time"] is not None
                          else None),
            attempt = metadict["attempt"],
            _error_recs = error_recs,
        )
        return meta

    def cache_policy(
        self,
        value_cache_policy: Optional[CachePolicy] = None,
    ) -> CachePolicy:
        """Construct cache policy for `TaskResult` from cache policy for
        `TaskResult.value`."""
        if value_cache_policy is None:
            return CachePolicy(type_ = "TaskResult")
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
@checker("TaskResult", priority = 90)
def is_task_result(obj: Any):
    """If could serialize and deserialize with this."""
    if isinstance(obj, TaskResult):
        return True
    return False


@serializer("TaskResult")
def task_result_serialize(
    data: TaskResult,
    addon: Optional[str | list[str]] = None,
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
    addon: Optional[str | list[str]] = None,
) -> TaskResult:
    """Deserialize bytes into TaskResult."""
    meta_size = int.from_bytes(bytes_[:4], "big")
    data = TaskResult.from_json(bytes_[4: 4 + meta_size])
    type_size = int.from_bytes(bytes_[4 + meta_size: 8 + meta_size])
    val_type = bytes_[8 + meta_size: 8 + meta_size + type_size].decode("utf8")
    if len(bytes_) > 8 + meta_size + type_size:
        value = deserialize(
            bytes_[8 + meta_size + type_size:],
            val_type,
        )
        data.value = value
    return data


# %%
@dataclass
class RetryPolicy:
    max_retries: int = 1
    delay_seconds: float = 1.0
    backoff_factor: float = 2.0
    max_delay: float = 60.0
    retry_on: Optional[tuple[Type[Exception]]] = None

    def should_retry(
        self,
        error: Optional[Exception],
        attempt: int,
    ) -> bool:
        if self.retry_on is not None and not isinstance(self.retry_on, tuple):
            self.retry_on = tuple(self.retry_on)
        if attempt >= self.max_retries:
            return False
        if (self.retry_on is not None
            and error is not None
            and not isinstance(error, self.retry_on)):
            return False
        return True

    def get_delay(self, attempt: int) -> float:
        delay = self.delay_seconds * (self.backoff_factor ** attempt)
        return min(delay, self.max_delay)


# %%
@dataclass
class ExecutionPolicy:
    timeout: Optional[float] = None
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
        name: Optional[str] = None,
        cache_policy: Optional[CachePolicy] = None,
        checkpoint_policy: Optional[CheckpointPolicy] = None,
        retry_policy: Optional[RetryPolicy] = None,
        execution_policy: Optional[ExecutionPolicy] = None,
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
        if name in ["name", "cache_policy", "checkpoint_policy",
                    "retry_policy", "execution_policy"]:
            return self._current_policies[name]
        # Get attributes start from `super()`.
        return getattr(super(), name)

    def __setattr__(self, name: str, value: Any):
        if name in ["name", "cache_policy", "checkpoint_policy",
                    "retry_policy", "execution_policy"]:
            self._current_policies[name] = value
        else:
            # Use `super().__setattr__` to set attributes on self.
            super().__setattr__(name, value)

    def with_policy(
        self,
        *,
        name: Optional[str] = None,
        cache_policy: Optional[CachePolicy] = None,
        checkpoint_policy: Optional[CheckpointPolicy] = None,
        retry_policy: Optional[RetryPolicy] = None,
        execution_policy: Optional[ExecutionPolicy] = None,
    ):
        """Set the temperary policies for constructing a Task.

        Temperary name should be reset once a Task has been constructed.
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
        pass

    def run(self, *args, **kwargs) -> Any:
        pass

    def submit(self, *args, **kwargs) -> Task:
        pass


# %%
class Task(Protocol):
    id_: str
    name: str
    def resolve_args(
        self,
        ctx: Optional[Context],
    ) -> tuple[list[Self], dict[str, Self], list[Self], dict[Self, Exception]]: ...
    def resolve_dependencies(self, ctx: Optional[Context]) -> list[Self]: ...
    def result(self, ctx: Optional[Context]) -> Any: ...


# %%
_current_context: contextvars.ContextVar[Optional[Context]] = contextvars.ContextVar(
    "_context",
    default = None,
)


class Future(Protocol):
    def result() -> Any: ...
    def add_done_callback(callback: callable): ...


class Context(Protocol):
    scheduler: Scheduler
    task_results: Cache
    def __enter__(self) -> Self:...
    def __exit__(self, exc_type, exc_val, exc_tb):...
    def shutdown(self): ...
    def get_result(self, task: str | Task) -> Optional[TaskResult]: ...
    def set_result(self, task: str | Task, result: TaskResult): ...
    def submit(self, task: Task | list[Task]): ...
    def run(self, task: Task | list[Task]) -> Any | list[Any]: ...


class Executor(Protocol):
    def submit(self, task_results: Cache, *task: Task) -> Future | list[Future]: ...
    def shutdown(self):...


class Scheduler(Protocol):
    def add(self, *tasks: Task) -> list[TaskResult]: ...
    def wait(self, *tasks: Task) -> Future | list[Future]: ...
    def shutdown(self):...

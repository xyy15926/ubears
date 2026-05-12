#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: protocols.py
#   Author: xyy15926
#   Created: 2026-05-06 15:36:50
#   Updated: 2026-05-07 20:06:16
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
import logging
from typing import Any, Optional, Tuple, Type, Protocol, List, Self, Dict, TypeVar
from dataclasses import dataclass
from enum import Enum
import contextvars

logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"),
)
logger = logging.getLogger()
logger.info("Logging Start.")


# %%
class TaskState(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    RETRYING = "RETRYING"


@dataclass
class TaskResult:
    state: TaskState = TaskState.PENDING
    value: Any = None
    error: Optional[Exception] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    attempt: int = 1

    @property
    def duration(self) -> float:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0

    def is_successful(self) -> bool:
        return self.state in (TaskState.SUCCESS, )

    def is_failed(self) -> bool:
        return self.state in (TaskState.FAILED, )


# %%
@dataclass
class RetryPolicy:
    max_retries: int = 1
    delay_seconds: float = 1.0
    backoff_factor: float = 2.0
    max_delay: float = 60.0
    retry_on: Optional[Tuple[Type[Exception]]] = None

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
class TaskFuture(Protocol):
    id_: str
    name: str
    def resolve_args(
        self,
        ctx: Optional[Context],
    ) -> Tuple[List[Self], Dict[str, Self], List[Self], Dict[Self, Exception]]:
        ...
    def resolve_dependencies(
        self,
        ctx: Optional[Context],
    ) -> List[Self]:
        ...
    def result(
        self,
        ctx: Optional[Context],
    ) -> Any:
        ...


# %%
class Context(Protocol):
    engine: Executor
    def __enter__(self) -> Self:...
    def __exit__(self, exc_type, exc_val, exc_tb):...
    def shutdown(self): ...
    def get_result(
        self,
        task_future: str | TaskFuture,
    ) -> Optional[TaskResult]: ...
    def set_result(
        self,
        task_future: str | TaskFuture,
        result: TaskResult,
    ): ...
    def process(
        self,
        task_future: TaskFuture | List[TaskFuture],
    ) -> Any | List[Any]: ...
    def submit(
        self,
        task_future: TaskFuture | List[TaskFuture],
    ): ...
    def wait_for(
        self,
        task_future: TaskFuture | List[TaskFuture],
    ) -> Any | List[Any]: ...
    def get_artifact(self, key: str) -> Optional[Any]: ...
    def set_artifact(self, key: str, value: Any): ...


_current_context: contextvars.ContextVar[Optional[Context]] = contextvars.ContextVar(
    "_context",
    default = None,
)


# %%
Future = TypeVar("Future")


class Executor(Protocol):
    def run(
        self,
        ctx: Context,
        *task_future: TaskFuture,
    ) -> List[TaskResult]: ...
    def submit(
        self,
        ctx: Context,
        *task_future: TaskFuture,
    ) -> Future | List[Future]: ...
    def gather(
        self,
        *future: Future,
    ) -> List[TaskResult]: ...
    def shutdown(self):...

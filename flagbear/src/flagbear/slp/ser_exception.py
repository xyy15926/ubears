#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: ser_exception.py
#   Author: xyy15926
#   Created: 2026-05-13 16:07:20
#   Updated: 2026-05-13 21:54:36
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations

import json
import traceback
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ExceptionRecord:
    """Record of a single exception in the chain."""
    exc_type: str
    exc_module: str
    message: str
    traceback_lines: list[str]
    cause_type: str | None = None
    context_type: str | None = None


# %%
def exception_to_records(
    exception: BaseException,
) -> list[ExceptionRecord]:
    """Flatten exception into a list of exception chain.

    Exception will be recorded in a list like a stack that the first exception
    is the exact exception raised.
    """
    records = []
    # Record visited exception in case reference loop.
    visited = set()

    def _walk(e: BaseException):
        if id(e) in visited:
            return
        visited.add(id(e))

        records.append(ExceptionRecord(
            exc_type = type(e).__name__,
            exc_module = type(e).__module__,
            message = str(e),
            traceback_lines = traceback.format_exception(
                type(e), e, e.__traceback__
            ),
            cause_type = (
                type(e.__cause__).__name__
                if e.__cause__ else None
            ),
            context_type = (
                type(e.__context__).__name__
                if e.__context__
                and not e.__suppress_context__
                else None
            ),
        ))

        if e.__cause__:
            _walk(e.__cause__)
        elif e.__context__ and not e.__suppress_context__:
            _walk(e.__context__)

    _walk(exception)
    return records


# %%
class UnrecoverableException(Exception):
    """Fallback exception if exception recorded can't be found."""
    pass


def resolve_exception_class(
    module: str,
    name: str,
) -> type[BaseException]:
    """Resolve the exception with module name and exception name."""
    try:
        import importlib
        mod = importlib.import_module(module)
        cls = getattr(mod, name)
        if not issubclass(cls, BaseException):
            raise TypeError(f"{cls} is not an exception")  # noqa: TRY301
    except (ImportError, AttributeError, TypeError):
        # Create a new exception class derived from `UnrecoverableException`.
        return type(name, (UnrecoverableException,), {"__module__": module})
    else:
        return cls


# %%
def restore_exception(
    records: list[dict[str, Any]],
) -> BaseException:
    """Restore exception from the exception chain records list.

    Exception will be recorded in a list like a stack that the first exception
    is the exact exception raised.
    """
    if not records:
        raise ValueError("Empty records")

    exceptions = []
    for rec in records:
        if isinstance(rec, ExceptionRecord):
            rec = asdict(rec)
        cls = resolve_exception_class(rec["exc_module"], rec["exc_type"])
        exception = cls(rec["message"]) if rec["message"] else cls()
        exceptions.append(exception)

    for i in range(len(exceptions) - 1):
        exceptions[i].__cause__ = exceptions[i + 1]

    # Return the last exception, namely the exact exception raised.
    return exceptions[0]


# %%
def str_exception(
    exception: BaseException,
) -> str:
    """Serialize exception into bytes."""
    recs = exception_to_records(exception)
    recs = [asdict(rec) for rec in recs]
    str_ = json.dumps(recs, ensure_ascii=False)
    return str_


def destr_exception(
    str_: str,
) -> BaseException:
    """Deserialize exception from bytes.

    The `__traceback__` can't be restored as the `__traceback__` stores
    the real stack frame which can't be serialized.
    """
    recs = json.loads(str_)
    exception = restore_exception(recs)
    return exception

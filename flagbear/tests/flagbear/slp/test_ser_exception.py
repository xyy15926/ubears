#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_ser_exception.py
#   Author: xyy15926
#   Created: 2026-05-13 20:08:13
#   Updated: 2026-05-13 21:55:18
#   Description:
# ---------------------------------------------------------

# %%
import pytest

if __name__ == "__main__":
    from importlib import reload
    from flagbear.slp import ser_exception
    reload(ser_exception)

from dataclasses import asdict
from flagbear.slp.ser_exception import(
    ExceptionRecord,
    exception_to_records,
    restore_exception,
    str_exception,
    destr_exception,
)


# %%
def test_ExceptionRecord():
    def raise_chain():
        raise RuntimeError("Runtime error") from ValueError("Value error")

    try:
        raise_chain()
    except Exception as e:
        error = e

    recs = exception_to_records(error)
    assert len(recs) == 2
    assert isinstance(recs[0], ExceptionRecord)

    error_ref = error
    restored = restore_exception(recs)
    while restored:
        assert type(restored) is type(error)
        assert str(restored) == str(error)
        restored = restored.__cause__
        error = error.__cause__

    error = error_ref
    rec_dicts = [asdict(rec) for rec in recs]
    restored = restore_exception(rec_dicts)
    while restored:
        assert type(restored) is type(error)
        assert str(restored) == str(error)
        restored = restored.__cause__
        error = error.__cause__


# %%
def test_ExceptionRecord_serialize():
    def raise_chain():
        raise RuntimeError("Runtime error") from ValueError("Value error")

    try:
        raise_chain()
    except Exception as e:
        error = e

    str_ = str_exception(error)
    restored = destr_exception(str_)
    while restored:
        assert type(restored) is type(error)
        assert str(restored) == str(error)
        restored = restored.__cause__
        error = error.__cause__

#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_protocols.py
#   Author: xyy15926
#   Created: 2026-05-13 15:18:23
#   Updated: 2026-05-20 21:53:49
#   Description:
# ---------------------------------------------------------

# %%

if __name__ == "__main__":
    from importlib import reload

    from flagbear.sched import protocols
    from flagbear.slp import serializer

    reload(serializer)
    reload(protocols)

from datetime import datetime

import numpy as np

from flagbear.sched.protocols import (
    TaskResult,
    TaskState,
)
from flagbear.slp.serializer import deserialize, serialize


# %%
def test_TaskResult_to_json():
    def raise_chain():
        raise RuntimeError("Runtime error") from ValueError("Value error")

    try:
        raise_chain()
    except Exception as e:
        error = e

    ori = TaskResult(
        state=TaskState.RUNNING,
        value={
            "a": 1,
        },
        error=error,
        start_time=datetime.now(),
        end_time=datetime.now(),
        attempt=1,
    )

    # Unspecify `value` type.
    bytes_, _ = serialize(ori, "TaskResult")
    assert b"json" in bytes_
    loaded = deserialize(bytes_, "TaskResult")
    assert loaded.state == ori.state
    assert loaded.value == ori.value
    assert loaded.start_time == ori.start_time
    assert loaded.end_time == ori.end_time
    assert loaded.attempt == ori.attempt

    error = ori.error
    restored = loaded.error
    while restored:
        assert type(restored) is type(error)
        assert str(restored) == str(error)
        restored = restored.__cause__
        error = error.__cause__

    # Specify `value` type.
    bytes_, _ = serialize(ori, "TaskResult:json")
    assert b"json" in bytes_
    loaded = deserialize(bytes_, "TaskResult:json")
    assert loaded.state == ori.state
    assert loaded.value == ori.value
    assert loaded.start_time == ori.start_time
    assert loaded.end_time == ori.end_time
    assert loaded.attempt == ori.attempt

    error = ori.error
    restored = loaded.error
    while restored:
        assert type(restored) is type(error)
        assert str(restored) == str(error)
        restored = restored.__cause__
        error = error.__cause__


# %%
def test_TaskResult_to_json_with_serialize_value_with_numpy():
    def raise_chain():
        raise RuntimeError("Runtime error") from ValueError("Value error")

    try:
        raise_chain()
    except Exception as e:
        error = e

    value = np.random.rand(4, 4)
    ori = TaskResult(
        state=TaskState.RUNNING,
        value=value,
        error=error,
        start_time=datetime.now(),
        end_time=datetime.now(),
        attempt=1,
    )

    # Unspecify `value` type.
    bytes_, _ = serialize(ori, "TaskResult")
    assert b"numpy" in bytes_
    loaded = deserialize(bytes_, "TaskResult")
    assert loaded.state == ori.state
    assert np.allclose(loaded.value, ori.value)
    assert loaded.start_time == ori.start_time
    assert loaded.end_time == ori.end_time
    assert loaded.attempt == ori.attempt

    error = ori.error
    restored = loaded.error
    while restored:
        assert type(restored) is type(error)
        assert str(restored) == str(error)
        restored = restored.__cause__
        error = error.__cause__

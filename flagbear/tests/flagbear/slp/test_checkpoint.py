#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_checkpoint.py
#   Author: xyy15926
#   Created: 2026-04-22 15:09:53
#   Updated: 2026-04-23 16:59:32
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
import pytest
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, force=True)
    from importlib import reload
    from flagbear.slp import storage, serializer, checkpoint
    reload(storage)
    reload(serializer)
    reload(checkpoint)

import shutil
import dataclasses
from datetime import datetime, timedelta
import threading
import hashlib
import json
import numpy as np
import pandas as pd

from flagbear.slp.finer import get_tmp_path
from flagbear.slp.storage import LocalFileStorage
from flagbear.slp.checkpoint import(
    concat_params,
    CheckpointManager,
)

PYTEST_DIR = "tmp/pytest_tmpdir"
TMP_DIR = get_tmp_path(PYTEST_DIR)


# %%
@pytest.fixture(scope="module", autouse=False)
def tmpfile_fixture(request):
    yield

    # Remove the tmp file during the pytest.
    # Clear only once with `scope=module`.
    pytest_tmp = TMP_DIR
    shutil.rmtree(pytest_tmp, ignore_errors=True)
    if not any(get_tmp_path().iterdir()):
        get_tmp_path().rmdir()


# %%
def test_concat_params():
    def foo(a, b, c, d = 3):
        pass

    # Json.dumps the arguments.
    args = (1, 2)
    kwargs = {"c": 2, "d": 1}
    foo(*args, **kwargs)
    key = concat_params(foo, args, kwargs, skip_args=[0, "d"])

    json_str = json.dumps(
        ((2, ), {"c": 2}),
        sort_keys=True,
        default=str,
    ).encode("utf8")
    dig = hashlib.md5(json_str).hexdigest()[:16]
    assert key.endswith(f"foo_v1_{dig}")

    # `skip_args` will skip some arguments.
    args2 = (3, 2)
    kwargs2 = {"c": 2}
    key2 = concat_params(foo, args2, kwargs2, skip_args=[0, "d"])
    assert key == key2

    # Stringfied part of the arguments and then json.dumps.
    arr = np.random.rand(3, 4)
    args = (1, 2, 3, arr)
    key = concat_params(foo, args, {})
    str_ = json.dumps(
        ((1, 2, 3, str(arr)), {}),
        sort_keys=True,
        default=str,
    ).encode("utf8")
    dig = hashlib.md5(str_).hexdigest()[:16]
    assert key.endswith(f"foo_v1_{dig}")


# %%
def test_CheckpointManager_cache_key(tmpfile_fixture):
    lstore = LocalFileStorage(TMP_DIR)
    ckpm = CheckpointManager(lstore)

    foo_once = False
    @ckpm.checkpoint
    def foo(a, b, c, d):
        # `nonlocal` only bind the varible on the exact upper level.
        # So this should be annotated if runned directly through terminal
        # at the `__main__` module.
        nonlocal foo_once
        if foo_once:
            raise RuntimeError("Should be called only once.")
        foo_once = True
        return [a, b, c, d]

    # `foo` will only be called once.
    foo_ret1 = foo(1, 2, 3, 4)
    assert foo_ret1 == [1, 2, 3, 4]

    key = ckpm.gen_key(foo, (1, 2, 3, 4), {}, override=True)
    assert key in lstore.list_keys()

    foo_ret2 = foo(1, 2, 3, 4)
    assert foo_ret2 == [1, 2, 3, 4]


# %%
def test_CheckpointManager_ttl_expire(tmpfile_fixture):
    lstore = LocalFileStorage(TMP_DIR)
    ckpm = CheckpointManager(lstore)

    @ckpm.checkpoint(ttl=timedelta(0), override=True)
    def foo(a, b, c, d):
        return [a, b, c, d]

    foo_ret1 = foo(1, 2, 3, 5)
    assert foo_ret1 == [1, 2, 3, 5]
    key1 = ckpm.gen_key(foo, (1, 2, 3, 5), {}, override = True)
    assert key1 in lstore.list_keys()

    # Cache will be deleted after expire.
    assert lstore.get(key1) is None
    assert key1 not in lstore.list_keys()


# %%
def test_CheckpointManager_meta_override(tmpfile_fixture):
    lstore = LocalFileStorage(TMP_DIR)
    ckpm = CheckpointManager(lstore)

    # Keep the history cache.
    @ckpm.checkpoint(ttl=timedelta(1), override=False, force=True)
    def foo(a, b, c, d):
        return [a, b, c, d]

    foo_ret1 = foo(1, 2, 3, 6)
    assert foo_ret1 == [1, 2, 3, 6]
    key1 = ckpm.gen_key(foo, (1, 2, 3, 6), {}, override = True)
    assert key1 in lstore.list_keys()

    # New cache will be generated.
    foo_ret2 = foo(1, 2, 3, 6)
    assert foo_ret2 == [1, 2, 3, 6]
    key2 = ckpm.gen_key(foo, (1, 2, 3, 6), {}, override = True)
    assert key1 != key2
    assert key1 in lstore.list_keys()
    assert key2 in lstore.list_keys()


# %%
def test_CheckpointManager_deserialize(tmpfile_fixture):
    lstore = LocalFileStorage(TMP_DIR)
    ckpm = CheckpointManager(lstore)

    # Keep the history cache.
    @ckpm.checkpoint()
    def foo(a, b, c, d):
        return [a, b, c, d]

    foo_ret1 = foo(1, 2, 3, 7)
    assert foo_ret1 == [1, 2, 3, 7]
    key1 = ckpm.gen_key(foo, (1, 2, 3, 7), {}, override = True)
    assert key1 in lstore.list_keys()

    # Get the data from history cache and deserialize mannaully.
    bytes_ = lstore.get(key1)
    assert ckpm.deserialize(bytes_) == [1, 2, 3, 7]


# %%
def test_CheckpointManager_serialize_manaully(tmpfile_fixture):
    lstore = LocalFileStorage(TMP_DIR)
    ckpm = CheckpointManager(lstore)

    # Pickle serializer.
    @ckpm.checkpoint(ser_type="pickle")
    def foo_pickle(a, b, c, d):
        return [a, b, c, d]

    foo_pickle_ret = foo_pickle(1, 2, 3, 8)
    assert foo_pickle_ret == [1, 2, 3, 8]
    key_pickle = ckpm.gen_key(foo_pickle, (1, 2, 3, 8), {}, override = True)
    bytes_pickle = lstore.get(key_pickle)
    assert b"pickle" in bytes_pickle
    assert b"json" not in bytes_pickle

    # Json serializer.
    @ckpm.checkpoint(ser_type="json")
    def foo_json(a, b, c, d):
        return [a, b, c, d]

    foo_json_ret = foo_json(1, 2, 3, 8)
    assert foo_json_ret == [1, 2, 3, 8]
    key_json = ckpm.gen_key(foo_json, (1, 2, 3, 8), {}, override = True)
    bytes_json = lstore.get(key_json)
    assert b"pickle" not in bytes_json
    assert b"json" in bytes_json


# %%
def test_CheckpointManager_complex_arugments(tmpfile_fixture):
    lstore = LocalFileStorage(TMP_DIR)
    ckpm = CheckpointManager(lstore)

    @ckpm.checkpoint()
    def foo(a, b, c, d):
        return [a, b, c, d]

    arr = np.random.rand(3, 4)
    foo_ret1 = foo(1, 2, 3, arr)
    assert foo_ret1[:3] == [1, 2, 3]
    assert np.allclose(foo_ret1[3], arr)

    foo_ret2 = foo(1, 2, 3, arr)
    assert foo_ret2[:3] == [1, 2, 3]
    assert np.allclose(foo_ret2[3], arr)

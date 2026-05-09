#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_checkpoint.py
#   Author: xyy15926
#   Created: 2026-04-22 15:09:53
#   Updated: 2026-05-03 22:08:54
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
import pytest
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, force=True)
    from importlib import reload
    from flagbear.slp import storage, serializer, cache, checkpoint
    reload(storage)
    reload(serializer)
    reload(cache)
    reload(checkpoint)

import shutil
import dataclasses
from datetime import datetime, timedelta
import threading
import hashlib
import json
import numpy as np

from flagbear.slp.finer import get_tmp_path
from flagbear.slp.storage import LocalFileStorage
from flagbear.slp.cache import MemoryCache, PersistentCache
from flagbear.slp.checkpoint import(
    json_args,
    CachePolicy,
    CheckpointPolicy,
    CheckpointManager,
)

PYTEST_DIR = "tmp/pytest_tmpdir"
TMP_DIR = get_tmp_path(PYTEST_DIR)


# %%
@pytest.fixture(scope="function", autouse=False)
def tmpfile_fixture(request):
    yield

    # Remove the tmp file during the pytest.
    # Clear only once with `scope=module`.
    pytest_tmp = TMP_DIR
    shutil.rmtree(pytest_tmp, ignore_errors=True)
    if not any(get_tmp_path().iterdir()):
        get_tmp_path().rmdir()


# %%
def test_json_args():
    # Json.dumps the arguments.
    args = (1, 2)
    kwargs = {"c": 2, "d": 1}
    key = json_args(args, kwargs, skip_args=[0, "d"])
    json_str = json.dumps(
        ((2, ), {"c": 2}),
        sort_keys=True,
        default=str,
    ).encode("utf8")
    dig = hashlib.md5(json_str).hexdigest()[:16]
    assert key == dig

    # `skip_args` will skip some arguments.
    args2 = (3, 2)
    kwargs2 = {"c": 2}
    key2 = json_args(args2, kwargs2, skip_args=[0, "d"])
    assert key == key2

    # Stringfied part of the arguments and then json.dumps.
    arr = np.random.rand(3, 4)
    args = (1, 2, 3, arr)
    key = json_args(args, {})
    str_ = json.dumps(
        ((1, 2, 3, str(arr)), {}),
        sort_keys=True,
        default=str,
    ).encode("utf8")
    dig = hashlib.md5(str_).hexdigest()[:16]
    assert key == dig


# %%
def test_CheckpointManager_memory_cache():
    cache = MemoryCache()
    ckpm = CheckpointManager(cache = cache)

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
    foo_ret2 = foo(1, 2, 3, 4)
    assert foo_ret2 == [1, 2, 3, 4]

    ckpm.clear_func_cache(foo)
    with pytest.raises(RuntimeError):
        _foo_ret3 = foo(1, 2, 3, 4)


# %%
def test_CheckpointManager_memory_cache_cache_mode():
    cache = MemoryCache()
    ckpm = CheckpointManager(cache = cache)
    checkpoint_policy = CheckpointPolicy("cache")
    cache_policy = CachePolicy(timedelta(1), None, None)

    @ckpm.checkpoint(
        checkpoint_policy = checkpoint_policy,
        cache_policy = cache_policy,
    )
    def foo(a, b, c, d):
        return [a, b, c, d]
    prefix = f"{foo.__module__}.{foo.__qualname__}"

    foo_ret1 = foo(1, 2, 3, 4)
    foo_ret2 = foo(1, 2, 3, 4)
    assert foo_ret1 == foo_ret2
    assert len(cache.list_keys(prefix)) == 1
    assert ckpm.get(foo, (1, 2, 3, 4), {}) == [1, 2, 3, 4]
    assert ckpm.set(foo, (1, 2, 3, 5), {}) == [1, 2, 3, 5]
    assert ckpm.get(foo, (1, 2, 3, 5), {}) == [1, 2, 3, 5]
    assert len(cache.list_keys(prefix)) == 2


# %%
def test_CheckpointManager_memory_cache_key_gen():
    cache = MemoryCache()
    ckpm = CheckpointManager(cache = cache)
    checkpoint_policy = CheckpointPolicy("cache", keyskip_args = [])
    cache_policy = CachePolicy(timedelta(1), None, None)

    @ckpm.checkpoint(
        checkpoint_policy = checkpoint_policy,
        cache_policy = cache_policy,
    )
    def foo(a, b, c, d):
        return [a, b, c, d]
    prefix = f"{foo.__module__}.{foo.__qualname__}"

    # Complex arguments.
    arr = np.random.rand(3, 4)
    foo_ret1 = foo(1, 2, 3, arr)
    assert foo_ret1[:3] == [1, 2, 3]
    assert np.allclose(foo_ret1[3], arr)

    foo_ret2 = foo(1, 2, 3, arr)
    assert foo_ret2[:3] == [1, 2, 3]
    assert np.allclose(foo_ret2[3], arr)
    assert len(cache.list_keys(prefix)) == 1

    arr2 = np.random.rand(3, 4)
    foo_ret2_fake = foo(1, 2, 3, arr2)
    assert foo_ret2_fake[:3] == [1, 2, 3]
    assert not np.allclose(foo_ret2_fake[3], arr)
    assert len(cache.list_keys(prefix)) == 2

    # Skip some arguments.
    cache = MemoryCache()
    ckpm = CheckpointManager(cache = cache)
    checkpoint_policy = CheckpointPolicy("cache", keyskip_args = [3, ])
    cache_policy = CachePolicy(timedelta(1), None, None)

    @ckpm.checkpoint(
        checkpoint_policy = checkpoint_policy,
        cache_policy = cache_policy,
    )
    def foo(a, b, c, d):
        return [a, b, c, d]
    prefix = f"{foo.__module__}.{foo.__qualname__}"

    arr = np.random.rand(3, 4)
    foo_ret1 = foo(1, 2, 3, arr)
    assert foo_ret1[:3] == [1, 2, 3]
    assert np.allclose(foo_ret1[3], arr)

    foo_ret2 = foo(1, 2, 3, arr)
    assert foo_ret2[:3] == [1, 2, 3]
    assert np.allclose(foo_ret2[3], arr)
    assert len(cache.list_keys(prefix)) == 1

    arr2 = np.random.rand(3, 4)
    # Hit caches indeed.
    foo_ret2_fake = foo(1, 2, 3, arr2)
    assert foo_ret2_fake[:3] == [1, 2, 3]
    assert np.allclose(foo_ret2_fake[3], arr)
    assert len(cache.list_keys(prefix)) == 1



# %%
def test_CheckpointManager_memory_cache_record_mode():
    cache = MemoryCache()
    ckpm = CheckpointManager(cache = cache)
    checkpoint_policy = CheckpointPolicy("record")
    cache_policy = CachePolicy(timedelta(1), None, None)

    @ckpm.checkpoint(
        checkpoint_policy = checkpoint_policy,
        cache_policy = cache_policy,
    )
    def foo(a, b, c, d):
        return [a, b, c, d]
    prefix = f"{foo.__module__}.{foo.__qualname__}"

    foo_ret1 = foo(1, 2, 3, 4)
    foo_ret2 = foo(1, 2, 3, 4)
    assert foo_ret1 == foo_ret2
    assert len(cache.list_keys(prefix)) == 2
    assert ckpm.get(foo, (1, 2, 3, 4), {}) != [1, 2, 3, 4]
    assert ckpm.get(foo, (1, 2, 3, 4), {}, checkpoint_policy) == [1, 2, 3, 4]
    assert ckpm.set(foo, (1, 2, 3, 5), {}, checkpoint_policy) == [1, 2, 3, 5]
    assert ckpm.get(foo, (1, 2, 3, 5), {}, checkpoint_policy) == [1, 2, 3, 5]
    # `foo` has been decorated with checkpoint in `record` mode.
    # And `foo` will be called in `.set`.
    assert len(cache.list_keys(prefix)) == 4


# %%
def test_CheckpointManager_memory_cache_manual_mode():
    cache = MemoryCache()
    ckpm = CheckpointManager(cache = cache)
    checkpoint_policy = CheckpointPolicy("manual")
    cache_policy = CachePolicy(timedelta(1), None, None)

    @ckpm.checkpoint(
        checkpoint_policy = checkpoint_policy,
        cache_policy = cache_policy,
    )
    def foo(a, b, c, d):
        return [a, b, c, d]
    prefix = f"{foo.__module__}.{foo.__qualname__}"

    foo_ret1 = foo(1, 2, 3, 4)
    foo_ret2 = foo(1, 2, 3, 4)
    assert foo_ret1 == foo_ret2
    assert len(cache.list_keys(prefix)) == 1
    foo_ret2 = foo(1, 2, 3, 4, force = True)
    assert len(cache.list_keys(prefix)) == 2
    assert ckpm.get(foo, (1, 2, 3, 4), {}) != [1, 2, 3, 4]
    assert ckpm.get(foo, (1, 2, 3, 4), {}, checkpoint_policy) == [1, 2, 3, 4]
    assert ckpm.set(foo, (1, 2, 3, 5), {}, checkpoint_policy) == [1, 2, 3, 5]
    assert ckpm.get(foo, (1, 2, 3, 5), {}, checkpoint_policy) == [1, 2, 3, 5]
    # `foo` has been decorated with checkpoint in `manual` mode.
    # And `foo` will be called in `.set`.
    assert len(cache.list_keys(prefix)) == 4


# %%
def test_CheckpointManager_memory_cache_ttl():
    # `ttl` won't take effects for checkpoint in `record` mode.
    # Checkpoint in `record` mode won't check the expires, so no cache
    # will be deleted though expired.
    cache = MemoryCache()
    ckpm = CheckpointManager(cache = cache)
    checkpoint_policy = CheckpointPolicy("record")
    cache_policy = CachePolicy(timedelta(0), None, None)

    @ckpm.checkpoint(
        checkpoint_policy = checkpoint_policy,
        cache_policy = cache_policy,
    )
    def foo(a, b, c, d):
        return [a, b, c, d]
    prefix = f"{foo.__module__}.{foo.__qualname__}"

    foo_ret1 = foo(1, 2, 3, 4)
    assert len(cache.list_keys(prefix)) == 1
    foo_ret2 = foo(1, 2, 3, 4)
    assert foo_ret1 == foo_ret2
    assert len(cache.list_keys(prefix)) == 2

    # And cache will be deleted for checkpoint in `manual` mode.
    cache = MemoryCache()
    ckpm = CheckpointManager(cache = cache)
    checkpoint_policy = CheckpointPolicy("manual")
    cache_policy = CachePolicy(timedelta(0), None, None)

    @ckpm.checkpoint(
        checkpoint_policy = checkpoint_policy,
        cache_policy = cache_policy,
    )
    def foo(a, b, c, d):
        return [a, b, c, d]
    prefix = f"{foo.__module__}.{foo.__qualname__}"

    foo_ret1 = foo(1, 2, 3, 4)
    assert len(cache.list_keys(prefix)) == 1
    foo_ret2 = foo(1, 2, 3, 4)
    assert foo_ret1 == foo_ret2
    assert len(cache.list_keys(prefix)) == 1


# %%
def test_CheckpointManager_persistent_cache(tmpfile_fixture):
    lstore = LocalFileStorage(TMP_DIR)
    cache = PersistentCache(lstore)
    ckpm = CheckpointManager(cache = cache)
    checkpoint_policy = CheckpointPolicy("cache")
    cache_policy = CachePolicy(timedelta(1))

    count = 1
    @ckpm.checkpoint(
        checkpoint_policy = checkpoint_policy,
        cache_policy = cache_policy,
    )
    def foo(a, b, c, d):
        nonlocal count
        count += 1
        return [a, b, c, d]
    prefix = f"{foo.__module__}.{foo.__qualname__}"

    foo_ret1 = foo(1, 2, 3, 7)
    assert foo_ret1 == [1, 2, 3, 7]
    assert lstore.list_keys(prefix) == cache.list_keys(prefix)
    key = lstore.list_keys(prefix)[0]
    assert cache.get(key) == [1, 2, 3, 7]
    assert b"json" in lstore.get(key)
    assert count == 2

    # Persisent cache.
    lstore = LocalFileStorage(TMP_DIR)
    cache = PersistentCache(lstore)
    ckpm = CheckpointManager(cache = cache)
    checkpoint_policy = CheckpointPolicy("cache")
    cache_policy = CachePolicy(timedelta(1))

    @ckpm.checkpoint(
        checkpoint_policy = checkpoint_policy,
        cache_policy = cache_policy,
    )
    def foo(a, b, c, d):
        nonlocal count
        count += 1
        return [a, b, c, d]

    foo_ret1 = foo(1, 2, 3, 7)
    assert foo_ret1 == [1, 2, 3, 7]
    assert count == 2


# %%
def test_CheckpointManager_persistent_cache_serialization_type_(tmpfile_fixture):
    lstore = LocalFileStorage(TMP_DIR)
    cache = PersistentCache(lstore)
    ckpm = CheckpointManager(cache = cache)
    checkpoint_policy = CheckpointPolicy("cache")
    cache_policy = CachePolicy(timedelta(1), "pickle")

    @ckpm.checkpoint(
        checkpoint_policy = checkpoint_policy,
        cache_policy = cache_policy,
    )
    def foo(a, b, c, d):
        return [a, b, c, d]
    prefix = f"{foo.__module__}.{foo.__qualname__}"

    foo_ret1 = foo(1, 2, 3, 7)
    assert foo_ret1 == [1, 2, 3, 7]
    assert lstore.list_keys(prefix) == cache.list_keys(prefix)
    key = lstore.list_keys(prefix)[0]
    assert cache.get(key) == [1, 2, 3, 7]

    # Serialization type should be `json` by default.
    assert b"pickle" in lstore.get(key)


# %%
def test_CheckpointManager_persistent_cache_inline(tmpfile_fixture):
    lstore = LocalFileStorage(TMP_DIR)
    cache = PersistentCache(lstore)
    ckpm = CheckpointManager(cache = cache)
    checkpoint_policy = CheckpointPolicy("cache")
    cache_policy = CachePolicy(timedelta(1), None, True)

    @ckpm.checkpoint(
        checkpoint_policy = checkpoint_policy,
        cache_policy = cache_policy,
    )
    def foo(a, b, c, d):
        return [a, b, c, d]
    prefix = f"{foo.__module__}.{foo.__qualname__}"

    foo_ret1 = foo(1, 2, 3, 7)
    assert foo_ret1 == [1, 2, 3, 7]
    assert lstore.list_keys(prefix) == cache.list_keys(prefix)
    key = lstore.list_keys(prefix)[0]

    # Inline cache still works after persistent storage deleted.
    assert lstore._data_path(key).is_file()
    lstore.delete(key)
    assert not lstore._data_path(key).is_file()
    assert cache.get(key) == [1, 2, 3, 7]
    assert not lstore._data_path(key).is_file()
    foo_ret1 = foo(1, 2, 3, 7)
    assert foo_ret1 == [1, 2, 3, 7]
    assert not lstore._data_path(key).is_file()

#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_cache.py
#   Author: xyy15926
#   Created: 2026-05-01 21:11:55
#   Updated: 2026-05-20 15:49:35
#   Description:
# ---------------------------------------------------------

# %%
import pytest
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, force=True)
    from importlib import reload
    from flagbear.slp import storage, cache
    reload(storage)
    reload(cache)

import numpy as np
import shutil
import dataclasses
from datetime import datetime, timedelta
import threading

from flagbear.slp.finer import get_tmp_path
from flagbear.slp.storage import(
    StorageBackend,
    LocalFileStorage,
)
from flagbear.slp.cache import(
    CacheMeta,
    meta_serialize,
    meta_deserialize,
    MemoryCache,
    PersistentCache,
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
def test_CacheMeta():
    meta = CacheMeta()
    json_bytes = meta.to_json()
    assert isinstance(json_bytes, bytes)
    loaded = meta.from_json(json_bytes)
    assert loaded == meta

    # From dict, CacheMeta and None
    now = datetime.now()
    meta = CacheMeta("somekey", created_at = now)
    assert meta == CacheMeta.from_meta({"key": "somekey", "created_at": now})
    assert meta == CacheMeta.from_meta(meta)


# %%
def test_meta_serialize():
    data = np.random.rand(10, 10)
    bytes_, meta = meta_serialize(data)
    assert meta.type_ == "numpy"
    loaded_data, loaded_meta = meta_deserialize(bytes_)
    assert np.allclose(data, loaded_data)
    assert meta == loaded_meta

    # Specify serialization type.
    meta = CacheMeta(type_ = "pickle")
    bytes_, meta = meta_serialize(data, meta)
    assert b"pickle" in bytes_
    loaded_data, loaded_meta = meta_deserialize(bytes_)
    assert np.allclose(data, loaded_data)
    assert meta == loaded_meta


# %%
def test_MemoryCache():
    mcache = MemoryCache()
    mcache.set("ok", 1)
    assert mcache.storage["ok"].last_accessed is None
    assert mcache.get("ok") == 1
    assert mcache.storage["ok"].last_accessed is not None
    assert mcache.get("nok") is None
    assert mcache._stats["hits"] == 1
    assert mcache._stats["misses"] == 1
    assert mcache.exists("ok")
    assert mcache.list_keys() == ["ok", ]
    assert mcache.delete("nok") is None
    assert mcache.delete("ok") is None
    assert mcache.list_keys() == [ ]

    # Time to live.
    ttl = timedelta(1)
    mcache = MemoryCache(ttl,)
    mcache.set("ok", 1)
    assert mcache.get("ok") == 1
    assert mcache.get("nok") is None
    assert mcache._stats["hits"] == 1
    assert mcache._stats["misses"] == 1
    assert mcache.exists("ok")
    assert mcache.list_keys() == ["ok", ]
    assert mcache.delete("nok") is None
    assert mcache.delete("ok") is None
    assert mcache.list_keys() == [ ]

    # Hook function.
    def on_hit(key: str):
        raise ValueError

    def on_miss(key: str):
        raise RuntimeError

    error_cache = MemoryCache(ttl, on_hit, on_miss)
    error_cache.set("ok", 1)
    with pytest.raises(ValueError):
        error_cache.get("ok")
    with pytest.raises(RuntimeError):
        error_cache.get("nok")

    # Cache expired.
    ttl = timedelta(0)
    mcache = MemoryCache(ttl,)
    mcache.set("ok", 1)
    assert mcache.exists("ok")
    assert mcache.get("ok") is None
    assert mcache.list_keys() == [ ]


# %%
def test_PersistetnCache_value_inline(tmpfile_fixture):
    lstorage = LocalFileStorage(TMP_DIR)

    pcache = PersistentCache(lstorage, max_mem_size = 1024)
    pcache.set("ok", 1)
    assert pcache.get("ok") == 1
    assert pcache.get("nok") is None
    assert pcache._stats["hits"] == 1
    assert pcache._stats["misses"] == 1
    assert pcache.exists("ok")
    assert pcache.list_keys() == ["ok", ]

    # Check inner meta-storage and local file storage.
    assert pcache.meta_storage["ok"].inline is True
    assert pcache.meta_storage["ok"].value == 1
    assert lstorage._data_path("ok").is_file()

    # Reinit persistent cachde from local storage.
    pcache = PersistentCache(lstorage, max_mem_size = 1024)
    assert pcache.meta_storage["ok"].key is None
    assert pcache.meta_storage["ok"].inline is None
    assert pcache.get("ok") == 1
    assert pcache.meta_storage["ok"].key == "ok"
    assert pcache.meta_storage["ok"].inline is True
    assert pcache.meta_storage["ok"].value == 1

    # Inline cache could be got though persistent storage deleted.
    lstorage.delete("ok")
    assert not lstorage._data_path("ok").is_file()
    assert pcache.get("ok") == 1

    # Delete could still be done once again for non-existed key.
    assert pcache.delete("nok") is None
    assert pcache.delete("ok") is None
    assert pcache.list_keys() == [ ]
    assert "ok" not in pcache.meta_storage
    assert not lstorage._data_path("ok").is_file()

    # Force not to save value inline.
    meta = CacheMeta("nonmem", inline = False)
    pcache.set("nonmem", 1, meta)
    assert not pcache.meta_storage["nonmem"].inline
    assert pcache.meta_storage["nonmem"].value is None
    assert lstorage._data_path("nonmem").is_file()
    lstorage.delete("nonmem")
    assert not lstorage._data_path("nonmem").is_file()
    with pytest.raises(RuntimeError):
        pcache.get("nonmem")


# %%
def test_PersistetnCache_value_not_inline(tmpfile_fixture):
    lstorage = LocalFileStorage(TMP_DIR)
    large_val = np.random.rand(40, 40).astype(np.float64)

    pcache = PersistentCache(lstorage, max_mem_size = 1024)
    pcache.set("ok", large_val)
    assert np.allclose(pcache.get("ok"), large_val)
    assert pcache._stats["hits"] == 1
    assert pcache.meta_storage["ok"].hits == 1

    # Check inner meta-storage and local file storage.
    assert not pcache.meta_storage["ok"].inline
    assert pcache.meta_storage["ok"].value is None
    assert lstorage._data_path("ok").is_file()

    # Recover from persistent storage.
    # But `._stats` and `.meta_storage` won't be recovered.
    new_lstorage = LocalFileStorage(TMP_DIR)
    new_pcache = PersistentCache(new_lstorage, max_mem_size = 1024)
    assert np.allclose(new_pcache.get("ok"), large_val)
    assert new_pcache._stats["hits"] == 1
    assert new_pcache.meta_storage["ok"].hits == 1
    assert new_pcache.meta_storage["ok"].inline is False

    # Force to save value inline.
    meta = CacheMeta("inmem", inline = True)
    pcache.set("inmem", large_val, meta)
    assert lstorage._data_path("inmem").is_file()
    lstorage.delete("inmem")
    assert not lstorage._data_path("inmem").is_file()
    assert np.allclose(pcache.get("inmem"), large_val)


# %%
def test_PersistetnCache_expire(tmpfile_fixture):
    lstorage = LocalFileStorage(TMP_DIR)
    ttl = timedelta(0)
    pcache = PersistentCache(lstorage, ttl = ttl, max_mem_size = 1024)

    large_val = np.random.rand(40, 40).astype(np.float64)
    pcache.set("ok", large_val)
    assert pcache.exists("ok")
    assert lstorage._data_path("ok").is_file()
    assert pcache.get("ok") is None
    assert pcache.list_keys() == [ ]
    assert not lstorage._data_path("ok").is_file()


# %%
def test_PersistetnCache_multi(tmpfile_fixture):
    lstorage = LocalFileStorage(TMP_DIR)
    pcache = PersistentCache(lstorage, max_mem_size = 1024)
    large_val = np.random.rand(40, 40).astype(np.float64)
    pcache.set("ok", large_val)

    ready = threading.Event()
    def update():
        ready.wait()
        pcache.set("ok", "haha")

    t = threading.Thread(target = update)
    t.start()
    assert np.allclose(pcache.get("ok"), large_val)
    ready.set()
    t.join()
    assert pcache.get("ok") == "haha"

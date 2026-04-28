#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_storage.py
#   Author: xyy15926
#   Created: 2026-04-22 11:38:44
#   Updated: 2026-04-22 23:00:40
#   Description:
# ---------------------------------------------------------

# %%
import pytest
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, force=True)
    from importlib import reload
    from flagbear.slp import storage
    reload(storage)

import numpy as np
import shutil
import dataclasses
from datetime import datetime
import threading

from flagbear.slp.finer import get_tmp_path
from flagbear.slp.storage import(
    DEFAULT_EXPIRE,
    StorageBackend,
    MetaData,
    LocalFileStorage,
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
def test_MetaData():
    meta = MetaData("meta", 10)
    ret = dict(
        key = "meta",
        size = 10,
        type_ = "pickle",
        hit_count = 0,
        created_at = datetime.now().isoformat(),
        expires_at = DEFAULT_EXPIRE,
        last_accessed = datetime.now().isoformat(),
    )
    assert dataclasses.asdict(meta).keys() == ret.keys()


# %%
def test_LocalFileStorage_single(tmpfile_fixture):
    lstorage = LocalFileStorage(TMP_DIR, 1024)
    assert (TMP_DIR / "meta").is_dir()
    assert (TMP_DIR / "data").is_dir()

    val1 = lstorage.get("key1")
    assert val1 is None

    # Inline storege.
    inline_val = np.random.rand(10, 10).astype(np.float64)
    lstorage.set("inline_key", inline_val.tobytes())
    assert lstorage.exists("inline_key")
    assert (lstorage.meta_dir / "inline_key.json").is_file()
    assert not (lstorage.data_dir / "inline_key.bin").exists()
    inline_bytes = lstorage.get("inline_key")
    inline_loaded = np.frombuffer(inline_bytes, np.float64).reshape(10, 10)
    assert np.all(inline_loaded == inline_val)
    assert "inline_key" in lstorage.list_keys()
    lstorage.delete("inline_key")
    assert not (lstorage.meta_dir / "inline_key.json").is_file()
    assert "inline_key" not in lstorage.list_keys()

    # Seperate storege.
    sep_val = np.random.rand(40, 40).astype(np.float64)
    lstorage.set("sep_key", sep_val.tobytes())
    assert lstorage.exists("sep_key")
    assert (lstorage.meta_dir / "sep_key.json").is_file()
    assert (lstorage.data_dir / "sep_key.bin").exists()
    sep_bytes = lstorage.get("sep_key")
    sep_loaded = np.frombuffer(sep_bytes, np.float64).reshape(40, 40)
    assert np.all(sep_loaded == sep_val)
    assert "sep_key" in lstorage.list_keys()
    lstorage.delete("sep_key")
    assert not (lstorage.meta_dir / "sep_key.json").is_file()
    assert not (lstorage.data_dir / "sep_key.bin").exists()
    assert "sep_key" not in lstorage.list_keys()


# %%
def test_LocalFileStorage_multi(tmpfile_fixture):
    lstorage = LocalFileStorage(TMP_DIR, 1024)
    assert (TMP_DIR / "meta").is_dir()
    assert (TMP_DIR / "data").is_dir()

    lstorage.set("inline_key", b"1")
    assert lstorage.get("inline_key") == b"1"

    ready = threading.Event()
    def update():
        ready.wait()
        lstorage.set("inline_key", b"2")

    t = threading.Thread(target = update)
    t.start()
    assert lstorage.get("inline_key") == b"1"
    ready.set()
    t.join()
    assert lstorage.get("inline_key") == b"2"


# %%
def test_LocalFileStorage_expire():
    # Expired and delete.
    lstorage = LocalFileStorage(TMP_DIR, 1024)
    inline_val = np.random.rand(10, 10).astype(np.float64)
    lstorage.set(
        "inline_key",
        inline_val.tobytes(),
        {
            "created_at": datetime.now().isoformat(),
            "expires_at": datetime.now().isoformat(),
        }
    )
    assert lstorage.get("inline_key") is None
    assert len(lstorage.list_keys()) == 0

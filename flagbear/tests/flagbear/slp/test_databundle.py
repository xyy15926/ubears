#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_databundle.py
#   Author: xyy15926
#   Created: 2026-04-11 20:47:49
#   Updated: 2026-04-12 19:54:44
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
import pytest
if __name__ == "__main__":
    from importlib import reload
    from flagbear.slp import finer, databundle
    reload(finer)
    reload(databundle)

import numpy as np
import pickle
import json
import shutil
from flagbear.slp.finer import get_tmp_path
from flagbear.slp.databundle import (
    DataBundle,
    DataBundleFactory,
    PickableBundle,
)

TMP_DIR = get_tmp_path() / "pytest_tmpdir"

# %%
@pytest.fixture(scope="module", autouse=False)
def tmpfile_fixture(request):
    yield

    # Remove the tmp file during the pytest.
    # Clear only once with `scope=module`.
    pytest_tmp = get_tmp_path() / TMP_DIR
    shutil.rmtree(pytest_tmp, ignore_errors=True)
    if not any(get_tmp_path().iterdir()):
        get_tmp_path().rmdir()


# %%
def test_DataBundleFactory_ndarray(tmpfile_fixture):
    def dumps_ndarray(arr):
        return arr.dumps()

    def loads_ndarray(bytes_):
        return pickle.loads(bytes_)

    cls_name = "NDABundle"
    zip_file = get_tmp_path() / TMP_DIR / (cls_name + ".zip")
    NDABundle = DataBundleFactory.from_slfunc(
        cls_name,
        dumps_ndarray,
        loads_ndarray,
    )
    data = np.random.randint(1, 100, (3, 4))
    bundle = NDABundle(data)
    bundle.save_bundle(zip_file, forced=True)
    assert zip_file.is_file()
    loaded = DataBundleFactory.load_instance(cls_name, zip_file)
    assert np.all(loaded.data == bundle.data)

    # Can't override the exists file by default.
    with pytest.raises(ValueError):
        bundle.save_bundle(zip_file)


# %%
def test_DataBundleFactory_dict(tmpfile_fixture):
    def dumps_json(content):
        return json.dumps(content)

    def loads_json(bytes_):
        return json.loads(bytes_)

    cls_name = "JsonBundle"
    zip_file = TMP_DIR / (cls_name + ".zip")
    JsonBundle = DataBundleFactory.from_slfunc(
        cls_name,
        dumps_json,
        loads_json,
    )
    data = {"country": "CHN", "city": "北京"}
    bundle = JsonBundle(data)
    bundle.trace("kaka", {"a": 1})
    bundle.trace("kaka", {"b": 2})
    bundle.add_metadata("xixi", 1)
    bundle.add_metadata("xixi", 2)
    bundle.save_bundle(zip_file, forced=True)
    assert zip_file.is_file()
    loaded = DataBundleFactory.load_instance(cls_name, zip_file)
    # `dataclasses.dataclass` could be compared directly.
    assert loaded == bundle
    assert loaded.lineage == {"kaka": {"a": 1, "b": 2}}
    assert loaded.metadata == {"xixi": 2}


# %%
def test_PiclableBundle_and_try_load(tmpfile_fixture):
    cls_name = "PickableBundle"
    zip_file = TMP_DIR / (cls_name + ".zip")

    data = {"country": "CHN", "city": "北京"}
    bundle = PickableBundle(data)
    bundle.save_bundle(zip_file, forced=True)
    loaded = DataBundleFactory.load_instance(cls_name, zip_file)

    # `dataclasses.dataclass` could be compared directly.
    assert loaded == bundle

    loaded = DataBundleFactory.try_load_instance(zip_file)
    assert loaded == bundle

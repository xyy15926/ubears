#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_databundle.py
#   Author: xyy15926
#   Created: 2026-04-11 20:47:49
#   Updated: 2026-04-27 22:32:19
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations

import pytest

if __name__ == "__main__":
    from importlib import reload

    from flagbear.slp import databundle, finer

    reload(finer)
    reload(databundle)

import json
import pickle
import shutil

import numpy as np

from flagbear.slp.databundle import (
    DataBundleFactory,
    PickableBundle,
    bundle_cache,
    concat_params,
)
from flagbear.slp.finer import get_tmp_path

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
def test_DataBundleFactory_ndarray(tmpfile_fixture):
    def dumps_ndarray(arr):
        return arr.dumps()

    def loads_ndarray(bytes_, metadata: dict | None = None):
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

    def loads_json(bytes_, metadata: dict | None = None):
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


# %%
def test_bundle_cache(tmpfile_fixture):
    from datetime import datetime

    today = datetime.now().isoformat().replace("-", "")[:8]
    bundle_dir = TMP_DIR

    bundle_type = "create_dict"

    @bundle_cache(bundle_type, dest=PYTEST_DIR)
    def create_dict(a=1, b=2):
        return {"a": a, "b": b}

    ret1 = create_dict()
    reg_name1 = concat_params(bundle_type, (), {})
    bundle_file = bundle_dir / f"{reg_name1}_{today}_0001.zip"
    assert bundle_file.is_file()

    ret1_loaded = create_dict()
    assert ret1 == ret1_loaded
    ret1_loaded = DataBundleFactory.load_instance(
        reg_name1 + ".zip", bundle_file
    )
    assert ret1 == ret1_loaded
    ret1_loaded = DataBundleFactory.try_load_instance(bundle_file)
    assert ret1 == ret1_loaded

    # Function with different params will regist as different DataBundle.
    ret2 = create_dict(1, b=3)
    reg_name2 = concat_params(bundle_type, (1,), {"b": 3})
    assert reg_name2 != reg_name1
    bundle_file2 = bundle_dir / f"{reg_name2}_{today}_0001.zip"
    assert bundle_file2.is_file()

    ret2_loaded = create_dict(1, b=3)
    assert ret2 == ret2_loaded
    ret2_loaded = DataBundleFactory.load_instance(
        reg_name2 + ".zip", bundle_file2
    )
    assert ret2 == ret2_loaded
    ret2_loaded = DataBundleFactory.try_load_instance(bundle_file2)
    assert ret2 == ret2_loaded

    # The former data bundle could still be loaded.
    ret1_again = create_dict()
    assert ret1 == ret1_again

    # Force to fetch the remote or expensive data.
    create_dict(1, b=3, forced=True)
    bundle_file2_new = bundle_dir / f"{reg_name2}_{today}_0002.zip"
    assert bundle_file2_new.is_file()

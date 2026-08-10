#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_databundle.py
#   Author: xyy15926
#   Created: 2026-04-11 20:47:49
#   Updated: 2026-04-20 20:18:21
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations

import pytest

if __name__ == "__main__":
    from importlib import reload

    from dirtbear.quant import dataloader
    from flagbear.slp import databundle, finer

    reload(finer)
    reload(databundle)
    reload(dataloader)

import shutil

import pandas as pd

from dirtbear.quant.dataloader import csv_cache
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
def test_csv_cache(tmpfile_fixture):
    ONLY_ONCE = True

    @csv_cache(dest=PYTEST_DIR)
    def fetch_df():
        nonlocal ONLY_ONCE
        if not ONLY_ONCE:
            raise ValueError("Should be called only once.")
        df = pd.DataFrame(
            {
                "a": [1, 1, 1],
                "b": [1, 1, 1.0],
                "c": ["a", "b", "测试"],
                "d": [1, 1.0, "测试"],
            }
        )
        ONLY_ONCE = False
        return df

    assert ONLY_ONCE
    df_ori = fetch_df()
    assert not ONLY_ONCE
    df_loaded = fetch_df()
    assert all(df_ori.data == df_loaded.data)

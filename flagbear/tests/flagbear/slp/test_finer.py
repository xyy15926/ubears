#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_finer.py
#   Author: xyy15926
#   Created: 2024-10-24 20:18:21
#   Updated: 2026-04-12 21:58:55
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
import pytest
if __name__ == "__main__":
    from importlib import reload
    from flagbear.slp import finer
    reload(finer)

import re
from datetime import date
import shutil
from flagbear.slp.finer import (
    date_order_mark,
    tmp_file,
    get_tmp_path,
    use_file,
    use_dir,
)

TMP_DIR = "pytest_tmpdir"
TMP_FNAME = f"{TMP_DIR}/tmpf.tmp"
TMP_FNAME_REGEX = f"{TMP_DIR}/tmpf_E0006.tmp"
_PTN = r"E\d{4}"
TMP_FNAME_REGEX_PTN = rf"{TMP_DIR}/tmpf_{_PTN}.tmp"


# %%
def test_date_order_mark():
    today = date.today().isoformat().replace("-", "")
    kstr = "kstr"
    assert date_order_mark(kstr) == f"{kstr}_{today}_0001"
    assert (date_order_mark(kstr, [f"{kstr}_{today}_0006"])
            == f"{kstr}_{today}_0007")
    assert (date_order_mark(kstr, [f"{kstr}_{today}_0006"], "20200101")
            == f"{kstr}_20200101_0001")
    assert (date_order_mark(kstr, [f"{kstr}_{today}_0006"], "20200101", 5)
            == f"{kstr}_20200101_0005")
    assert (date_order_mark(kstr, [f"{kstr}_{today}_0006"], None, 5)
            == f"{kstr}_{today}_0011")

    # Regex key string.
    kstr_ptn = r"kstr_E\d{4}"
    kstr = r"kstr_E0001"
    assert date_order_mark(kstr_ptn) == f"{kstr_ptn}_{today}_0001"
    assert (date_order_mark(kstr_ptn, [f"{kstr}_{today}_0006"])
            == f"{kstr}_{today}_0007")
    assert (date_order_mark(kstr_ptn, [f"{kstr}_{today}_0006"], "20200101")
            == f"{kstr_ptn}_20200101_0001")
    assert (date_order_mark(kstr_ptn, [f"{kstr}_{today}_0006"], "20200101", 5)
            == f"{kstr_ptn}_20200101_0005")
    assert (date_order_mark(kstr_ptn, [f"{kstr}_{today}_0006"], None, 5)
            == f"{kstr}_{today}_0011")

    # Regex key string may lead to unpredictable result.
    kstr2 = r"kstr_E0002"
    assert (date_order_mark(kstr_ptn,
                            [f"{kstr}_{today}_0006", f"{kstr2}_{today}_0001"],
                            None, 5)
            == f"{kstr2}_{today}_0011")


# %%
@pytest.fixture(scope="function", autouse=False)
def tmpfile_fixture(request):
    # Process before test.
    print(request)
    tmp_fname = tmp_file(TMP_FNAME)
    regex_tfname = tmp_file(TMP_FNAME_REGEX)
    tmp_fname.touch()
    regex_tfname.touch()

    # Return the result.
    yield tmp_fname, regex_tfname

    # Process after test.
    tfdir = get_tmp_path() / TMP_DIR
    shutil.rmtree(tfdir, ignore_errors=True)
    if not any(get_tmp_path().iterdir()):
        get_tmp_path().rmdir()


# %%
def test_tmp_file(tmpfile_fixture):
    tmp_fname, regex_tfname = tmpfile_fixture
    assert tmp_fname.is_file()
    assert regex_tfname.is_file()

    # Raise FileExistsError since `tmp_file` exists, and directory with
    # the same name can't be made.
    with pytest.raises(FileExistsError):
        tmp_file(tmp_fname / "any_file")

    # `tmp_file` will update order mark automatically.
    ordix = re.search(r"_(\d{4})\.", tmp_fname.name).groups()[0]
    nbname = tmp_fname.name.replace(ordix, f"{int(ordix) + 1:04}")
    assert tmp_file(TMP_FNAME).name == nbname

    # `tmp_file` can return exact filename with fuzzy regex.
    ordix = re.search(r"_(\d{4})\.", regex_tfname.name).groups()[0]
    nbname = regex_tfname.name.replace(ordix, f"{int(ordix) + 1:04}")
    assert tmp_file(TMP_FNAME_REGEX_PTN).name == nbname

    assert tmp_file(TMP_FNAME, None, 0) == tmp_fname
    assert tmp_file(TMP_FNAME_REGEX_PTN, None, 0) == regex_tfname


# %%
def test_use_file(tmpfile_fixture):
    tmp_fname, regex_tfname = tmpfile_fixture

    # `use_file` will update order mark automatically.
    ordix = re.search(r"_(\d{4})\.", tmp_fname.name).groups()[0]
    nbname = tmp_fname.name.replace(ordix, f"{int(ordix) + 1:04}")
    assert use_file(TMP_FNAME).name == nbname

    # `use_file` can return exact filename with fuzzy regex.
    ordix = re.search(r"_(\d{4})\.", regex_tfname.name).groups()[0]
    nbname = regex_tfname.name.replace(ordix, f"{int(ordix) + 1:04}")
    assert use_file(TMP_FNAME_REGEX_PTN).name == nbname

    # `use_file` will try to use the exact absolute path name if provide.
    assert use_file(str(tmp_fname)) == tmp_fname
    # `use_file` accept `Path` as the argument.
    assert use_file(tmp_fname) == tmp_fname

    assert use_file(TMP_FNAME, None, 0) == tmp_fname
    assert use_file(TMP_FNAME_REGEX_PTN, None, 0) == regex_tfname


# %%
def test_use_dir(tmpfile_fixture):
    tmp_fname, regex_tfname = tmpfile_fixture

    # `use_dir` will update order mark automatically.
    ordix = re.search(r"_(\d{4})\.", tmp_fname.name).groups()[0]
    nbname = tmp_fname.name.replace(ordix, f"{int(ordix) + 1:04}")
    new_dir = use_dir(TMP_FNAME)
    assert new_dir.name == nbname
    assert new_dir.is_dir()

    # `use_dir` can return exact filename with fuzzy regex.
    ordix = re.search(r"_(\d{4})\.", regex_tfname.name).groups()[0]
    nbname = regex_tfname.name.replace(ordix, f"{int(ordix) + 1:04}")
    new_dir = use_dir(TMP_FNAME_REGEX_PTN)
    assert new_dir.name == nbname
    assert new_dir.is_dir()

    # `use_dir` will try to use the exact absolute path name if provide.
    new_dir = use_dir(str(tmp_fname.parent / nbname))
    assert new_dir.is_dir()
    assert new_dir.parent == regex_tfname.parent

    # `use_dir` accept `Path` as the argument.
    nbname = regex_tfname.name.replace(ordix, f"{int(ordix) + 2:04}")
    new_dir = use_dir(tmp_fname.parent / nbname)
    assert new_dir.is_dir()
    assert new_dir.parent == regex_tfname.parent

    assert use_file(TMP_FNAME_REGEX_PTN, None, 0).name == nbname

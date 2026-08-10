#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_callables.py
#   Author: xyy15926arruns
#   Created: 2025-02-20 18:56:36
#   Updated: 2026-04-03 11:53:10
#   Description:
# ---------------------------------------------------------

# %%
import pytest

if __name__ == "__main__":
    from importlib import reload

    from dirtbear.dflater import callables
    from dirtbear.locale import calender
    from flagbear.llp import parser

    reload(callables)
    reload(parser)
    reload(calender)

import numpy as np
import pandas as pd

from dirtbear.dflater.callables import (
    argmax,
    argmaxs,
    argmins,
    avg,
    cb_fst,
    cb_max,
    cb_min,
    coef_var,
    day_itvl,
    drop_duplicates,
    flat1_max,
    get_hour,
    getn,
    is_busiday,
    isin,
    map,
    # Aggregation
    max,
    mon_itvl,
    nnfilter,
    not_busiday,
    sep_map,
    # Transformation
    sortby,
    sum,
)
from dirtbear.dflater.exoptim import get_envp


# ---------------------------------------------------------------------------
#                    * * * Aggregation Callables * * *
# ---------------------------------------------------------------------------
# %%
def test_nnfilter():
    envp = get_envp()

    # List, np.ndarray, pd.Series.
    x = [1, 2, 2, None]
    ret = nnfilter(x)
    assert np.all(ret == [1, 2, 2])
    env = {"x": x}
    ret = envp.bind_env(env).parse("nnfilter(x)")
    assert np.all(ret == [1, 2, 2])

    x = np.array([1, 2, 2, None])
    ret = nnfilter(x)
    assert np.all(ret == [1, 2, 2])
    env = {"x": x}
    ret = envp.bind_env(env).parse("nnfilter(x)")
    assert np.all(ret == [1, 2, 2])

    x = pd.Series([1, 2, 2, None])
    ret = nnfilter(x)
    assert np.all(np.isclose(ret, [1, 2, 2, np.nan], equal_nan=True))
    env = pd.DataFrame({"x": x})
    ret = envp.bind_env(env).parse("nnfilter(x)")
    assert np.all(np.isclose(ret, [1, 2, 2, np.nan], equal_nan=True))

    # Empty sequence.
    ret = nnfilter([])
    assert ret == []
    env = {"x": []}
    ret = envp.bind_env(env).parse("nnfilter(x)")
    assert ret == []


# %%
def test_getn():
    envp = get_envp()

    # List, np.ndarray, pd.Series.
    x = [1, 2, 3, 4]
    ret = getn(x, 1)
    assert ret == 2
    ret = getn(x, 7)
    assert ret is None
    ret = getn(x, None)
    assert ret is None

    env = {"x": x}
    ret = envp.bind_env(env).parse("getn(x, 1)")
    assert ret == 2
    ret = envp.bind_env(env).parse("getn(x, 7)")
    assert ret is None
    ret = envp.bind_env(env).parse("getn(x, None)")
    assert ret is None

    x = np.array([1, 2, 3, 4])
    ret = getn(x, 1)
    assert ret == 2
    ret = getn(x, 7)
    assert ret is None
    ret = getn(x, None)
    assert ret is None

    env = {"x": x}
    ret = envp.bind_env(env).parse("getn(x, 1)")
    assert ret == 2
    ret = envp.bind_env(env).parse("getn(x, 7)")
    assert ret is None
    ret = envp.bind_env(env).parse("getn(x, None)")
    assert ret is None

    x = pd.Series([1, 2, 3, 4])
    ret = getn(x, 1)
    assert ret == 2
    ret = getn(x, 7)
    assert ret is None
    ret = getn(x, None)
    assert ret is None

    env = pd.DataFrame({"x": x})
    ret = envp.bind_env(env).parse("getn(x, 1)")
    assert ret == 2
    ret = envp.bind_env(env).parse("getn(x, 7)")
    assert ret is None
    ret = envp.bind_env(env).parse("getn(x, None)")
    assert ret is None

    # Empty sequence.
    ret = getn([], 1)
    assert ret is None
    ret = envp.bind_env(env).parse("getn([], 1)")
    assert ret is None


# %%
def _assert_dropdup_equal(ret, expected):
    for le, re in zip(ret.ravel(), expected, strict=True):
        assert le == re or np.isnan(le)


def _dropdup_single(envp, x, env):
    expected = [1, 2, 3, "a", None, np.nan]
    _assert_dropdup_equal(drop_duplicates([x]), expected)
    _assert_dropdup_equal(drop_duplicates(x), expected)
    _assert_dropdup_equal(
        envp.bind_env(env).parse("drop_duplicates([x,])"), expected
    )
    _assert_dropdup_equal(
        envp.bind_env(env).parse("drop_duplicates(x)"), expected
    )


def test_dropdup():
    envp = get_envp()

    # Single List, np.ndarray and pd.Series.
    x = [1, 1, 2, 3, "a", None, None, np.nan, np.nan]
    _dropdup_single(envp, x, {"x": x})

    x = np.array([1, 1, 2, 3, "a", None, None, np.nan, np.nan], dtype="O")
    _dropdup_single(envp, x, {"x": x})

    x = pd.Series([1, 1, 2, 3, "a", None, None, np.nan, np.nan])
    _dropdup_single(envp, x, pd.DataFrame({"x": x}))

    # Empty single sequences.
    assert len(drop_duplicates([])) == 0
    assert len(drop_duplicates([[]])) == 0
    assert len(envp.bind_env({"x": []}).parse("drop_duplicates([])")) == 0
    assert len(envp.bind_env({"x": []}).parse("drop_duplicates([[]])")) == 0

    # Single factor of multiple sequences.
    ret = drop_duplicates([1, 1])
    assert np.all(ret == [1])
    assert ret.shape == (1,)
    ret = envp.bind_env({}).parse("drop_duplicates([1, 1])")
    assert np.all(ret == [1])
    assert ret.shape == (1,)

    # Multiple List, np.ndarray and pd.Series.
    expected_multi = [1, 1, 2, 2, 3, "a", "a", 5, None, 5, np.nan, 6]
    x = [1, 1, 2, 3, "a", None, None, np.nan, np.nan]
    y = [1, 1, 2, "a", 5, 5, 5, 6, 6]
    _assert_dropdup_equal(drop_duplicates([x, y]), expected_multi)
    _assert_dropdup_equal(
        envp.bind_env({"x": x, "y": y}).parse("drop_duplicates([x,y])"),
        expected_multi,
    )

    x = np.array([1, 1, 2, 3, "a", None, None, np.nan, np.nan], dtype="O")
    y = np.array([1, 1, 2, "a", 5, 5, 5, 6, 6], dtype="O")
    _assert_dropdup_equal(drop_duplicates([x, y]), expected_multi)
    _assert_dropdup_equal(
        envp.bind_env({"x": x, "y": y}).parse("drop_duplicates([x,y])"),
        expected_multi,
    )

    x = pd.Series([1, 1, 2, 3, "a", None, None, np.nan, np.nan])
    y = pd.Series([1, 1, 2, "a", 5, 5, 5, 6, 6])
    _assert_dropdup_equal(drop_duplicates([x, y]), expected_multi)
    _assert_dropdup_equal(
        envp.bind_env(pd.DataFrame({"x": x, "y": y})).parse(
            "drop_duplicates([x,y])"
        ),
        expected_multi,
    )

    # Empty multiple sequences.
    assert len(drop_duplicates([[], []])) == 0
    assert len(envp.bind_env({}).parse("drop_duplicates([[], []])")) == 0

    # Single factor of multiple sequences.
    ret = drop_duplicates([[1, 1], [1, 1]])
    assert np.all(ret == [[1, 1]])
    assert ret.shape == (1, 2)
    ret = envp.bind_env({}).parse("drop_duplicates([[1, 1], [1, 1]])")
    assert np.all(ret == [[1, 1]])
    assert ret.shape == (1, 2)


# %%
def test_maxminavg():
    envp = get_envp()

    # List, np.ndarray, pd.Series for max, avg and sum.
    # Skip min.
    x = [1, 2, 2, np.nan]
    ret = max(x)
    assert ret == 2
    ret = avg(x)
    assert ret == 5 / 3
    ret = sum(x)
    assert ret == 5

    env = {"x": x}
    ret = envp.bind_env(env).parse("max(x)")
    assert ret == 2
    ret = envp.bind_env(env).parse("avg(x)")
    assert ret == 5 / 3
    ret = envp.bind_env(env).parse("sum(x)")
    assert ret == 5

    x = np.array([1, 2, 2, np.nan])
    ret = max(x)
    assert ret == 2
    ret = avg(x)
    assert ret == 5 / 3
    ret = sum(x)
    assert ret == 5

    env = {"x": x}
    ret = envp.bind_env(env).parse("max(x)")
    assert ret == 2
    ret = envp.bind_env(env).parse("avg(x)")
    assert ret == 5 / 3
    ret = envp.bind_env(env).parse("sum(x)")
    assert ret == 5

    x = pd.Series([1, 2, 2, np.nan])
    ret = max(x)
    assert ret == 2
    ret = avg(x)
    assert ret == 5 / 3
    ret = sum(x)
    assert ret == 5

    env = {"x": x}
    ret = envp.bind_env(env).parse("max(x)")
    assert ret == 2
    ret = envp.bind_env(env).parse("avg(x)")
    assert ret == 5 / 3
    ret = envp.bind_env(env).parse("sum(x)")
    assert ret == 5

    # Empty sequence for max, avg and sum.
    # Skip min.
    ret = max([])
    assert np.isnan(ret)
    ret = avg([])
    assert np.isnan(ret)
    ret = sum([])
    assert ret == 0

    ret = envp.bind_env(env).parse("max([])")
    assert np.isnan(ret)
    ret = envp.bind_env(env).parse("avg([])")
    assert np.isnan(ret)
    ret = envp.bind_env(env).parse("sum([])")
    assert ret == 0

    # All-nan sequence for max.
    x = [np.nan] * 5
    ret = max(x)
    assert np.isnan(ret)
    ret = avg(x)
    assert np.isnan(ret)
    ret = sum(x)
    assert ret == 0

    env = {"x": x}
    ret = envp.bind_env(env).parse("max(x)")
    assert np.isnan(ret)
    ret = envp.bind_env(env).parse("avg(x)")
    assert np.isnan(ret)
    ret = envp.bind_env(env).parse("sum(x)")
    assert ret == 0

    # All-nan sequence for max.
    x = [np.inf] * 5
    ret = max(x)
    assert np.isinf(ret)
    ret = avg(x)
    assert np.isinf(ret)
    ret = sum(x)
    assert np.isinf(ret)

    env = {"x": x}
    ret = envp.bind_env(env).parse("max(x)")
    assert np.isinf(ret)
    ret = envp.bind_env(env).parse("avg(x)")
    assert np.isinf(ret)
    ret = envp.bind_env(env).parse("sum(x)")
    assert np.isinf(ret)


# %%
def test_argmaxmin():
    envp = get_envp()

    # List, np.ndarray, pd.Series for `argmax`.
    # Skip `argmin`.
    x = [1, 1, 2, 2, np.nan]
    ret = argmax(x)
    assert ret == 2
    env = {"x": x}
    ret = envp.bind_env(env).parse("argmax(x)")
    assert ret == 2

    x = np.array([1, 1, 2, 2, np.nan])
    ret = argmax(x)
    assert ret == 2
    env = {"x": x}
    ret = envp.bind_env(env).parse("argmax(x)")
    assert ret == 2

    x = pd.Series([1, 1, 2, 2, np.nan])
    ret = argmax(x)
    assert ret == 2
    env = pd.DataFrame({"x": x})
    ret = envp.bind_env(env).parse("argmax(x)")
    assert ret == 2

    # Empty sequence for `argmax`.
    # Skip `argmin`.
    ret = argmax([])
    assert ret is None
    ret = envp.bind_env(env).parse("argmax([])")
    assert ret is None


# %%
def test_argmaxsmins():
    envp = get_envp()

    # List, np.ndarray, pd.Series of numbers.
    x = [1, 1, 2, 2, np.nan]
    y = [1, 2, 3, 4, 5]
    ret = argmaxs(x, y)
    assert np.all(ret == [3, 4])
    ret = argmins(x, y)
    assert np.all(ret == [1, 2])

    env = {"x": x, "y": y}
    ret = envp.bind_env(env).parse("argmaxs(x, y)")
    assert np.all(ret == [3, 4])
    ret = envp.bind_env(env).parse("argmins(x, y)")
    assert np.all(ret == [1, 2])

    x = np.array([1, 1, 2, 2, np.nan])
    y = np.array([1, 2, 3, 4, 5])
    ret = argmaxs(x, y)
    assert np.all(ret == [3, 4])
    ret = argmins(x, y)
    assert np.all(ret == [1, 2])

    env = {"x": x, "y": y}
    ret = envp.bind_env(env).parse("argmaxs(x, y)")
    assert np.all(ret == [3, 4])
    ret = envp.bind_env(env).parse("argmins(x, y)")
    assert np.all(ret == [1, 2])

    x = pd.Series([1, 1, 2, 2, np.nan])
    y = pd.Series([1, 2, 3, 4, 5])
    ret = argmaxs(x, y)
    assert np.all(ret == [3, 4])
    ret = argmins(x, y)
    assert np.all(ret == [1, 2])

    env = pd.DataFrame({"x": x, "y": y})
    ret = envp.bind_env(env).parse("argmaxs(x, y)")
    assert np.all(ret == [3, 4])
    ret = envp.bind_env(env).parse("argmins(x, y)")
    assert np.all(ret == [1, 2])

    # All NAs.
    x = np.array([np.nan] * 5)
    y = np.array([1, 2, 3, 4, 5])
    with pytest.warns(RuntimeWarning):
        ret = argmaxs(x, y)
        assert len(ret) == 0

    env = {"x": x, "y": y}
    with pytest.warns(RuntimeWarning):
        ret = envp.bind_env(env).parse("argmaxs(x, y)")
        assert len(ret) == 0

    # np.ndarray, pd.Series of datetime64.
    x = np.array(
        ["2021-12-12", "2021-11-11", "2022-11-11", "2021-11-11", "NaT"],
        dtype="M8[D]",
    )
    y = [1, 2, 3, 4, 5]
    ret = argmaxs(x, y)
    assert np.all(ret == 3)
    ret = argmins(x, y)
    assert np.all(ret == [2, 4])

    env = {"x": x, "y": y}
    ret = envp.bind_env(env).parse("argmaxs(x, y)")
    assert np.all(ret == 3)
    ret = envp.bind_env(env).parse("argmins(x, y)")
    assert np.all(ret == [2, 4])

    x = pd.Series(
        ["2021-12-12", "2021-11-11", "2022-11-11", "2021-11-11", "NaT"],
        dtype="M8[ms]",
    )
    y = [1, 2, 3, 4, 5]
    ret = argmaxs(x, y)
    assert np.all(ret == 3)
    ret = argmins(x, y)
    assert np.all(ret == [2, 4])

    env = pd.DataFrame({"x": x, "y": y})
    ret = envp.bind_env(env).parse("argmaxs(x, y)")
    assert np.all(ret == 3)
    ret = envp.bind_env(env).parse("argmins(x, y)")
    assert np.all(ret == [2, 4])

    # Empty sequence.
    ret = argmaxs([], [])
    assert len(ret) == 0
    ret = argmins([], [])
    assert len(ret) == 0

    ret = envp.bind_env(env).parse("argmaxs([], [])")
    assert len(ret) == 0
    ret = envp.bind_env(env).parse("argmins([], [])")
    assert len(ret) == 0


# %%
def test_cals():
    envp = get_envp()

    # `flat1_max` with sequences of difference length.
    x = [1, 1, 1, 1, 0, 4, 5, 0]
    ret = flat1_max(x)
    assert ret == 4
    ret = flat1_max([])
    assert ret == 0
    ret = flat1_max([5])
    assert ret == 1
    ret = flat1_max([0])
    assert ret == 0

    env = {"x": x}
    ret = envp.bind_env(env).parse("flat1_max(x)")
    assert ret == 4
    ret = envp.bind_env(env).parse("flat1_max([])")
    assert ret == 0
    ret = envp.bind_env(env).parse("flat1_max([5])")
    assert ret == 1
    ret = envp.bind_env(env).parse("flat1_max([0])")
    assert ret == 0

    # Coefficent of variation.
    x = [1, 2, 1, 2, 0]
    ret = coef_var(x)
    assert ret == np.std(x) / np.mean(x)
    ret = coef_var([])
    assert ret == 0

    env = {"x": x}
    ret = envp.bind_env(env).parse("coef_var(x)")
    assert ret == np.std(env["x"]) / np.mean(env["x"])
    ret = envp.bind_env(env).parse("coef_var([])")
    assert ret == 0


# %% -------------------------------------------------------------------------
#                    * * * Transformation Callables * * *
# %% -------------------------------------------------------------------------
def test_sortby():
    envp = get_envp()

    # List, np.ndarray and pd.Series.
    x = [11, np.nan, 13, 4, np.nan]
    y = [1, 12, 3, 14, np.nan]
    ret = sortby(x, y)
    assert np.all(np.isclose(ret, [11, 13, np.nan, 4, np.nan], equal_nan=True))

    env = {"x": x, "y": y}
    ret = envp.bind_env(env).parse("sortby(x, y)")
    assert np.all(np.isclose(ret, [11, 13, np.nan, 4, np.nan], equal_nan=True))

    x = np.array([11, np.nan, 13, 4, np.nan])
    y = np.array([1, 12, 3, 14, np.nan])
    ret = sortby(x, y)
    assert np.all(np.isclose(ret, [11, 13, np.nan, 4, np.nan], equal_nan=True))

    env = {"x": x, "y": y}
    ret = envp.bind_env(env).parse("sortby(x, y)")
    assert np.all(np.isclose(ret, [11, 13, np.nan, 4, np.nan], equal_nan=True))

    x = pd.Series([11, np.nan, 13, 4, np.nan])
    y = pd.Series([1, 12, 3, 14, np.nan])
    ret = sortby(x, y)
    assert np.all(np.isclose(ret, [11, 13, np.nan, 4, np.nan], equal_nan=True))

    env = pd.DataFrame({"x": x, "y": y})
    ret = envp.bind_env(env).parse("sortby(x, y)")
    assert np.all(np.isclose(ret, [11, 13, np.nan, 4, np.nan], equal_nan=True))

    # Empty sequence.
    ret = sortby([], [])
    assert len(ret) == 0
    ret = envp.bind_env(env).parse("sortby([], [])")
    assert len(ret) == 0


# %%
def test_map():
    ref = dict(zip("abcde", range(1, 6), strict=True))

    def refcall(ele):
        refi = dict(zip("abcde", range(1, 6), strict=True))
        return refi.get(ele, 7)

    envp = get_envp({"ref": ref, "refcall": refcall})

    # Map
    x = ["a", "c", "d", "e", "b", None, "NA"]
    ret = map(x, ref)
    assert np.all(
        np.isclose(ret, [1, 3, 4, 5, 2, np.nan, np.nan], equal_nan=True)
    )
    assert np.issubdtype(ret.dtype, np.floating)
    ret = map(x, ref, None)
    assert np.all(ret == [1, 3, 4, 5, 2, None, None])
    ret = map(x, refcall)
    assert np.all(np.isclose(ret, [1, 3, 4, 5, 2, 7, 7], equal_nan=True))

    env = {"x": x}
    ret = envp.bind_env(env).parse("map(x, ref)")
    assert np.all(
        np.isclose(ret, [1, 3, 4, 5, 2, np.nan, np.nan], equal_nan=True)
    )
    ret = envp.bind_env(env).parse("map(x, ref, None)")
    assert np.all(ret == [1, 3, 4, 5, 2, None, None])
    ret = envp.bind_env(env).parse("map(x, refcall)")
    assert np.all(np.isclose(ret, [1, 3, 4, 5, 2, 7, 7], equal_nan=True))

    x = np.array(["a", "c", "d", "e", "b", None, "NA"])
    ret = map(x, ref)
    assert np.all(
        np.isclose(ret, [1, 3, 4, 5, 2, np.nan, np.nan], equal_nan=True)
    )
    ret = map(x, ref, None)
    assert np.all(ret == [1, 3, 4, 5, 2, None, None])
    ret = map(x, refcall)
    assert np.all(np.isclose(ret, [1, 3, 4, 5, 2, 7, 7], equal_nan=True))

    x = np.array(["a", "c", "d", "e", "b", None, "NA"])
    env = {"x": x}
    ret = envp.bind_env(env).parse("map(x, ref)")
    assert np.all(
        np.isclose(ret, [1, 3, 4, 5, 2, np.nan, np.nan], equal_nan=True)
    )
    ret = envp.bind_env(env).parse("map(x, ref, None)")
    assert np.all(ret == [1, 3, 4, 5, 2, None, None])
    ret = envp.bind_env(env).parse("map(x, refcall)")
    assert np.all(np.isclose(ret, [1, 3, 4, 5, 2, 7, 7], equal_nan=True))

    x = pd.Series(["a", "c", "d", "e", "b", None, "NA"])
    ret = map(x, ref)
    assert np.all(
        np.isclose(ret, [1, 3, 4, 5, 2, np.nan, np.nan], equal_nan=True)
    )
    ret = map(x, ref, None)
    assert np.all(ret == [1, 3, 4, 5, 2, None, None])
    ret = map(x, refcall)
    assert np.all(np.isclose(ret, [1, 3, 4, 5, 2, 7, 7], equal_nan=True))

    env = pd.DataFrame({"x": x})
    ret = envp.bind_env(env).parse("map(x, ref)")
    assert np.all(
        np.isclose(ret, [1, 3, 4, 5, 2, np.nan, np.nan], equal_nan=True)
    )
    ret = envp.bind_env(env).parse("map(x, ref, None)")
    assert np.all(ret == [1, 3, 4, 5, 2, None, None])
    ret = envp.bind_env(env).parse("map(x, refcall)")
    assert np.all(np.isclose(ret, [1, 3, 4, 5, 2, 7, 7], equal_nan=True))


# %%
def test_map_with_multiple_seqs():
    def refcall(x, y):
        return x + y

    x = list(range(5))
    y = list(range(5, 10))
    env = {"x": x, "y": y}
    envp = get_envp({"refcall": refcall})
    ret = envp.bind_env(env).parse("map(x, y, refcall)")
    assert np.all(ret == list(range(5, 14, 2)))

    x = np.arange(5)
    y = np.arange(5, 10)
    env = {"x": x, "y": y}
    envp = get_envp({"refcall": refcall})
    ret = envp.bind_env(env).parse("map(x, y, refcall)")
    assert np.all(ret == list(range(5, 14, 2)))

    x = np.arange(5)
    y = np.arange(5, 10)
    env = {"x": pd.Series(x), "y": pd.Series(y)}
    envp = get_envp({"refcall": refcall})
    ret = envp.bind_env(env).parse("map(x, y, refcall)")
    assert np.all(ret == list(range(5, 14, 2)))


# %%
def _assert_sep_map(x, ref, refcall, envp=None):
    expected_ref = [(1, 2, 3, 4, 5), (1, 3), ()]
    expected_refcall = [(1, 2, 3, 4, 5, 7), (1, 3, 7), (7,)]
    expected_ref_str = ["1:2:3:4:5", "1:3", ""]
    expected_refcall_str = ["1:2:3:4:5:7", "1:3:7", "7"]

    for le, re in zip(sep_map(x, ref), expected_ref, strict=True):
        assert le == re
    assert np.all(sep_map(x, ref, ",", ":") == expected_ref_str)
    for le, re in zip(sep_map(x, refcall), expected_refcall, strict=True):
        assert le == re
    assert np.all(sep_map(x, refcall, ",", ":") == expected_refcall_str)

    if envp is not None:
        env = {"x": x}
        for le, re in zip(
            envp.bind_env(env).parse("sep_map(x, ref)"),
            expected_ref,
            strict=True,
        ):
            assert le == re
        assert np.all(
            envp.bind_env(env).parse('sep_map(x, ref, ",", ":")')
            == expected_ref_str
        )
        for le, re in zip(
            envp.bind_env(env).parse("sep_map(x, refcall)"),
            expected_refcall,
            strict=True,
        ):
            assert le == re
        assert np.all(
            envp.bind_env(env).parse('sep_map(x, refcall, ",", ":")')
            == expected_refcall_str
        )


def test_sep_map():
    ref = dict(zip("abcde", range(1, 6), strict=True))

    def refcall(ele):
        refi = dict(zip("abcde", range(1, 6), strict=True))
        return refi.get(ele, 7)

    envp = get_envp({"ref": ref, "refcall": refcall})
    # List.
    _assert_sep_map(["a,b,c,d,e,na,g", "a,g,c", "g"], ref, refcall, envp)
    # np.ndarray.
    _assert_sep_map(
        np.array(["a,b,c,d,e,na,g", "a,g,c", "g"]), ref, refcall, envp
    )
    # pd.Series.
    _assert_sep_map(
        pd.Series(["a,b,c,d,e,na,g", "a,g,c", "g"]), ref, refcall, envp
    )
    # pd.DataFrame env.
    x = pd.Series(["a,b,c,d,e,na,g", "a,g,c", "g"])
    env = pd.DataFrame({"x": x})
    for le, re in zip(
        envp.bind_env(env).parse("sep_map(x, ref)"),
        [(1, 2, 3, 4, 5), (1, 3), ()],
        strict=True,
    ):
        assert le == re
    assert np.all(
        envp.bind_env(env).parse('sep_map(x, ref, ",", ":")')
        == ["1:2:3:4:5", "1:3", ""]
    )
    for le, re in zip(
        envp.bind_env(env).parse("sep_map(x, refcall)"),
        [(1, 2, 3, 4, 5, 7), (1, 3, 7), (7,)],
        strict=True,
    ):
        assert le == re
    assert np.all(
        envp.bind_env(env).parse('sep_map(x, refcall, ",", ":")')
        == ["1:2:3:4:5:7", "1:3:7", "7"]
    )


# %%
def test_isin():
    envp = get_envp()

    # List, np.ndarry and pd.Series to check:
    # 1. If the elements in the former sequence are in the latter sequence.
    x = [1, 2, 3, 4, None, np.nan, "a"]
    y = [1, np.nan, "a", None]
    ret = isin(x, y)
    assert np.all(ret == [1, 0, 0, 0, 1, 1, 1])

    env = {"x": x, "y": y}
    ret = envp.bind_env(env).parse("isin(x, y)")
    assert np.all(ret == [1, 0, 0, 0, 1, 1, 1])

    x = np.array([1, 2, 3, 4, None, np.nan, "a"])
    y = np.array([1, np.nan, "a", None])
    ret = isin(x, y)
    assert np.all(ret == [1, 0, 0, 0, 1, 1, 1])

    env = {"x": x, "y": y}
    ret = envp.bind_env(env).parse("isin(x, y)")
    assert np.all(ret == [1, 0, 0, 0, 1, 1, 1])

    x = pd.Series([1, 2, 3, 4, None, np.nan, "a"])
    y = pd.Series([1, np.nan, "a", None])
    ret = isin(x, y)
    assert np.all(ret == [1, 0, 0, 0, 1, 1, 1])

    env = pd.DataFrame({"x": x, "y": y})
    ret = envp.bind_env(env).parse("isin(x, y)")
    assert np.all(ret == [1, 0, 0, 0, 1, 1, 1])

    # 2. If the former scalar is in the elements of the latter sequence.
    x = "a"
    y = ["a", "abc", "bc", None, np.nan, ("a", "b"), ["abc"]]
    ret = isin(x, y)
    assert np.all(ret == [1, 1, 0, 0, 0, 1, 0])

    env = {"x": x, "y": y}
    ret = envp.bind_env(env).parse("isin(x, y)")
    assert np.all(ret == [1, 1, 0, 0, 0, 1, 0])

    x = "a"
    y = np.array(
        ["a", "abc", "bc", None, np.nan, ("a", "b"), ["abc"]], dtype="O"
    )
    ret = isin(x, y)
    assert np.all(ret == [1, 1, 0, 0, 0, 1, 0])

    env = {"x": x, "y": y}
    ret = envp.bind_env(env).parse("isin(x, y)")
    assert np.all(ret == [1, 1, 0, 0, 0, 1, 0])

    x = "a"
    y = pd.Series(
        ["a", "abc", "bc", None, np.nan, ("a", "b"), ["abc"]], dtype="O"
    )
    ret = isin(x, y)
    assert np.all(ret == [1, 1, 0, 0, 0, 1, 0])

    env = {"x": x, "y": y}
    ret = envp.bind_env(env).parse("isin(x, y)")
    assert np.all(ret == [1, 1, 0, 0, 0, 1, 0])


# %%
def test_cb_fstmaxmin():
    envp = get_envp()

    # List, np.ndarray, pd.Series of numbers.
    x = [11, np.nan, 13, 4, np.nan]
    y = [np.nan, 12, 3, 14, np.nan]
    ret = cb_fst(x, y)
    assert np.all(np.isclose(ret, [11, 12, 13, 4, np.nan], equal_nan=True))
    ret = cb_max(x, y)
    assert np.all(np.isclose(ret, [11, 12, 13, 14, np.nan], equal_nan=True))
    ret = cb_min(x, y)
    assert np.all(np.isclose(ret, [11, 12, 3, 4, np.nan], equal_nan=True))

    env = {"x": x, "y": y}
    ret = envp.bind_env(env).parse("cb_fst(x, y)")
    assert np.all(np.isclose(ret, [11, 12, 13, 4, np.nan], equal_nan=True))
    ret = envp.bind_env(env).parse("cb_max(x, y)")
    assert np.all(np.isclose(ret, [11, 12, 13, 14, np.nan], equal_nan=True))
    ret = envp.bind_env(env).parse("cb_min(x, y)")
    assert np.all(np.isclose(ret, [11, 12, 3, 4, np.nan], equal_nan=True))

    x = np.array([11, np.nan, 13, 4, np.nan])
    y = np.array([np.nan, 12, 3, 14, np.nan])
    ret = cb_fst(x, y)
    assert np.all(np.isclose(ret, [11, 12, 13, 4, np.nan], equal_nan=True))
    ret = cb_max(x, y)
    assert np.all(np.isclose(ret, [11, 12, 13, 14, np.nan], equal_nan=True))
    ret = cb_min(x, y)
    assert np.all(np.isclose(ret, [11, 12, 3, 4, np.nan], equal_nan=True))

    env = {"x": x, "y": y}
    ret = envp.bind_env(env).parse("cb_fst(x, y)")
    assert np.all(np.isclose(ret, [11, 12, 13, 4, np.nan], equal_nan=True))
    ret = envp.bind_env(env).parse("cb_max(x, y)")
    assert np.all(np.isclose(ret, [11, 12, 13, 14, np.nan], equal_nan=True))
    ret = envp.bind_env(env).parse("cb_min(x, y)")
    assert np.all(np.isclose(ret, [11, 12, 3, 4, np.nan], equal_nan=True))

    x = pd.Series([11, np.nan, 13, 4, np.nan])
    y = pd.Series([np.nan, 12, 3, 14, np.nan])
    ret = cb_fst(x, y)
    assert np.all(np.isclose(ret, [11, 12, 13, 4, np.nan], equal_nan=True))
    ret = cb_max(x, y)
    assert np.all(np.isclose(ret, [11, 12, 13, 14, np.nan], equal_nan=True))
    ret = cb_min(x, y)
    assert np.all(np.isclose(ret, [11, 12, 3, 4, np.nan], equal_nan=True))

    env = pd.DataFrame({"x": x, "y": y})
    ret = envp.bind_env(env).parse("cb_fst(x, y)")
    assert np.all(np.isclose(ret, [11, 12, 13, 4, np.nan], equal_nan=True))
    ret = envp.bind_env(env).parse("cb_max(x, y)")
    assert np.all(np.isclose(ret, [11, 12, 13, 14, np.nan], equal_nan=True))
    ret = envp.bind_env(env).parse("cb_min(x, y)")
    assert np.all(np.isclose(ret, [11, 12, 3, 4, np.nan], equal_nan=True))

    # np.ndarray, pd.Series of datetime64.
    x = np.array(
        ["2021-11-11", "NaT", "2021-11-13", "2021-11-14", "NaT"], dtype="M8[D]"
    )
    y = np.array(
        ["2021-12-11", "2021-12-12", "2021-12-13", "2021-12-14", "NaT"],
        dtype="M8[D]",
    )
    ret = cb_fst(x, y)
    assert np.all(
        ret[:-1]
        == np.array(
            ["2021-11-11", "2021-12-12", "2021-11-13", "2021-11-14"],
            dtype="M8[D]",
        )
    )
    assert np.isnat(ret[-1])
    ret = cb_max(x, y)
    assert np.all(
        ret[:-1]
        == np.array(
            ["2021-12-11", "2021-12-12", "2021-12-13", "2021-12-14"],
            dtype="M8[D]",
        )
    )
    assert np.isnat(ret[-1])
    ret = cb_min(x, y)
    assert np.all(
        ret[:-1]
        == np.array(
            ["2021-11-11", "2021-12-12", "2021-11-13", "2021-11-14"],
            dtype="M8[D]",
        )
    )
    assert np.isnat(ret[-1])

    env = {"x": x, "y": y}
    ret = envp.bind_env(env).parse("cb_fst(x, y)")
    assert np.all(
        ret[:-1]
        == np.array(
            ["2021-11-11", "2021-12-12", "2021-11-13", "2021-11-14"],
            dtype="M8[D]",
        )
    )
    assert np.isnat(ret[-1])
    ret = envp.bind_env(env).parse("cb_max(x, y)")
    assert np.all(
        ret[:-1]
        == np.array(
            ["2021-12-11", "2021-12-12", "2021-12-13", "2021-12-14"],
            dtype="M8[D]",
        )
    )
    assert np.isnat(ret[-1])
    ret = envp.bind_env(env).parse("cb_min(x, y)")
    assert np.all(
        ret[:-1]
        == np.array(
            ["2021-11-11", "2021-12-12", "2021-11-13", "2021-11-14"],
            dtype="M8[D]",
        )
    )
    assert np.isnat(ret[-1])

    x = pd.Series(
        ["2021-11-11", "NaT", "2021-11-13", "2021-11-14", "NaT"],
        dtype="M8[ms]",
    )
    y = pd.Series(
        ["2021-12-11", "2021-12-12", "2021-12-13", "2021-12-14", "NaT"],
        dtype="M8[ms]",
    )
    ret = cb_fst(x, y)
    assert np.all(
        ret[:-1]
        == np.array(
            ["2021-11-11", "2021-12-12", "2021-11-13", "2021-11-14"],
            dtype="M8[D]",
        )
    )
    assert np.isnat(ret[-1])
    ret = cb_max(x, y)
    assert np.all(
        ret[:-1]
        == np.array(
            ["2021-12-11", "2021-12-12", "2021-12-13", "2021-12-14"],
            dtype="M8[D]",
        )
    )
    assert np.isnat(ret[-1])
    ret = cb_min(x, y)
    assert np.all(
        ret[:-1]
        == np.array(
            ["2021-11-11", "2021-12-12", "2021-11-13", "2021-11-14"],
            dtype="M8[D]",
        )
    )
    assert np.isnat(ret[-1])

    env = pd.DataFrame({"x": x, "y": y})
    ret = envp.bind_env(env).parse("cb_fst(x, y)")
    assert np.all(
        ret[:-1]
        == np.array(
            ["2021-11-11", "2021-12-12", "2021-11-13", "2021-11-14"],
            dtype="M8[D]",
        )
    )
    assert np.isnat(ret[-1])
    ret = envp.bind_env(env).parse("cb_max(x, y)")
    assert np.all(
        ret[:-1]
        == np.array(
            ["2021-12-11", "2021-12-12", "2021-12-13", "2021-12-14"],
            dtype="M8[D]",
        )
    )
    assert np.isnat(ret[-1])
    ret = envp.bind_env(env).parse("cb_min(x, y)")
    assert np.all(
        ret[:-1]
        == np.array(
            ["2021-11-11", "2021-12-12", "2021-11-13", "2021-11-14"],
            dtype="M8[D]",
        )
    )
    assert np.isnat(ret[-1])

    # Object for `cb_first` only.
    x = ["a", None, None, 11, np.nan, 13, 4, np.nan]
    y = [None, "b", None, np.nan, 12, 3, 14, np.nan]
    ret = cb_fst(x, y)
    for le, re in zip(
        ret, ["a", "b", None, 11, 12, 13, 4, np.nan], strict=True
    ):
        assert le == re or np.isnan(le)

    x = np.array(["a", None, None, 11, np.nan, 13, 4, np.nan])
    y = np.array([None, "b", None, np.nan, 12, 3, 14, np.nan])
    ret = cb_fst(x, y)
    for le, re in zip(
        ret, ["a", "b", None, 11, 12, 13, 4, np.nan], strict=True
    ):
        assert le == re or np.isnan(le)

    x = pd.Series(["a", None, None, 11, np.nan, 13, 4, np.nan])
    y = pd.Series([None, "b", None, np.nan, 12, 3, 14, np.nan])
    ret = cb_fst(x, y)
    for le, re in zip(
        ret, ["a", "b", None, 11, 12, 13, 4, np.nan], strict=True
    ):
        assert le == re or np.isnan(le)


# %%
def test_mon_day_itvl():
    envp = get_envp()

    # List or string, np.ndarray and pd.Series.
    x = ["2021-11-11", "NaT", "2021-11-13", "2021-11-14", "NaT"]
    y = ["2021-12-01", "2021-12-31", "2021-12-31", "2021-12-02", "NaT"]
    ret = mon_itvl(x, y)
    assert np.all(
        np.isclose(ret, [-1, np.nan, -1, -1, np.nan], equal_nan=True)
    )
    ret = day_itvl(x, y)
    assert np.all(
        np.isclose(ret, [-20, np.nan, -48, -18, np.nan], equal_nan=True)
    )

    env = {"x": x, "y": y}
    ret = envp.bind_env(env).parse("mon_itvl(x, y)")
    assert np.all(
        np.isclose(ret, [-1, np.nan, -1, -1, np.nan], equal_nan=True)
    )
    ret = envp.bind_env(env).parse("day_itvl(x, y)")
    assert np.all(
        np.isclose(ret, [-20, np.nan, -48, -18, np.nan], equal_nan=True)
    )

    x = np.array(
        ["2021-11-11", "NaT", "2021-11-13", "2021-11-14", "NaT"], dtype="M8[D]"
    )
    y = np.array(
        ["2021-12-01", "2021-12-31", "2021-12-31", "2021-12-02", "NaT"],
        dtype="M8[D]",
    )
    ret = mon_itvl(x, y)
    assert np.all(
        np.isclose(ret, [-1, np.nan, -1, -1, np.nan], equal_nan=True)
    )
    ret = day_itvl(x, y)
    assert np.all(
        np.isclose(ret, [-20, np.nan, -48, -18, np.nan], equal_nan=True)
    )

    env = {"x": x, "y": y}
    ret = envp.bind_env(env).parse("mon_itvl(x, y)")
    assert np.all(
        np.isclose(ret, [-1, np.nan, -1, -1, np.nan], equal_nan=True)
    )
    ret = envp.bind_env(env).parse("day_itvl(x, y)")
    assert np.all(
        np.isclose(ret, [-20, np.nan, -48, -18, np.nan], equal_nan=True)
    )

    x = pd.Series(
        ["2021-11-11", "NaT", "2021-11-13", "2021-11-14", "NaT"], dtype="M8[s]"
    )
    y = pd.Series(
        ["2021-12-01", "2021-12-31", "2021-12-31", "2021-12-02", "NaT"],
        dtype="M8[s]",
    )
    ret = mon_itvl(x, y)
    assert np.all(
        np.isclose(ret, [-1, np.nan, -1, -1, np.nan], equal_nan=True)
    )
    ret = day_itvl(x, y)
    assert np.all(
        np.isclose(ret, [-20, np.nan, -48, -18, np.nan], equal_nan=True)
    )

    env = pd.DataFrame({"x": x, "y": y})
    ret = envp.bind_env(env).parse("mon_itvl(x, y)")
    assert np.all(
        np.isclose(ret, [-1, np.nan, -1, -1, np.nan], equal_nan=True)
    )
    ret = envp.bind_env(env).parse("day_itvl(x, y)")
    assert np.all(
        np.isclose(ret, [-20, np.nan, -48, -18, np.nan], equal_nan=True)
    )

    # Empty sequence.
    ret = mon_itvl([], [])
    assert len(ret) == 0
    ret = day_itvl([], [])
    assert len(ret) == 0

    ret = envp.bind_env(env).parse("mon_itvl([], [])")
    assert len(ret) == 0
    ret = envp.bind_env(env).parse("day_itvl([], [])")
    assert len(ret) == 0


# %%
def test_busiday():
    envp = get_envp()

    # List of string, np.ndarray and pd.Series.
    x = ["2021-11-11", "2025-01-01", "2021-11-13", "2021-11-14", "NaT"]
    ret = is_busiday(x)
    assert np.all(ret == [1, 0, 0, 0, 0])
    ret = not_busiday(x)
    assert np.all(ret == [0, 1, 1, 1, 0])

    env = {"x": x}
    ret = envp.bind_env(env).parse("is_busiday(x)")
    assert np.all(ret == [1, 0, 0, 0, 0])
    ret = envp.bind_env(env).parse("not_busiday(x)")
    assert np.all(ret == [0, 1, 1, 1, 0])

    x = np.array(
        ["2021-11-11", "2025-01-01", "2021-11-13", "2021-11-14", "NaT"]
    )
    ret = is_busiday(x)
    assert np.all(ret == [1, 0, 0, 0, 0])
    ret = not_busiday(x)
    assert np.all(ret == [0, 1, 1, 1, 0])

    env = {"x": x}
    ret = envp.bind_env(env).parse("is_busiday(x)")
    assert np.all(ret == [1, 0, 0, 0, 0])
    ret = envp.bind_env(env).parse("not_busiday(x)")
    assert np.all(ret == [0, 1, 1, 1, 0])

    x = pd.Series(
        ["2021-11-11", "2025-01-01", "2021-11-13", "2021-11-14", "NaT"]
    )
    ret = is_busiday(x)
    assert np.all(ret == [1, 0, 0, 0, 0])
    ret = not_busiday(x)
    assert np.all(ret == [0, 1, 1, 1, 0])

    env = pd.DataFrame({"x": x})
    ret = envp.bind_env(env).parse("is_busiday(x)")
    assert np.all(ret == [1, 0, 0, 0, 0])
    ret = envp.bind_env(env).parse("not_busiday(x)")
    assert np.all(ret == [0, 1, 1, 1, 0])

    # Single string and empty sequence.
    ret = is_busiday("2021-11-11")
    assert ret
    ret = is_busiday([])
    assert len(ret) == 0

    ret = envp.bind_env(env).parse('is_busiday("2021-11-11")')
    assert ret
    ret = envp.bind_env(env).parse("is_busiday([])")
    assert len(ret) == 0


# %%
def test_gethour():
    envp = get_envp()

    # List of string, np.ndarray and pd.Series.
    x = ["2021-11-11T11:11:12", "2025-01-01T13:12:12", "NaT"]
    ret = get_hour(x)
    assert np.all(np.isclose(ret, [11, 13, np.nan], equal_nan=True))

    env = {"x": x}
    ret = envp.bind_env(env).parse("get_hour(x)")
    assert np.all(np.isclose(ret, [11, 13, np.nan], equal_nan=True))

    x = np.array(["2021-11-11T11:11:12", "2025-01-01T13:12:12", "NaT"])
    ret = get_hour(x)
    assert np.all(np.isclose(ret, [11, 13, np.nan], equal_nan=True))

    env = {"x": x}
    ret = envp.bind_env(env).parse("get_hour(x)")
    assert np.all(np.isclose(ret, [11, 13, np.nan], equal_nan=True))

    x = np.array(
        ["2021-11-11T11:11:12", "2025-01-01T13:12:12", "NaT"], dtype="M8[s]"
    )
    ret = get_hour(x)
    assert np.all(np.isclose(ret, [11, 13, np.nan], equal_nan=True))

    env = pd.DataFrame({"x": x})
    ret = envp.bind_env(env).parse("get_hour(x)")
    assert np.all(np.isclose(ret, [11, 13, np.nan], equal_nan=True))

    x = pd.Series(["2021-11-11T11:11:12", "2025-01-01T13:12:12", "NaT"])
    ret = get_hour(x)
    assert np.all(np.isclose(ret, [11, 13, np.nan], equal_nan=True))

    env = pd.DataFrame({"x": x})
    ret = envp.bind_env(env).parse("get_hour(x)")
    assert np.all(np.isclose(ret, [11, 13, np.nan], equal_nan=True))

    # Empty sequence.
    ret = get_hour([])
    assert len(ret) == 0
    ret = envp.bind_env(env).parse("get_hour([])")
    assert len(ret) == 0

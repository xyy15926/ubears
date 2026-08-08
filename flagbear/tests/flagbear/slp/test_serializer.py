#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_serializer.py
#   Author: xyy15926
#   Created: 2026-04-22 17:12:09
#   Updated: 2026-05-20 22:21:40
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations

import io
import json

import numpy as np
import pandas as pd
import pytest

if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, force=True)
    from importlib import reload

    from flagbear.slp import serializer

    reload(serializer)

from flagbear.slp import serializer
from flagbear.slp.serializer import deserialize, serialize


# %%
def test_serializer_None_zero_bytes():
    for type_ in ["numpy", "pddf_csv", "pddf_feather", "pddf_parquet"]:
        bytes_, _ = serialize(None, type_)
        assert bytes_ == b""
        data = deserialize(b"", type_)
        assert data is None

    for type_ in ["json", "pickle"]:
        data = deserialize(b"", type_)
        assert data is None


# %%
def test_serialize_json():
    dict_ = {
        "a": 1,
        "b": 2,
    }
    bytes_, type_ = serialize(dict_)
    assert type_ == "json"
    assert bytes_ == json.dumps(
        dict_, separators=(",", ":"), ensure_ascii=False
    ).encode("utf8")
    assert dict_ == deserialize(bytes_, type_)

    list_ = [1, 2, "编译", 2.9]
    bytes_, type_ = serialize(list_)
    assert bytes_ == json.dumps(
        list_, separators=(",", ":"), ensure_ascii=False
    ).encode("utf8")
    assert list_ == deserialize(bytes_, type_)


# %%
def test_serialize_numpy():
    nda = np.random.rand(4, 4)
    bytes_, type_ = serialize(nda)
    buffer = io.BytesIO()
    assert type_ == "numpy"
    np.save(buffer, nda)
    assert bytes_ == buffer.getvalue()
    assert np.all(nda == deserialize(bytes_, type_))


@pytest.mark.skip("Too expensive to serialize and compress large ndarray.")
def test_serialize_numpy_with_zlib():
    import zlib

    nda = np.random.rand(4000, 4000)
    bytes_, type_ = serialize(nda)
    buffer = io.BytesIO()
    assert type_ == "numpy"
    np.save(buffer, nda)
    assert bytes_ == zlib.compress(buffer.getvalue())
    assert np.all(nda == deserialize(bytes_, type_))


# %%
def test_serialize_pddf_csv_and_pickle():
    df = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [1, 2.0, 3.0],
            "c": ["c", "编译", "c"],
            "d": [1, np.nan, 2],
        }
    )
    bytes_, type_ = serialize(df)
    assert type_ == "pddf_feather"
    loaded = deserialize(bytes_, type_)
    pd.testing.assert_frame_equal(df, loaded)

    df2 = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [1, 2.0, 3.0],
            "c": ["c", "编译", "c"],
            "d": [1, np.nan, "d"],
        }
    )
    bytes_, type_ = serialize(df2)
    assert type_ == "pickle" or type_ == "pddf_feather"
    loaded = deserialize(bytes_, type_)
    pd.testing.assert_frame_equal(df2, loaded)

    # CSV should be used with any column with `dtype=object`, namely consist of
    # mixed type.
    # Force to serialize the DataFrame with CSV.
    bytes_, type_ = serialize(df2, "pddf_csv")
    assert bytes_ == df2.to_csv(None).encode("utf8")
    loaded = deserialize(bytes_, "pddf_csv")
    # But the DataFrame can't be recovered soundly.
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(df2, loaded, check_dtype=False)

#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_geoenc.py
#   Author: xyy15926
#   Created: 2024-07-24 21:13:00
#   Updated: 2026-04-03 13:38:31
#   Description:
# ---------------------------------------------------------

# %%
import pytest
from pytest import mark

if __name__ == "__main__":
    from importlib import reload
    from dirtbear.locale import geoenc

    reload(geoenc)

from dirtbear.locale.geoenc import CHNGovEncoder


# %%
def test_get_gregion_toker():
    sent = "中国北京市深圳市开平区UBeaRLy"
    toker = CHNGovEncoder.get_gregion_toker()

    # Precise mode.
    tok_gen = toker.cut(sent)
    # Return-all mode, where the tokens could overlap with others.
    tok_gen_all = toker.cut(sent, cut_all=True)
    toks = toker.lcut(sent)
    toks_all = toker.lcut(sent, cut_all=True)

    assert list(tok_gen) == toks
    assert list(tok_gen_all) == toks_all
    assert len(toks) < len(toks_all)

    # For search engine as search-key, like return-all mode.
    tok_gen_se = toker.cut_for_search(sent)
    tok_se = toker.lcut_for_search(sent)

    assert list(tok_gen_se) == tok_se


# %%
def test_geo_encode():
    geoenc = CHNGovEncoder()
    # ptoker = geoenc.ptoker
    addr = "广东深圳龙华区观澜壹方城2楼"
    with_all = geoenc.encode(addr)
    addr = "深圳龙华区观澜壹方城2楼"
    miss_prov = geoenc.encode(addr)
    addr = "广东龙华区观澜壹方城2楼"
    miss_city = geoenc.encode(addr)

    assert with_all == miss_prov
    assert with_all == miss_city

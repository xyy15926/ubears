#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_str_width.py
#   Author: xyy15926
#   Created: 2026-08-12 14:00:00
#   Updated: 2026-08-12 14:00:00
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations

if __name__ == "__main__":
    from importlib import reload

    from dirtbear.docer.mdnorm import str_width

    reload(str_width)

from dirtbear.docer.mdnorm.str_width import str_width


# %%
def test_str_width_ascii():
    assert str_width("hello") == 5
    assert str_width("abc") == 3
    assert str_width("") == 0
    assert str_width("a b c") == 5


def test_str_width_cjk():
    assert str_width("你好") == 4
    assert str_width("中文") == 4
    assert str_width("a你b") == 4
    assert str_width("hello你好") == 9


def test_str_width_tab():
    assert str_width("\t") == 4
    assert str_width("a\t") == 5
    assert str_width("\tab") == 6


def test_str_width_mixed():
    assert str_width("a\t你") == 7
    assert str_width("\t你") == 6

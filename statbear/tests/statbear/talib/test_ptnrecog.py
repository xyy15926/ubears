#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_candlestick.py
#   Author: xyy15926
#   Created: 2024-11-25 13:59:19
#   Updated: 2025-01-21 15:07:22
#   Description:
# ---------------------------------------------------------

# %%
import pytest

try:
    import talib as ta
except ImportError:
    pytestmark = pytest.mark.skip(reason="TA-Lib uninstalled.")
import numpy as np

if __name__ == "__main__":
    from importlib import reload

    from statbear.talib import momentum, overlap, ptnrecog

    reload(overlap)
    reload(momentum)
    reload(ptnrecog)

from statbear.talib.ptnrecog import (
    abandoned_baby,
    advance_block,
    belthold,
    black_crows3,
    break_away,
    closing_marubozu,
    conceal_baby_swall,
    counter_attack,
    crows2,
    darkcloud_cover,
    doji,
    doji_star,
    dragonfly_doji,
    engulfing,
    evening_doji_star,
    evening_star,
    gap_side_side_white,
    gravestone_doji,
    hammer,
    hangingman,
    inside3,
    line_strike3,
    outside3,
    stars_insouth3,
    white_soldiers3,
)


# %%
def mock_data(n: int = 100):
    dt = np.random.rand(n, 3)
    dt.sort(axis=1)
    return dt[:, 2], dt[:, 1], dt[:, 0]


def mock_data_all(n: int = 100):
    high, close, low = mock_data(n)
    high_, open_, low_ = mock_data(n)
    high = np.where(high > high_, high, high_)
    low = np.where(low < low_, low, low_)
    volume = np.random.rand(n) * 10000

    return open_, high, low, close, volume


TEST_N = 1000


# %%
def test_belthold():
    for _i in range(TEST_N):
        open_, high, low, close, _volume = mock_data_all()
        val = belthold(open_, high, low, close)
        ta_val = ta.CDLBELTHOLD(open_, high, low, close)

        assert np.all(np.isclose(val, ta_val, equal_nan=True))


# %%
def test_closing_marubozu():
    for _i in range(TEST_N):
        open_, high, low, close, _volume = mock_data_all()
        val = closing_marubozu(open_, high, low, close)
        ta_val = ta.CDLCLOSINGMARUBOZU(open_, high, low, close)

        assert np.all(np.isclose(val, ta_val, equal_nan=True))


# %%
def test_doji():
    for _i in range(TEST_N):
        open_, high, low, close, _volume = mock_data_all()
        val = doji(open_, high, low, close)
        ta_val = ta.CDLDOJI(open_, high, low, close)

        assert np.all(np.isclose(val, ta_val, equal_nan=True))


# %%
def test_dragonfly_doji():
    for _i in range(TEST_N):
        open_, high, low, close, _volume = mock_data_all()
        val = dragonfly_doji(open_, high, low, close)
        ta_val = ta.CDLDRAGONFLYDOJI(open_, high, low, close)

        assert np.all(np.isclose(val, ta_val, equal_nan=True))


# %%
def test_gravestone_doji():
    for _i in range(TEST_N):
        open_, high, low, close, _volume = mock_data_all()
        val = gravestone_doji(open_, high, low, close)
        ta_val = ta.CDLGRAVESTONEDOJI(open_, high, low, close)

        assert np.all(np.isclose(val, ta_val, equal_nan=True))


# %%
def test_counter_attack():
    for _i in range(TEST_N):
        open_, high, low, close, _volume = mock_data_all()
        val = counter_attack(open_, high, low, close)
        ta_val = ta.CDLCOUNTERATTACK(open_, high, low, close)

        assert np.all(np.isclose(val, ta_val, equal_nan=True))


# %%
def test_darkcloud_cover():
    for _i in range(TEST_N):
        open_, high, low, close, _volume = mock_data_all()
        val = darkcloud_cover(open_, high, low, close, 0.5)
        ta_val = ta.CDLDARKCLOUDCOVER(open_, high, low, close, 0.5)

        assert np.all(np.isclose(val, ta_val, equal_nan=True))


# %%
def test_doji_star():
    for _i in range(TEST_N):
        open_, high, low, close, _volume = mock_data_all()
        val = doji_star(open_, high, low, close)
        ta_val = ta.CDLDOJISTAR(open_, high, low, close)

        assert np.all(np.isclose(val, ta_val, equal_nan=True))


# %%
def test_engulfing():
    for _i in range(TEST_N):
        open_, high, low, close, _volume = mock_data_all()
        val = engulfing(open_, high, low, close)
        ta_val = ta.CDLENGULFING(open_, high, low, close)

        assert np.all(np.isclose(val, ta_val, equal_nan=True))


# %%
def test_hammer():
    for _i in range(TEST_N):
        open_, high, low, close, _volume = mock_data_all()
        val = hammer(open_, high, low, close)
        ta_val = ta.CDLHAMMER(open_, high, low, close)

        assert np.all(np.isclose(val, ta_val, equal_nan=True))


# %%
def test_hangingman():
    for _i in range(TEST_N):
        open_, high, low, close, _volume = mock_data_all()
        val = hangingman(open_, high, low, close)
        ta_val = ta.CDLHANGINGMAN(open_, high, low, close)

        assert np.all(np.isclose(val, ta_val, equal_nan=True))


# %%
def test_crows2():
    for _i in range(TEST_N):
        open_, high, low, close, _volume = mock_data_all()
        val = crows2(open_, high, low, close)
        ta_val = ta.CDL2CROWS(open_, high, low, close)

        assert np.all(np.isclose(val, ta_val, equal_nan=True))


# %%
def test_blackcrows3():
    for _i in range(TEST_N):
        open_, high, low, close, _volume = mock_data_all()
        val = black_crows3(open_, high, low, close)
        ta_val = ta.CDL3BLACKCROWS(open_, high, low, close)

        assert np.all(np.isclose(val, ta_val, equal_nan=True))


# %%
def test_inside3():
    for _i in range(TEST_N):
        open_, high, low, close, _volume = mock_data_all()
        val = inside3(open_, high, low, close)
        ta_val = ta.CDL3INSIDE(open_, high, low, close)

        assert np.all(np.isclose(val, ta_val, equal_nan=True))


# %%
def test_outside3():
    for _i in range(TEST_N):
        open_, high, low, close, _volume = mock_data_all()
        val = outside3(open_, high, low, close)
        ta_val = ta.CDL3OUTSIDE(open_, high, low, close)

        assert np.all(np.isclose(val, ta_val, equal_nan=True))


# %%
def test_starsinsouth3():
    for _i in range(TEST_N):
        open_, high, low, close, _volume = mock_data_all()
        val = stars_insouth3(open_, high, low, close)
        ta_val = ta.CDL3STARSINSOUTH(open_, high, low, close)

        assert np.all(np.isclose(val, ta_val, equal_nan=True))


# %%
def test_whitesoldiers3():
    for _i in range(TEST_N):
        open_, high, low, close, _volume = mock_data_all()
        val = white_soldiers3(open_, high, low, close)
        ta_val = ta.CDL3WHITESOLDIERS(open_, high, low, close)

        assert np.all(np.isclose(val, ta_val, equal_nan=True))


# %%
def test_line_strike_3():
    for _i in range(TEST_N):
        open_, high, low, close, _volume = mock_data_all()
        val = line_strike3(open_, high, low, close)
        ta_val = ta.CDL3LINESTRIKE(open_, high, low, close)

        assert np.all(np.isclose(val, ta_val, equal_nan=True))


# %%
def test_abandoned_baby():
    for _i in range(TEST_N):
        open_, high, low, close, _volume = mock_data_all()
        val = abandoned_baby(open_, high, low, close)
        ta_val = ta.CDLABANDONEDBABY(open_, high, low, close)

        assert np.all(np.isclose(val, ta_val, equal_nan=True))


# %%
def test_evening_doji_star():
    for _i in range(TEST_N):
        open_, high, low, close, _volume = mock_data_all()
        val = evening_doji_star(open_, high, low, close)
        ta_val = ta.CDLEVENINGDOJISTAR(open_, high, low, close)

        assert np.all(np.isclose(val, ta_val, equal_nan=True))


# %%
def test_evening_star():
    for _i in range(TEST_N):
        open_, high, low, close, _volume = mock_data_all()
        val = evening_star(open_, high, low, close)
        ta_val = ta.CDLEVENINGSTAR(open_, high, low, close)

        assert np.all(np.isclose(val, ta_val, equal_nan=True))


# %%
def test_gap_side_side_white():
    for _i in range(TEST_N):
        open_, high, low, close, _volume = mock_data_all()
        val = gap_side_side_white(open_, high, low, close)
        ta_val = ta.CDLGAPSIDESIDEWHITE(open_, high, low, close)

        assert np.all(np.isclose(val, ta_val, equal_nan=True))


# %%
def test_conceal_baby_swall():
    for _i in range(TEST_N):
        open_, high, low, close, _volume = mock_data_all()
        val = conceal_baby_swall(open_, high, low, close)
        ta_val = ta.CDLCONCEALBABYSWALL(open_, high, low, close)

        assert np.all(np.isclose(val, ta_val, equal_nan=True))


# %%
def test_break_away():
    for _i in range(TEST_N):
        open_, high, low, close, _volume = mock_data_all()
        val = break_away(open_, high, low, close)
        ta_val = ta.CDLBREAKAWAY(open_, high, low, close)

        assert np.all(np.isclose(val, ta_val, equal_nan=True))


# %%
def test_advance_block():
    for _i in range(TEST_N):
        open_, high, low, close, _volume = mock_data_all()
        val = advance_block(open_, high, low, close)
        ta_val = ta.CDLADVANCEBLOCK(open_, high, low, close)

        assert np.all(np.isclose(val, ta_val, equal_nan=True))

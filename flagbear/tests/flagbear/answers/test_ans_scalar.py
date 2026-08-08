#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_ans_scalar.py
#   Author: xyy15926
#   Created: 2024-01-12 14:42:57
#   Updated: 2026-06-07 20:10:06
#   Description:
# ---------------------------------------------------------

# %%

if __name__ == "__main__":
    from importlib import reload

    from flagbear.answers import ans_scalar

    reload(ans_scalar)

from flagbear.answers.ans_scalar import (
    euclid_gcd,
    euclid_lcm,
    floyd_steinberg,
    poly_eval,
    qpower,
    russian_mul,
)


# %%
def test_qpower():
    assert qpower(2323, 23423) == 2323**23423


def test_eculid_gcd():
    assert euclid_gcd(56, 49) == 7
    assert euclid_gcd(49, 56) == 7
    assert euclid_lcm(49, 56) == 49 * 56 / 7


def test_poly_eval():
    p = [4, 2, 19, 3.4, 234, 1.3]
    x = 5
    ret = 0
    for idx, pp in enumerate(p):
        ret += pp * (x**idx)
    assert poly_eval(p, x) == ret


def russian_mul():
    assert russian_mul(24324, 243234) == 24324 * 243234


# %%
def test_floyd_steinberg():
    import numpy as np

    pixels = np.random.randint(0, 256, (100, 100))
    pixels = floyd_steinberg(pixels)
    assert (pixels == 255).sum() + (pixels == 0).sum() == 1e4

    pixels = np.random.randint(0, 256, (100, 100))
    arr = pixels.tolist()
    pixels = np.asarray(floyd_steinberg(arr))
    assert (pixels == 255).sum() + (pixels == 0).sum() == 1e4

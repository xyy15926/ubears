#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: ans_scalar.py
#   Author: xyy15926
#   Created: 2024-01-12 14:21:00
#   Updated: 2026-06-07 20:16:35
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    import numpy as np
import logging

Num = TypeVar("Number")

# %%
logger = logging.getLogger(__name__)


# %%
def euclid_gcd(x: int, y: int) -> int:
    """Euclid GCD.

    gcd(x, y) = gcd(y, mod(x, y))
    """
    while x != 0 and y != 0:
        tmp = x % y
        x = y
        y = tmp
    return x or y


def euclid_lcm(x: int, y: int) -> int:
    """Euclid LCM.

    lcm(x, y) = x * y / gcd(x, y)
    """
    return x * y / euclid_gcd(x, y)


# %%
def poly_eval(p: list[Num], x: Num) -> Num:
    """Polynomial evaluation.

      p0 + p1 * x + ... + p4 * x^4 + p5 * x^5
    = p0 + p1 * x + ... + (p4 + p5 * x) * x^4
    = ...
    """
    ans = 0
    for i in p[::-1]:
        ans = ans * x + i
    return ans


# %%
def bmultiply(x: int, y: int):
    pass


# %%
def russian_mul(x: int, y: int):
    """Russian mutilply.

    n       m
    ------------------------------------------
    50      65
    25      130         130
    12      260
     6      520
     3      1040        1040
     1      2080        2080
    ------------------------------------------
    sum=130+1040+2080=3205
    """
    ans = 0
    while x > 1:
        if x & 1 == 1:
            ans += y
        x >>= 1
        y <<= 1
    return ans


# %%
def qpower(x: int, n: int):
    """Quick power.

    1. Assuming n = 0b10101
    2. x^n = x^(0b10000 + 0b100 + 0b1)
           = x^(0b10000) * x^(0b100) * x^(0b1)
    3. x^(10000) = x^(0b1000)^2
    Namely, muliply the power result of corresponsible `1` in `n`.
    """
    ans = 1
    while n > 0:
        if n & 1 == 1:
            ans *= x
        x *= x
        n >>= 1
    return ans


# %%
def floyd_steinberg(
    pixels: list[list[int]] | np.ndarray,
):
    """Floyd-Steinberg algorithm to compress grey-image into bool-image.

    Compress one pixel to 255 or 0 and then disperse error to surrounding
    pixels.

                    cur             7/16 * err
    3/16 * err      5/16 * err      1/16 * err

    1. The updation is inplace.
    2. The correlations are just gotten from experience.

    Ref:
    -----------------------
    - https://www.zhihu.com/question/68411978/answer/2046754950697564025

    Params:
    -----------------------
    pixels: 2D array of int8 representing a grey image.

    Return:
    -----------------------
    2D array of 0 or 255 representing a bool-image.
    """
    rown = len(pixels)
    coln = len(pixels[0])
    for yidx in range(rown - 1):
        for xidx in range(coln - 1):
            old = pixels[yidx][xidx]
            new = 255 if old > 128 else 0
            pixels[yidx][xidx] = new
            err = old - new

            # Disperse errors to surrounding pixels.
            pixels[yidx][xidx+1] += err * 7 // 16
            pixels[yidx+1][xidx-1] += err * 3 // 16
            pixels[yidx+1][xidx] += err * 5 // 16
            pixels[yidx+1][xidx+1] += err * 1 // 16
        else:
            xidx += 1
            old = pixels[yidx][xidx]
            new = 255 if old > 128 else 0
            pixels[yidx][xidx] = new
    else:
        yidx += 1
        for xidx in range(coln - 1):
            old = pixels[yidx][xidx]
            new = 255 if old > 128 else 0
            pixels[yidx][xidx] = new
            err = old - new
        else:
            xidx += 1
            old = pixels[yidx][xidx]
            new = 255 if old > 128 else 0
            pixels[yidx][xidx] = new

    return pixels

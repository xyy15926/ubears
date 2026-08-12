#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: str_width.py
#   Author: mimo-v2.5
#   Created: 2026-08-12 13:59:19
#   Updated: 2026-08-12 15:07:27
#   Description:
#     String display width calculation utilities.
# ---------------------------------------------------------
"""String display width calculation utilities.

Provides functions for calculating the display width of strings,
accounting for CJK characters and tab stops.
"""

# %%
from __future__ import annotations

import unicodedata


# %%
def str_width(s: str) -> int:
    """Calculate the display width of a string.

    This function accounts for different character display widths:
    - ASCII characters have width 1
    - CJK (Chinese, Japanese, Korean) characters have width 2
    - Tab characters have width 4

    Params:
    ------------------------
    s: String to calculate width for.

    Returns:
    ------------------------
    int: Display width of the string.
    """
    width = 0
    for c in s:
        if c == "\t":
            width += 4
        elif unicodedata.east_asian_width(c) in ("F", "W"):
            width += 2
        else:
            width += 1
    return width

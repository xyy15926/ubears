#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: __init__.py
#   Author: xyy15926
#   Created: 2026-07-03 10:00:00
#   Updated: 2026-07-03 10:00:00
#   Description: Const subpackage - pattern matching constants
# ---------------------------------------------------------

from flagbear.const.patterns import REGEXS
from flagbear.const.tokens import (
    LEX_ENDFLAG,
    LEX_RESERVEDS,
    LEX_SKIPS,
    LEX_TOKEN_SPECS,
    LEX_TOKEN_PRECS,
)
from flagbear.const.prods import (
    SYN_STARTSYM,
    SYN_ARITH_PRODS,
    SYN_EXPR_PRODS,
)

__all__ = [
    "REGEXS",
    "LEX_ENDFLAG",
    "LEX_RESERVEDS",
    "LEX_SKIPS",
    "LEX_TOKEN_SPECS",
    "LEX_TOKEN_PRECS",
    "SYN_STARTSYM",
    "SYN_ARITH_PRODS",
    "SYN_EXPR_PRODS",
]
